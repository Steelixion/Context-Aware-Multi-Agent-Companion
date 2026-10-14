"""Audio arbiter: single speaker, priority queue, hard preemption for safety warnings."""
from __future__ import annotations

import asyncio
import itertools
import time
from dataclasses import dataclass

from ..bus import Bus
from ..state import Session
from .tts import TTS

SAFETY, CHAT = 0, 5


@dataclass(order=True)
class _Item:
    priority: int
    seq: int
    text: str = ""
    lang: str = "en"


class AudioArbiter:
    def __init__(self, bus: Bus, session: Session, tts: TTS, enabled: bool = True,
                 chat_mute_seconds: float = 10.0):
        self.bus, self.session, self.tts, self.enabled = bus, session, tts, enabled
        self.chat_mute = chat_mute_seconds
        self.q: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.seq = itertools.count()
        self.cache: dict[tuple, tuple] = {}
        self.speaking = False
        self._cur_priority = CHAT
        self._task: asyncio.Task | None = None
        try:
            import sounddevice as sd  # noqa: F401
            self.have_audio = enabled
        except Exception:  # noqa: BLE001
            self.have_audio = False

    async def start(self) -> None:
        self._task = asyncio.create_task(self._worker(), name="audio-arbiter")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()

    async def prerender(self, texts: dict[str, str]) -> None:
        """Cache safety warnings as PCM at startup so warning needs no TTS at alert time."""
        if not self.have_audio:
            return
        loop = asyncio.get_running_loop()
        for lang, text in texts.items():
            res = await loop.run_in_executor(None, self.tts.synth, text, lang)
            if res:
                self.cache[(text, lang)] = res

    # ---- public API ----
    async def say(self, text: str, lang: str = "en") -> None:
        if self.session.safety_active:
            return  # chat is muted
        self.q.put_nowait(_Item(CHAT, next(self.seq), text, lang))

    async def preempt(self, text: str, lang: str = "en") -> None:
        """Mute all chat, stop current playback immediately, speak the warning first."""
        self.session.safety_hold_until = time.time() + self.chat_mute
        self._drain(keep_safety=True)
        if self.speaking and self._cur_priority > SAFETY:
            self._stop_playback()
        self.q.put_nowait(_Item(SAFETY, next(self.seq), text, lang))

    def flush_chat(self) -> None:
        self._drain(keep_safety=True)
        if self.speaking and self._cur_priority > SAFETY:
            self._stop_playback()

    # ---- internals ----
    def _drain(self, keep_safety: bool) -> None:
        keep = []
        while not self.q.empty():
            it = self.q.get_nowait()
            if keep_safety and it.priority == SAFETY:
                keep.append(it)
        for it in keep:
            self.q.put_nowait(it)

    def _stop_playback(self) -> None:
        if self.have_audio:
            import sounddevice as sd
            sd.stop()
        self._interrupted = True

    async def _worker(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            it: _Item = await self.q.get()
            if it.priority > SAFETY and self.session.safety_active:
                continue
            self._cur_priority, self.speaking, self._interrupted = it.priority, True, False
            self.bus.publish("audio", dict(speaking=True, text=it.text, priority=it.priority))
            try:
                pcm = self.cache.get((it.text, it.lang))
                if pcm is None and self.have_audio:
                    pcm = await loop.run_in_executor(None, self.tts.synth, it.text, it.lang)
                if pcm is not None and self.have_audio:
                    await loop.run_in_executor(None, self._play, *pcm)
                else:  # no audio device/TTS: emulate speaking time so state machine stays honest
                    await self._sleep_interruptible(min(len(it.text) * 0.055, 6.0))
            except Exception as e:  # noqa: BLE001
                print(f"[audio] playback error: {e}")
            finally:
                self.speaking = False
                self.bus.publish("audio", dict(speaking=False, text="", priority=it.priority))

    async def _sleep_interruptible(self, secs: float) -> None:
        end = time.time() + secs
        while time.time() < end and not self._interrupted:
            await asyncio.sleep(0.05)

    @staticmethod
    def _play(data, sr) -> None:
        import sounddevice as sd
        sd.play(data, sr)
        sd.wait()
