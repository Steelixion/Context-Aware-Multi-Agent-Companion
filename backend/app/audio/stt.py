"""Microphone listener: energy VAD -> faster-whisper (multilingual) + speech emotion.
Runs in a background thread; publishes {"text","lang","emotion"} on the 'speech' topic."""
from __future__ import annotations

import asyncio
import queue
import threading
import time

import numpy as np

from ..bus import Bus
from .emotion import EmotionClassifier

SR = 16000


class SpeechListener:
    def __init__(self, bus: Bus, loop: asyncio.AbstractEventLoop, arbiter, model_size: str,
                 emotion: EmotionClassifier):
        self.bus, self.loop, self.arbiter = bus, loop, arbiter
        self.model_size, self.emotion = model_size, emotion
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="stt", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        try:
            import sounddevice as sd
            from faster_whisper import WhisperModel
            model = WhisperModel(self.model_size, device="auto", compute_type="int8")
        except Exception as e:  # noqa: BLE001
            print(f"[stt] disabled: {e}")
            return
        q: queue.Queue = queue.Queue()
        block = int(SR * 0.05)

        def cb(indata, frames, t, status):
            q.put(indata[:, 0].copy())

        voiced: list[np.ndarray] = []
        silence, speaking_started = 0.0, False
        with sd.InputStream(samplerate=SR, channels=1, blocksize=block, callback=cb):
            print("[stt] listening")
            while not self._stop.is_set():
                try:
                    chunk = q.get(timeout=0.2)
                except queue.Empty:
                    continue
                if self.arbiter.speaking:       # don't transcribe our own voice
                    voiced, speaking_started, silence = [], False, 0.0
                    continue
                loud = float(np.sqrt(np.mean(chunk ** 2))) > 0.015
                if loud:
                    speaking_started, silence = True, 0.0
                if speaking_started:
                    voiced.append(chunk)
                    silence += 0 if loud else 0.05
                    if silence > 0.8:
                        audio = np.concatenate(voiced)
                        voiced, speaking_started, silence = [], False, 0.0
                        if len(audio) > SR * 0.4:
                            self._handle(model, audio)

    def _handle(self, model, audio: np.ndarray) -> None:
        segs, info = model.transcribe(audio, beam_size=1, vad_filter=True)
        text = " ".join(s.text.strip() for s in segs).strip()
        if not text:
            return
        emo = self.emotion.classify(audio, SR)
        msg = dict(text=text, lang=info.language, emotion=emo, source="voice", ts=time.time())
        self.loop.call_soon_threadsafe(self.bus.publish, "speech", msg)
