"""Real-time 'microphone' for the app: streams 16 kHz int16 PCM to ws://.../ws/audio in 50 ms
messages on a monotonic deadline schedule (never faster than real time). Between utterances
it sends silence, or the scenario's continuous noise bed. Optionally mixes in the app's own
speaker output (from the loopback capture) to simulate cabin echo for the self-hearing test.

Speech is sent as it would be spoken, chunk by chunk, so the app's VAD sees the words arrive
at natural speed, with natural pauses, exactly as from a microphone.
"""
from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np

from .events import EventLog
from .noise import NoiseGen, at_rms

SR = 16000


class AudioFeed:
    def __init__(self, url: str, log: EventLog, chunk_ms: int = 50):
        self.url, self.log = url, log
        self.chunk = int(SR * chunk_ms / 1000)
        self.period = chunk_ms / 1000
        self._queue: deque = deque()               # (audio_id, samples, meta) waiting to start
        self._current: tuple | None = None         # (audio_id, samples, pos, meta)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.noise: NoiseGen | None = None
        self.noise_rms = 0.0
        self.echo_source = None                    # callable(n) -> float32 samples or None
        self.echo_gain = 0.0
        self.sent_chunks = 0
        self.late_chunks = 0
        self.max_lateness = 0.0
        self.error: str | None = None
        self.connected = threading.Event()
        self.idle = threading.Event()
        self.idle.set()
        self._thread = threading.Thread(target=self._run, name="audio-feed", daemon=True)

    def set_noise_bed(self, kind: str | None, noise_rms: float, seed: int) -> None:
        with self._lock:
            self.noise = NoiseGen(kind, seed) if kind else None
            self.noise_rms = noise_rms

    def set_echo(self, source, gain_db: float | None) -> None:
        with self._lock:
            self.echo_source = source
            self.echo_gain = 0.0 if gain_db is None else 10 ** (gain_db / 20)

    def say(self, audio_id: str, samples: np.ndarray, **meta) -> None:
        """Queue an utterance; it starts at the next chunk boundary (or after the current one)."""
        with self._lock:
            self._queue.append((audio_id, np.asarray(samples, np.float32), meta))
            self.idle.clear()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(5)

    def _next_chunk(self) -> tuple[np.ndarray, list[tuple[str, str, dict]]]:
        marks = []
        out = np.zeros(self.chunk, np.float32)
        with self._lock:
            filled = 0
            while filled < self.chunk:
                if self._current is None:
                    if not self._queue:
                        break
                    aid, s, meta = self._queue.popleft()
                    self._current = (aid, s, 0, meta)
                    marks.append(("speech_start", aid, meta | {"offset_samples": filled}))
                aid, s, pos, meta = self._current
                take = min(self.chunk - filled, len(s) - pos)
                out[filled:filled + take] = s[pos:pos + take]
                filled += take
                pos += take
                if pos >= len(s):
                    marks.append(("speech_end", aid, meta | {"offset_samples": filled}))
                    self._current = None
                else:
                    self._current = (aid, s, pos, meta)
            if self._current is None and not self._queue:
                self.idle.set()
            if self.noise is not None:
                out += at_rms(self.noise.generate(self.chunk), self.noise_rms)
            if self.echo_source is not None and self.echo_gain > 0:
                e = self.echo_source(self.chunk)
                if e is not None and len(e) == self.chunk:
                    out += e * self.echo_gain
        return out, marks

    def _run(self) -> None:
        from websockets.sync.client import connect
        try:
            with connect(self.url, max_size=None, open_timeout=10) as ws:
                self.connected.set()
                self.log.add("harness", "audio_connected", url=self.url)
                t0 = time.monotonic()
                k = 0
                while not self._stop.is_set():
                    due = t0 + k * self.period
                    now = time.monotonic()
                    if due > now:
                        time.sleep(due - now)
                    sent_at = time.monotonic()
                    lateness = sent_at - due
                    if lateness > self.period:
                        self.late_chunks += 1
                    self.max_lateness = max(self.max_lateness, lateness)
                    chunk, marks = self._next_chunk()
                    pcm = (np.clip(chunk, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
                    ws.send(pcm)
                    # the chunk covers [due, due + period) of 'mic time'; the speech boundary
                    # inside it is at due + offset/SR (the schedule, not the send jitter)
                    for kind, aid, meta in marks:
                        off = meta.pop("offset_samples", 0)
                        self.log.add("harness", kind, t=due + off / SR, audio_id=aid,
                                     sent_at=sent_at, **meta)
                    self.sent_chunks += 1
                    k += 1
        except Exception as e:  # noqa: BLE001
            self.error = repr(e)
            self.log.add("harness", "audio_error", error=repr(e))
        finally:
            self.log.add("harness", "audio_stats", sent_chunks=self.sent_chunks,
                         late_chunks=self.late_chunks, max_lateness_s=round(self.max_lateness, 4))
