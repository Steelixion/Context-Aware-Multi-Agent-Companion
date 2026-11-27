"""Speaker-output capture (WASAPI loopback of the default output device).

Gives the true audible onset/offset of everything the app plays (replies and warnings), and
feeds the echo path for the self-hearing test. Each 10 ms block is stamped with the monotonic
time at which the capture call returned, minus the block length (so the stamp is the block's
start). Device buffering adds a small, unknown constant delay (typically 10-30 ms); it is the
same for every measurement and is reported as a limitation.
"""
from __future__ import annotations

import math
import threading
import time
from collections import deque

import numpy as np

from .events import EventLog

CAPTURE_SR = 48000
OUT_SR = 16000
BLOCK = 480              # 10 ms at 48 kHz


class Loopback:
    def __init__(self, log: EventLog, wav_path=None):
        self.log, self.wav_path = log, wav_path
        self.db: list[float] = []          # dBFS per 10 ms block
        self.t0: float | None = None       # monotonic time of block 0
        self.blocks16: list[np.ndarray] = []
        self.echo_ring: deque = deque()
        self.echo_lock = threading.Lock()
        self.echo_delay_samples = 0
        self.available = False
        self.error: str | None = None
        self.device = ""
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, name="loopback", daemon=True)

    def start(self, timeout: float = 10.0) -> bool:
        self._thread.start()
        self._ready.wait(timeout)
        return self.available

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(5)
        if self.wav_path and self.blocks16:
            import soundfile as sf
            sf.write(str(self.wav_path), np.concatenate(self.blocks16), OUT_SR, subtype="PCM_16")

    def set_echo_delay(self, ms: float) -> None:
        self.echo_delay_samples = int(OUT_SR * ms / 1000)
        with self.echo_lock:
            self.echo_ring.clear()
            self.echo_ring.extend(np.zeros(self.echo_delay_samples, np.float32))

    def echo_read(self, n: int) -> np.ndarray | None:
        """Next n samples (16 kHz) of speaker output for the mic echo path; zeros if behind."""
        with self.echo_lock:
            if len(self.echo_ring) < n:
                return np.zeros(n, np.float32)
            return np.array([self.echo_ring.popleft() for _ in range(n)], np.float32)

    def _run(self) -> None:
        try:
            import soundcard as sc
            spk = sc.default_speaker()
            self.device = spk.name
            mic = sc.get_microphone(id=str(spk.name), include_loopback=True)
            with mic.recorder(samplerate=CAPTURE_SR, channels=1, blocksize=BLOCK) as rec:
                self.available = True
                self.log.add("harness", "loopback_started", device=self.device)
                self._ready.set()
                k = 0
                while not self._stop.is_set():
                    data = rec.record(numframes=BLOCK)
                    t_end = time.monotonic()
                    x = np.asarray(data, np.float32).reshape(-1)
                    if self.t0 is None:
                        self.t0 = t_end - len(x) / CAPTURE_SR
                    r = float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0
                    self.db.append(20 * math.log10(r) if r > 1e-9 else -120.0)
                    y = x[: len(x) // 3 * 3].reshape(-1, 3).mean(axis=1)   # crude 48k -> 16k
                    self.blocks16.append(y)
                    with self.echo_lock:
                        self.echo_ring.extend(y.tolist())
                        excess = len(self.echo_ring) - OUT_SR * 2
                        for _ in range(max(0, excess)):
                            self.echo_ring.popleft()
                    k += 1
        except Exception as e:  # noqa: BLE001
            self.error = repr(e)
            self.log.add("harness", "loopback_error", error=repr(e))
        finally:
            self._ready.set()

    def summary(self) -> dict:
        return dict(available=self.available, device=self.device, t0=self.t0,
                    hop_s=BLOCK / CAPTURE_SR, db=[round(v, 1) for v in self.db], error=self.error)
