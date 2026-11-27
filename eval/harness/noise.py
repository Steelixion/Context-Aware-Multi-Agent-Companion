"""Deterministic noise generators (seeded, stateful so they can stream chunk by chunk) and
SNR mixing. Shared by the golden-dataset builder and the live noise bed in the runner.

Noise types:
  white  - Gaussian white noise
  pink   - 1/f noise (Paul Kellet's refined filter on white noise)
  cabin  - car-cabin approximation: brown (1/f^2) road rumble, low-passed engine harmonics
           (fundamental ~32 Hz with slow RPM drift, harmonics 2..6), and a little white hiss
These are synthetic stand-ins for real cabin recordings (documented limitation).
"""
from __future__ import annotations

import math

import numpy as np

SR = 16000
NOISE_TYPES = ("cabin", "pink", "white")


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if x.size else 0.0


def dbfs(x: np.ndarray) -> float:
    r = rms(x)
    return 20 * math.log10(r) if r > 0 else -math.inf


class NoiseGen:
    def __init__(self, kind: str, seed: int, sr: int = SR):
        if kind not in NOISE_TYPES:
            raise ValueError(f"unknown noise type {kind}")
        self.kind, self.sr = kind, sr
        self.rng = np.random.default_rng(seed)
        self.pink_state = np.zeros(7)
        self.brown = 0.0
        self.n = 0
        self.phase = np.zeros(6)

    def _pink(self, w: np.ndarray) -> np.ndarray:
        b = self.pink_state
        out = np.empty_like(w)
        for i, x in enumerate(w):         # Kellet filter (fine for offline + 50 ms chunks)
            b[0] = 0.99886 * b[0] + x * 0.0555179
            b[1] = 0.99332 * b[1] + x * 0.0750759
            b[2] = 0.96900 * b[2] + x * 0.1538520
            b[3] = 0.86650 * b[3] + x * 0.3104856
            b[4] = 0.55000 * b[4] + x * 0.5329522
            b[5] = -0.7616 * b[5] - x * 0.0168980
            out[i] = b[0] + b[1] + b[2] + b[3] + b[4] + b[5] + b[6] + x * 0.5362
            b[6] = x * 0.115926
        return out * 0.11

    def _cabin(self, n: int) -> np.ndarray:
        w = self.rng.standard_normal(n)
        brown = np.empty(n)
        acc = self.brown
        for i in range(n):                # leaky integrator -> 1/f^2 rumble
            acc = 0.995 * acc + w[i] * 0.05
            brown[i] = acc
        self.brown = acc
        t = (self.n + np.arange(n)) / self.sr
        f0 = 32.0 + 3.0 * np.sin(2 * np.pi * 0.05 * t)          # slow RPM drift
        eng = np.zeros(n)
        for h in range(1, 7):
            inc = 2 * np.pi * f0 * h / self.sr
            ph = self.phase[h - 1] + np.cumsum(inc)
            eng += np.sin(ph) / h
            self.phase[h - 1] = ph[-1] % (2 * np.pi)
        hiss = self.rng.standard_normal(n) * 0.02
        return brown * 0.8 + eng * 0.15 + hiss

    def generate(self, n: int) -> np.ndarray:
        """n samples, unit-less; scale with at_rms()."""
        if self.kind == "white":
            x = self.rng.standard_normal(n)
        elif self.kind == "pink":
            x = self._pink(self.rng.standard_normal(n))
        else:
            x = self._cabin(n)
        self.n += n
        return x.astype(np.float32)


def at_rms(x: np.ndarray, target_rms: float) -> np.ndarray:
    r = rms(x)
    return (x * (target_rms / r)).astype(np.float32) if r > 0 else x


def noise_rms_for_snr(speech: np.ndarray, snr_db: float) -> float:
    """Noise RMS so that 20*log10(rms(speech_active)/rms(noise)) == snr_db. Speech RMS is
    measured over active samples only (|x| above 1% of peak), so leading/trailing silence in
    the TTS output does not inflate the SNR."""
    peak = float(np.max(np.abs(speech))) if speech.size else 0.0
    active = speech[np.abs(speech) > 0.01 * peak] if peak > 0 else speech
    return rms(active) / (10 ** (snr_db / 20))


def mix_at_snr(speech: np.ndarray, kind: str, snr_db: float, seed: int, pad_s: float = 0.0) -> tuple[np.ndarray, float]:
    """Speech + noise at the given SNR, with pad_s of noise before and after.
    Returns (mixture float32, noise_rms) so a live noise bed can match the level."""
    pad = int(pad_s * SR)
    s = np.concatenate([np.zeros(pad, np.float32), speech.astype(np.float32), np.zeros(pad, np.float32)])
    nr = noise_rms_for_snr(speech, snr_db)
    noise = at_rms(NoiseGen(kind, seed).generate(s.size), nr)
    y = s + noise
    peak = float(np.max(np.abs(y)))
    if peak > 0.99:                       # avoid clipping; scale both together (SNR unchanged)
        y = y * (0.99 / peak)
        nr *= 0.99 / peak
    return y.astype(np.float32), nr
