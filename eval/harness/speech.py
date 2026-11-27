"""Reference speech synthesis for the golden dataset (independent of the app's TTS).

Engine: Kokoro-82M through kokoro-onnx (offline, CPU). Model files are fetched once by
build_golden.py --fetch-tts into eval/.cache/kokoro and verified against pinned SHA-256.
Output: 16 kHz mono float32, peak-safe, RMS-normalised to SPEECH_DBFS.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

SR = 16000
SPEECH_DBFS = -26.0          # typical close-talk speech level into a car mic

KOKORO_FILES = {             # pinned: name -> (url, sha256)
    "kokoro-v1.0.onnx": ("https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
                         "7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5"),
    "voices-v1.0.bin": ("https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
                        "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d"),
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def resample_poly(x: np.ndarray, up: int, down: int, taps_per_phase: int = 32) -> np.ndarray:
    """Rational resampling with a Kaiser-windowed sinc low-pass (deterministic, numpy only)."""
    if up == down:
        return x.astype(np.float32)
    g = math.gcd(up, down)
    up, down = up // g, down // g
    cutoff = 1.0 / max(up, down)
    n = taps_per_phase * max(up, down)
    t = np.arange(-n, n + 1)
    h = cutoff * np.sinc(cutoff * t) * np.kaiser(len(t), 8.0)
    h *= up / h.sum()
    y = np.zeros(len(x) * up, np.float64)
    y[::up] = x
    y = np.convolve(y, h, mode="same")
    return y[::down].astype(np.float32)


def normalise(x: np.ndarray, target_dbfs: float = SPEECH_DBFS) -> np.ndarray:
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak == 0:
        return x.astype(np.float32)
    active = x[np.abs(x) > 0.01 * peak]
    r = float(np.sqrt(np.mean(np.square(active, dtype=np.float64))))
    y = x * (10 ** (target_dbfs / 20) / r)
    p = float(np.max(np.abs(y)))
    if p > 0.95:
        y *= 0.95 / p
    return y.astype(np.float32)


class Synth:
    def __init__(self, cache: Path):
        from kokoro_onnx import Kokoro
        for name, (_, sha) in KOKORO_FILES.items():
            p = cache / name
            if not p.is_file():
                raise FileNotFoundError(f"{p} missing: run python eval/golden/build_golden.py --fetch-tts")
            got = sha256_file(p)
            if got != sha:
                raise RuntimeError(f"{p} checksum mismatch: {got} != {sha}")
        self.k = Kokoro(str(cache / "kokoro-v1.0.onnx"), str(cache / "voices-v1.0.bin"))
        self.engine = "kokoro-onnx / kokoro-v1.0"

    def say(self, text: str, voice: str, speed: float, lang: str = "en-us") -> np.ndarray:
        lang = "en-gb" if voice.startswith("b") else lang
        samples, sr = self.k.create(text, voice=voice, speed=speed, lang=lang)
        x = np.asarray(samples, np.float32)
        if sr != SR:
            x = resample_poly(x, SR, sr)
        return normalise(x)
