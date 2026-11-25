import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest

from app.config import MODELS, Config

CFG = Config()
MODEL_DIR = CFG.asr_model_dir
VAD_MODEL = CFG.vad_model
ASSETS = MODELS / "test-assets"
needs_model = pytest.mark.skipif(not MODEL_DIR, reason="ASR model not downloaded (scripts/download_models.py)")
needs_vad = pytest.mark.skipif(not VAD_MODEL, reason="VAD model not downloaded (scripts/download_models.py)")
needs_vision = pytest.mark.skipif(
    not CFG.face_model or not CFG.pose_model or not (ASSETS / "business-person.png").is_file(),
    reason="vision models / test images not downloaded (scripts/download_models.py --test-assets)")

SR = 16000
# reference transcripts of the model's sample WAVs (lowercase, no punctuation, like the model output)
REFERENCES = {
    "en": "ask not what your country can do for you ask what you can do for your country",
    "de": "alles hat ein ende nur die wurst hat zwei",
    "fr": "ne vous demandez pas ce que votre pays peut faire pour vous "
          "demandez-vous plutôt ce que vous pouvez faire pour lui",
    "es": "no preguntes qué puede hacer tu país por ti pregunta qué puedes hacer tú por tu país",
}


def cer(hyp: str, ref: str) -> float:
    """Character error rate (Levenshtein distance / reference length)."""
    prev = list(range(len(ref) + 1))
    for i, h in enumerate(hyp, 1):
        cur = [i]
        for j, r in enumerate(ref, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (h != r)))
        prev = cur
    return prev[-1] / max(1, len(ref))


def load_wav(lang: str) -> tuple[np.ndarray, int]:
    import soundfile as sf
    audio, sr = sf.read(str(Path(MODEL_DIR) / "test_wavs" / f"{lang}.wav"), dtype="float32", always_2d=True)
    return audio[:, 0], sr


def speech16k(lang: str) -> np.ndarray:
    a, sr = load_wav(lang)
    return resample(a, sr, SR)


def resample(audio: np.ndarray, sr: int, target: int) -> np.ndarray:
    n = int(round(len(audio) * target / sr))
    return np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(SR * seconds), np.float32)


def noise(kind: str, seconds: float, rms: float, seed: int = 7) -> np.ndarray:
    """white / pink / brown noise, or 'cabin' (rumble + engine hum harmonics, slowly modulated)."""
    rng = np.random.default_rng(seed)
    n = int(SR * seconds)
    w = rng.standard_normal(n)
    if kind == "white":
        x = w
    else:
        f = np.fft.rfftfreq(n, 1 / SR)
        f[0] = f[1]
        x = np.fft.irfft(np.fft.rfft(w) / f ** {"pink": 0.5, "brown": 1.0, "cabin": 0.75}[kind], n)
        if kind == "cabin":
            t = np.arange(n) / SR
            x = x / x.std() + 0.6 * sum(np.sin(2 * np.pi * k * 33 * t + k) / k for k in (1, 2, 3, 4))
            x *= 1 + 0.2 * np.sin(2 * np.pi * 0.3 * t)
    return (x / x.std() * rms).astype(np.float32)


def at_rms(x: np.ndarray, rms: float) -> np.ndarray:
    return (x / (np.sqrt(np.mean(x ** 2)) + 1e-12) * rms).astype(np.float32)


@pytest.fixture(scope="session")
def transcriber():
    if not MODEL_DIR:
        pytest.skip("ASR model not downloaded")
    from app.audio.stt import Transcriber
    return Transcriber(MODEL_DIR, num_threads=4)


def vision_cfg(**overrides) -> dict:
    """Vision settings as the worker process receives them."""
    return Config().vision_dict() | overrides


class FakeInputStream:
    """Stands in for sounddevice.InputStream: plays `audio` (16 kHz) into the callback in
    50 ms blocks, `speed` times faster than real time, then goes quiet."""
    audio = np.zeros(0, dtype=np.float32)
    speed = 4.0
    opened = threading.Event()

    def __init__(self, samplerate, channels, blocksize, callback, dtype="float32"):
        assert samplerate == SR and channels == 1
        self.block, self.cb, self._stop = blocksize, callback, threading.Event()

    def __enter__(self):
        FakeInputStream.opened.set()
        threading.Thread(target=self._feed, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self._stop.set()

    def _feed(self):
        a = FakeInputStream.audio
        for i in range(0, len(a), self.block):
            if self._stop.is_set():
                return
            blk = np.zeros((self.block, 1), dtype=np.float32)
            seg = a[i:i + self.block]
            blk[:len(seg), 0] = seg
            self.cb(blk, self.block, None, None)
            time.sleep(self.block / SR / FakeInputStream.speed)


@pytest.fixture
def fake_mic(monkeypatch):
    """Replace the sounddevice module with a fake microphone. Set .audio before starting."""
    FakeInputStream.opened = threading.Event()
    FakeInputStream.audio, FakeInputStream.speed = np.zeros(0, dtype=np.float32), 4.0
    mod = types.ModuleType("sounddevice")
    mod.InputStream = FakeInputStream
    mod.play = lambda *a, **k: None
    mod.wait = lambda: None
    mod.stop = lambda: None
    monkeypatch.setitem(sys.modules, "sounddevice", mod)
    return FakeInputStream
