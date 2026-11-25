"""Voice activity detection: turns the microphone stream into utterances.

SileroSegmenter (sherpa-onnx + Silero VAD v5, CPU) is used whenever the model file is present.
It is a neural speech detector, so engine, road and fan noise do not trigger it, which a
loudness threshold cannot manage in a car. EnergySegmenter is the fallback when the model is
missing; it only works in a quiet room.

Both accept 16 kHz mono float32 blocks of any size and return the finished utterances.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

SR = 16000
END_SILENCE_S = 0.8     # a pause this long ends an utterance (shorter pauses stay inside it)
MIN_SPEECH_S = 0.25     # shorter sounds (clicks, bumps, coughs) are not speech
MAX_UTTERANCE_S = 20.0  # longer speech is split here (the ASR model handles up to ~40 s)
PAD_LEAD_S = 0.4        # context added around each Silero segment: measured to lower the
PAD_TAIL_S = 0.3        # ASR error rate (CER 0.060 -> 0.049 over clean and noisy speech)


class _Ring:
    """Fixed-size history of the stream, addressed by absolute sample index."""

    def __init__(self, seconds: float):
        self.cap = int(seconds * SR)
        self.buf = np.zeros(self.cap, np.float32)
        self.total = 0

    def push(self, x: np.ndarray) -> None:
        n = len(x)
        if n >= self.cap:
            self.total += n - self.cap
            x, n = x[-self.cap:], self.cap
        i = self.total % self.cap
        first = min(n, self.cap - i)
        self.buf[i:i + first] = x[:first]
        self.buf[:n - first] = x[first:]
        self.total += n

    def get(self, start: int, end: int) -> np.ndarray:
        start, end = max(start, self.total - self.cap, 0), min(end, self.total)
        if end <= start:
            return np.zeros(0, np.float32)
        idx = np.arange(start, end) % self.cap
        return self.buf[idx]


class SileroSegmenter:
    name = "silero"

    def __init__(self, model_path: str | Path, threshold: float = 0.5):
        import sherpa_onnx
        p = Path(model_path) if model_path else None
        if p is None or not p.is_file():   # sherpa-onnx exits the process on an empty model path
            raise FileNotFoundError(f"Silero VAD model not found: '{model_path}'")
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"VAD threshold must be in (0, 1), got {threshold}")
        c = sherpa_onnx.VadModelConfig()
        c.silero_vad.model = str(p)
        c.silero_vad.threshold = threshold
        c.silero_vad.min_silence_duration = END_SILENCE_S
        c.silero_vad.min_speech_duration = MIN_SPEECH_S
        c.silero_vad.max_speech_duration = MAX_UTTERANCE_S
        c.silero_vad.window_size = 512
        c.sample_rate = SR
        c.num_threads = 1
        c.provider = "cpu"
        if not c.validate():
            raise ValueError(f"invalid Silero VAD config for '{p}'")
        self._vad = sherpa_onnx.VoiceActivityDetector(c, buffer_size_in_seconds=MAX_UTTERANCE_S + 10)
        # segment starts are absolute sample indices since construction/reset: keep the same
        # history here so each segment can be cut again with PAD_LEAD_S / PAD_TAIL_S around it
        self._ring = _Ring(MAX_UTTERANCE_S * 1.5 + END_SILENCE_S + PAD_LEAD_S + 5)

    def accept(self, samples: np.ndarray) -> list[np.ndarray]:
        x = np.ascontiguousarray(samples, dtype=np.float32).reshape(-1)
        self._ring.push(x)
        self._vad.accept_waveform(x)
        return self._drain()

    def flush(self) -> list[np.ndarray]:
        """End of stream: return speech still in progress."""
        self._vad.flush()
        return self._drain()

    def reset(self) -> None:
        """Drop everything, including speech in progress (used while the companion talks)."""
        self._vad.reset()
        self._ring = _Ring(self._ring.cap / SR)

    def _drain(self) -> list[np.ndarray]:
        out = []
        while not self._vad.empty():
            seg = self._vad.front
            start, n = int(seg.start), len(seg.samples)
            padded = self._ring.get(start - int(PAD_LEAD_S * SR), start + n + int(PAD_TAIL_S * SR))
            out.append(padded if len(padded) >= n else np.asarray(seg.samples, dtype=np.float32))
            self._vad.pop()
        return out


class EnergySegmenter:
    """Loudness threshold with pre-roll. Fallback only: any steady noise above VAD_RMS
    (an engine, a fan) makes it think the driver never stops talking."""
    name = "energy"
    BLOCK = int(SR * 0.05)
    VAD_RMS = 0.015
    PRE_ROLL_S = 0.3

    def __init__(self):
        self._buf = np.zeros(0, np.float32)
        self.reset()

    def reset(self) -> None:
        self._pre: list[np.ndarray] = []
        self._voiced: list[np.ndarray] = []
        self._n = self._onset = self._last_loud = 0
        self._silence, self._active = 0.0, False
        self._buf = np.zeros(0, np.float32)

    def accept(self, samples: np.ndarray) -> list[np.ndarray]:
        self._buf = np.concatenate([self._buf, np.asarray(samples, np.float32).reshape(-1)])
        out = []
        while len(self._buf) >= self.BLOCK:
            block, self._buf = self._buf[:self.BLOCK], self._buf[self.BLOCK:]
            utt = self._block(block)
            if utt is not None:
                out.append(utt)
        return out

    def flush(self) -> list[np.ndarray]:
        utt = self._finish() if self._active else None
        self.reset()
        return [utt] if utt is not None else []

    def _block(self, block: np.ndarray) -> np.ndarray | None:
        loud = float(np.sqrt(np.mean(block ** 2))) > self.VAD_RMS
        if not self._active:
            if not loud:
                self._pre = (self._pre + [block])[-int(self.PRE_ROLL_S * SR / self.BLOCK):]
                return None
            self._voiced, self._silence, self._active = self._pre, 0.0, True
            self._n = self._onset = sum(map(len, self._pre))
            self._pre = []
        self._voiced.append(block)
        self._n += len(block)
        if loud:
            self._silence, self._last_loud = 0.0, self._n
        else:
            self._silence += len(block) / SR
        if self._silence > END_SILENCE_S or self._n >= SR * MAX_UTTERANCE_S:
            return self._finish()
        return None

    def _finish(self) -> np.ndarray | None:
        audio = np.concatenate(self._voiced)
        speech_s = (self._last_loud - self._onset) / SR
        self._voiced, self._n, self._silence, self._active = [], 0, 0.0, False
        return audio if speech_s >= MIN_SPEECH_S else None


def make_segmenter(model_path: str, threshold: float = 0.5) -> tuple[SileroSegmenter | EnergySegmenter, str]:
    """(segmenter, note). Falls back to the energy segmenter, with the reason, if Silero can't load."""
    try:
        return SileroSegmenter(model_path, threshold), ""
    except Exception as e:  # noqa: BLE001
        return EnergySegmenter(), f"Silero VAD unavailable ({e}); using the energy VAD"
