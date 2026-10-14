"""Speech-emotion classification (acoustic). HF audio-classification model if configured,
otherwise a crude prosody heuristic (loudness / zero-crossing) so the pipeline still has a signal."""
from __future__ import annotations

import numpy as np

_MAP = {"neu": "neutral", "hap": "happy", "ang": "angry", "sad": "sad",
        "neutral": "neutral", "happy": "happy", "angry": "angry", "sadness": "sad",
        "anger": "angry", "fear": "stressed", "disgust": "angry", "surprise": "happy"}


class EmotionClassifier:
    def __init__(self, model_id: str = ""):
        self.pipe = None
        if model_id:
            try:
                from transformers import pipeline
                self.pipe = pipeline("audio-classification", model=model_id)
            except Exception as e:  # noqa: BLE001
                print(f"[emotion] model unavailable ({e}); using heuristic")

    def classify(self, audio: np.ndarray, sr: int = 16000) -> str:
        if self.pipe is not None:
            try:
                top = self.pipe({"raw": audio, "sampling_rate": sr}, top_k=1)[0]["label"].lower()
                return _MAP.get(top, top)
            except Exception:  # noqa: BLE001
                pass
        rms = float(np.sqrt(np.mean(audio ** 2)) + 1e-9)
        zcr = float(np.mean(np.abs(np.diff(np.sign(audio)))) / 2)
        if rms > 0.12 and zcr > 0.12:
            return "stressed"
        if rms < 0.02:
            return "sad"   # low energy / flat
        return "neutral"
