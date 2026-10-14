"""Offline TTS. Piper neural voices (loaded once, in-process) if available per language,
else the Piper CLI, else pyttsx3 (OS voices). Returns PCM for the audio arbiter."""
from __future__ import annotations

import io
import os
import subprocess
import tempfile
import threading
import wave


class TTS:
    def __init__(self, piper_bin: str = "piper", piper_models: str = ""):
        self.piper_bin = piper_bin
        self.models = dict(p.split("=", 1) for p in piper_models.split(",") if "=" in p)
        self._voices: dict[str, object] = {}
        self._lock = threading.Lock()
        self.engine_name = "piper" if self.models else "pyttsx3"

    def _voice(self, lang: str):
        path = self.models.get(lang) or self.models.get("en")
        if not path:
            return None
        with self._lock:
            if path not in self._voices:
                from piper import PiperVoice
                self._voices[path] = PiperVoice.load(path)
            return self._voices[path]

    def synth(self, text: str, lang: str = "en"):
        """Blocking. Returns (float32 ndarray, samplerate) or None. Call via an executor."""
        import soundfile as sf
        try:
            if self.models:
                try:
                    voice = self._voice(lang)
                    buf = io.BytesIO()
                    with wave.open(buf, "wb") as w:
                        voice.synthesize_wav(text, w)
                    buf.seek(0)
                    return sf.read(buf, dtype="float32")
                except ImportError:
                    return self._piper_cli(text, lang)
            return self._pyttsx3(text)
        except Exception as e:  # noqa: BLE001
            print(f"[tts] synth failed: {e}")
            return None

    def _piper_cli(self, text: str, lang: str):
        import soundfile as sf
        path = tempfile.mktemp(suffix=".wav")
        try:
            model = self.models.get(lang) or self.models["en"]
            subprocess.run([self.piper_bin, "--model", model, "--output_file", path],
                           input=text.encode("utf-8"), check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
            return sf.read(path, dtype="float32")
        finally:
            if os.path.exists(path):
                os.remove(path)

    @staticmethod
    def _pyttsx3(text: str):
        import pyttsx3
        import soundfile as sf
        path = tempfile.mktemp(suffix=".wav")
        try:
            eng = pyttsx3.init()
            eng.save_to_file(text, path)
            eng.runAndWait()
            return sf.read(path, dtype="float32")
        finally:
            if os.path.exists(path):
                os.remove(path)
