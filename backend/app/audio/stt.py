"""Speech-to-text: mic -> VAD (Silero, sherpa-onnx) -> Parakeet TDT int8 (English)
-> speech emotion -> 'speech' bus topic. Everything runs on the CPU.

SpeechListener owns a background thread. Transcriber is the ASR engine on its own (no mic
needed), so it can be used and tested on files.
"""
from __future__ import annotations

import asyncio
import queue
import threading
import time
from pathlib import Path

import numpy as np

from ..bus import Bus
from .emotion import EmotionClassifier
from .vad import SR, make_segmenter

BLOCK_S = 0.05      # mic callback block (50 ms)
ECHO_TAIL_S = 0.3   # keep the mic muted this long after the companion stops talking (room echo)
ASR_LANG = "en"     # Parakeet TDT v2 is English-only; no language code is passed to the recognizer
PARAKEET_FILES = ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt")


def find_parakeet_files(model_dir: str | Path) -> tuple[Path, Path, Path, Path]:
    """(encoder, decoder, joiner, tokens) inside a sherpa-onnx Parakeet model directory."""
    d = Path(model_dir) if model_dir else None
    if d is None or not d.is_dir():
        raise FileNotFoundError(f"Parakeet model folder not found: '{model_dir}' "
                                "(run: python scripts/download_models.py --only stt)")
    paths = tuple(d / name for name in PARAKEET_FILES)
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Parakeet model is incomplete in '{d}': missing {', '.join(missing)}")
    return paths


class Transcriber:
    """sherpa-onnx OfflineRecognizer for Parakeet TDT (NeMo transducer, CPU).
    Blocking; keep it off the event loop (a thread or run_in_executor)."""

    def __init__(self, model_dir: str | Path, num_threads: int = 2):
        import sherpa_onnx
        encoder, decoder, joiner, tokens = find_parakeet_files(model_dir)
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(encoder), decoder=str(decoder), joiner=str(joiner), tokens=str(tokens),
            num_threads=max(1, num_threads), provider="cpu", model_type="nemo_transducer",
            decoding_method="greedy_search")
        self.name = Path(model_dir).name
        self._lock = threading.Lock()

    def transcribe(self, audio: np.ndarray, sr: int = SR) -> str:
        """Mono float32 samples in [-1, 1] at any rate (sherpa-onnx resamples) -> text."""
        audio = np.ascontiguousarray(np.asarray(audio, dtype=np.float32).reshape(-1))
        if audio.size < sr * 0.1:   # under 100 ms: too short for the encoder
            return ""
        with self._lock:
            stream = self.recognizer.create_stream()
            stream.accept_waveform(sr, audio)
            self.recognizer.decode_stream(stream)
            return stream.result.text.strip()


class SpeechListener:
    def __init__(self, bus: Bus, loop: asyncio.AbstractEventLoop, arbiter, session,
                 model_dir: str, emotion: EmotionClassifier, num_threads: int = 2,
                 vad_model: str = "", vad_threshold: float = 0.5):
        self.bus, self.loop, self.arbiter, self.session = bus, loop, arbiter, session
        self.model_dir, self.num_threads = model_dir, num_threads
        self.vad_model, self.vad_threshold = vad_model, vad_threshold
        self.emotion = emotion
        self.transcriber: Transcriber | None = None
        self.segmenter = None
        self.status = "starting"
        self._muted_until = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def engine(self) -> str:
        """Short label for the UI: 'parakeet + silero VAD' when listening, else the status."""
        if self.status == "listening" and self.segmenter is not None:
            return f"parakeet + {self.segmenter.name} VAD"
        return f"parakeet ({self.status})"

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="stt", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _set_status(self, status: str) -> None:
        """Record and broadcast ('stt' topic -> dashboard chip) the listener's state."""
        self.status = status
        self._publish("stt", dict(status=status, engine=self.engine))

    def _publish(self, topic: str, msg: dict) -> None:
        try:
            self.loop.call_soon_threadsafe(self.bus.publish, topic, msg)
        except RuntimeError:   # event loop already closed (app shutting down)
            pass

    def _muted(self) -> bool:
        """True while the companion speaks and for ECHO_TAIL_S after (decided at capture time)."""
        now = time.monotonic()
        if self.arbiter.speaking:
            self._muted_until = now + ECHO_TAIL_S
            return True
        return now < self._muted_until

    def _run(self) -> None:
        try:
            self._set_status("loading")
            t0 = time.time()
            self.transcriber = Transcriber(self.model_dir, self.num_threads)
            self.segmenter, note = make_segmenter(self.vad_model, self.vad_threshold)
            _log(f"[stt] loaded {self.transcriber.name} + {self.segmenter.name} VAD "
                 f"in {time.time() - t0:.1f}s" + (f"; {note}" if note else ""))
        except Exception as e:  # noqa: BLE001
            self._set_status("no model")
            _log(f"[stt] disabled: {e}")
            return
        try:
            import sounddevice as sd
            q: queue.Queue = queue.Queue()

            def cb(indata, frames, t, status):
                q.put((indata[:, 0].copy(), self._muted()))

            with sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                                blocksize=int(SR * BLOCK_S), callback=cb):
                self._set_status("listening")
                _log("[stt] listening")
                self._loop(q)
        except Exception as e:  # noqa: BLE001
            self._set_status("no mic")
            _log(f"[stt] microphone unavailable: {e}")

    def _loop(self, q: queue.Queue) -> None:
        """Feed unmuted audio to the VAD; transcribe each utterance it completes."""
        was_muted = False
        while not self._stop.is_set():
            try:
                chunk, muted = q.get(timeout=0.2)
            except queue.Empty:
                continue
            if muted:
                if not was_muted:        # drop any half-heard utterance once, when muting starts
                    self.segmenter.reset()
                was_muted = True
                continue
            was_muted = False
            for utterance in self.segmenter.accept(chunk):
                self._handle(utterance)

    def _handle(self, audio: np.ndarray) -> None:
        try:
            t0 = time.time()
            text = self.transcriber.transcribe(audio, SR)
            dt = time.time() - t0
        except Exception as e:  # noqa: BLE001
            _log(f"[stt] transcription failed: {e}")
            return
        if not text:
            return
        emo = self.emotion.classify(audio, SR)
        self._publish("speech", dict(text=text, lang=ASR_LANG, emotion=emo, source="voice",
                                     ts=time.time()))
        _log(f"[stt] {len(audio) / SR:.1f}s audio -> {dt * 1000:.0f} ms, lang={ASR_LANG}: {text}")


def _log(line: str) -> None:
    """print() that survives consoles/pipes whose encoding can't show the transcript's script."""
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "backslashreplace").decode("ascii"), flush=True)


def _cli() -> None:
    """python -m app.audio.stt file1.wav ...   |   python -m app.audio.stt --mic"""
    import argparse
    from types import SimpleNamespace

    from ..config import Config
    cfg = Config()
    p = argparse.ArgumentParser(description="Transcribe audio files, or the live mic, on the CPU.")
    p.add_argument("files", nargs="*", help="audio files (any rate; first channel is used)")
    p.add_argument("--mic", action="store_true", help="listen on the default microphone (Ctrl+C to stop)")
    p.add_argument("--model-dir", default=cfg.asr_model_dir)
    p.add_argument("--vad-model", default=cfg.vad_model)
    p.add_argument("--threads", type=int, default=cfg.asr_threads)
    a = p.parse_args()
    if not a.files and not a.mic:
        p.error("give audio files or --mic")
    if a.mic:
        printer = SimpleNamespace(call_soon_threadsafe=lambda fn, topic, msg: None)  # _handle logs it
        listener = SpeechListener(Bus(), printer, SimpleNamespace(speaking=False),
                                  SimpleNamespace(), a.model_dir, EmotionClassifier(),
                                  a.threads, a.vad_model, cfg.vad_threshold)
        try:
            listener._run()
        except KeyboardInterrupt:
            pass
        return
    import soundfile as sf
    tr = Transcriber(a.model_dir, a.threads)
    for f in a.files:
        audio, sr = sf.read(f, dtype="float32", always_2d=True)
        t0 = time.time()
        text = tr.transcribe(audio[:, 0], sr)
        dt, dur = time.time() - t0, max(len(audio) / sr, 1e-9)
        _log(f"{f}  ({dur:.1f}s, {dt * 1000:.0f} ms, RTF {dt / dur:.3f}, "
             f"lang={ASR_LANG})\n  {text}")


if __name__ == "__main__":
    _cli()
