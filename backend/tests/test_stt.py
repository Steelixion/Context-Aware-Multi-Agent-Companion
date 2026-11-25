"""Parakeet ASR engine (sherpa-onnx, CPU) and the microphone listener."""
import asyncio
import inspect
import queue
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from app.audio import stt
from app.audio.emotion import EmotionClassifier
from app.audio.stt import SR, SpeechListener, Transcriber, find_parakeet_files
from app.audio.vad import EnergySegmenter, SileroSegmenter
from app.bus import Bus
from app.config import Config

from .conftest import (MODEL_DIR, REFERENCES, VAD_MODEL, cer, load_wav, needs_model, needs_vad, resample,
                       silence, speech16k)


# --------------------------------------------------------------------------- engine
def test_find_parakeet_files_errors(tmp_path):
    for bad in ("", str(tmp_path / "missing"), str(tmp_path)):
        with pytest.raises(FileNotFoundError):
            find_parakeet_files(bad)


def test_find_parakeet_files(tmp_path):
    from app.audio.stt import PARAKEET_FILES
    for name in PARAKEET_FILES:
        (tmp_path / name).write_bytes(b"")
    found = find_parakeet_files(tmp_path)
    assert [p.name for p in found] == list(PARAKEET_FILES)


def test_cpu_only():
    assert "provider" not in inspect.signature(Transcriber).parameters
    assert not hasattr(Config(), "asr_provider")


@needs_model
def test_config_autodetects_the_parakeet_model():
    assert "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8" in MODEL_DIR


@needs_model
@pytest.mark.parametrize("lang", ["en"])
def test_transcribes_sample_wavs(transcriber, lang):
    audio, sr = load_wav(lang)
    text = transcriber.transcribe(audio, sr)
    assert cer(text, REFERENCES[lang]) < 0.15, text


@needs_model
@pytest.mark.parametrize("rate", [8000, 16000, 44100, 48000])
def test_any_sample_rate(transcriber, rate):
    audio, sr = load_wav("en")
    assert cer(transcriber.transcribe(resample(audio, sr, rate), rate), REFERENCES["en"]) < 0.15


@needs_model
def test_accepts_float64_and_2d_input(transcriber):
    text = transcriber.transcribe(speech16k("en").astype(np.float64).reshape(-1, 1), SR)
    assert cer(text, REFERENCES["en"]) < 0.15, text


@needs_model
@pytest.mark.parametrize("n", [0, 1, 300, 399, 1000])
def test_tiny_inputs_return_empty_without_crashing(transcriber, n):
    # sherpa-onnx < 2.0 aborted the whole process on < 400 samples; we never pass those in
    assert transcriber.transcribe(np.zeros(n, dtype=np.float32), SR) == ""


@needs_model
def test_silence_gives_no_text(transcriber):
    assert transcriber.transcribe(silence(2), SR) == ""


@needs_model
def test_noise_does_not_crash(transcriber):
    assert isinstance(transcriber.transcribe(np.random.default_rng(0).normal(0, 0.05, SR * 2), SR), str)


@needs_model
def test_concurrent_calls_are_safe(transcriber):
    audio, sr = load_wav("de")
    out, errs = [], []

    def work():
        try:
            out.append(transcriber.transcribe(audio, sr))
        except Exception as e:  # noqa: BLE001
            errs.append(e)
    ts = [threading.Thread(target=work) for _ in range(3)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errs and len(out) == 3 and len(set(out)) == 1


# --------------------------------------------------------------------------- listener
class _FeedQueue:
    """queue.Queue look-alike handing out (chunk, muted) pairs, then stopping the listener."""

    def __init__(self, items, listener):
        self.items, self.listener, self.i = list(items), listener, 0

    def get(self, timeout=None):
        if self.i >= len(self.items):
            self.listener.stop()
            raise queue.Empty
        self.i += 1
        return self.items[self.i - 1]


def _blocks(audio: np.ndarray, muted=False):
    b = int(SR * stt.BLOCK_S)
    audio = np.pad(audio, (0, -len(audio) % b))
    return [(audio[i:i + b], muted(i / SR) if callable(muted) else muted) for i in range(0, len(audio), b)]


def _listener(lang="en", segmenter=None):
    published, statuses = [], []

    def call_soon_threadsafe(fn, topic, msg):
        (published if topic == "speech" else statuses).append((topic, msg))
    lst = SpeechListener(Bus(), SimpleNamespace(call_soon_threadsafe=call_soon_threadsafe),
                         SimpleNamespace(speaking=False), SimpleNamespace(lang=lang), MODEL_DIR or "",
                         EmotionClassifier(), vad_model=VAD_MODEL)
    lst.segmenter = segmenter
    lst.statuses = statuses
    return lst, published


def _cuts(items, segmenter):
    lst, _ = _listener(segmenter=segmenter)
    cut = []
    lst._handle = lambda audio: cut.append(len(audio) / SR)
    lst._loop(_FeedQueue(items, lst))
    return cut


@needs_vad
def test_listener_ignores_audio_captured_while_muted():
    items = _blocks(np.concatenate([silence(1), speech16k("en"), silence(1.5)]), muted=True)
    assert _cuts(items, SileroSegmenter(VAD_MODEL)) == []


@needs_vad
def test_listener_discards_utterance_interrupted_by_companion_speech():
    en = speech16k("en")
    start = 1.0 + len(en) / SR / 2              # the companion starts talking half-way through
    items = _blocks(np.concatenate([silence(1), en, silence(1.5)]), muted=lambda t: t >= start)
    assert _cuts(items, SileroSegmenter(VAD_MODEL)) == []


@needs_vad
def test_listener_hears_speech_after_companion_stops():
    items = _blocks(np.concatenate([silence(2), speech16k("en"), silence(1.5)]), muted=lambda t: t < 1.0)
    assert len(_cuts(items, SileroSegmenter(VAD_MODEL))) == 1


def test_echo_tail_keeps_mic_muted_briefly(monkeypatch):
    lst, _ = _listener()
    clock = [100.0]
    monkeypatch.setattr(stt.time, "monotonic", lambda: clock[0])
    lst.arbiter.speaking = True
    assert lst._muted()
    lst.arbiter.speaking = False
    clock[0] += stt.ECHO_TAIL_S / 2
    assert lst._muted()                          # room echo of the last word
    clock[0] += stt.ECHO_TAIL_S
    assert not lst._muted()


@needs_model
@needs_vad
def test_listener_transcribes_and_publishes(transcriber):
    lst, published = _listener(lang="de", segmenter=SileroSegmenter(VAD_MODEL))  # detection must win
    lst.transcriber = transcriber
    lst._loop(_FeedQueue(_blocks(np.concatenate([silence(0.5), speech16k("en"), silence(1.5)])), lst))
    assert len(published) == 1
    topic, msg = published[0]
    assert topic == "speech" and msg["source"] == "voice" and msg["lang"] == "en"
    assert cer(msg["text"], REFERENCES["en"]) < 0.15, msg
    assert msg["emotion"] in ("neutral", "sad", "stressed", "happy", "angry")


def test_listener_keeps_session_language_when_unsure():
    lst, published = _listener(lang="ur")
    lst.transcriber = SimpleNamespace(transcribe=lambda a, sr: "ok")
    lst._handle(silence(1))
    assert published[0][1]["lang"] is None   # the companion then keeps session.lang ("ur")


def test_listener_uses_session_language_for_arabic_script_ties():
    lst, published = _listener(lang="ur")
    lst.transcriber = SimpleNamespace(transcribe=lambda a, sr: "کیا")
    lst._handle(silence(1))
    assert published[0][1]["lang"] == "ur"


def test_listener_publishes_nothing_for_empty_transcripts():
    lst, published = _listener()
    lst.transcriber = SimpleNamespace(transcribe=lambda a, sr: "")
    lst._handle(silence(1))
    assert published == []


def test_listener_survives_transcriber_errors():
    lst, published = _listener()

    def boom(a, sr):
        raise RuntimeError("decoder exploded")
    lst.transcriber = SimpleNamespace(transcribe=boom)
    lst._handle(silence(1))   # must not raise
    assert published == []


def test_status_publish_after_loop_closed_is_ignored():
    loop = asyncio.new_event_loop()
    loop.close()
    lst = SpeechListener(Bus(), loop, SimpleNamespace(speaking=False), SimpleNamespace(lang="en"),
                         "", EmotionClassifier())
    lst._set_status("no mic")   # must not raise during shutdown
    assert lst.status == "no mic"


def test_log_survives_unencodable_consoles(monkeypatch):
    out = []

    def cp1252_print(s, **kw):
        s.encode("cp1252")   # raises for Urdu, like print() to a redirected Windows stdout
        out.append(s)
    monkeypatch.setattr("builtins.print", cp1252_print)
    stt._log("[stt] lang=ur: آپ کیسے ہیں")
    assert len(out) == 1 and out[0].isascii() and out[0].startswith("[stt] lang=ur: ")


# --------------------------------------------------------------------------- full thread with a mic
@needs_model
@needs_vad
def test_run_with_fake_microphone(fake_mic):
    fake_mic.audio = np.concatenate([silence(0.5), speech16k("en"), silence(2)])
    lst, published = _listener(lang="en")
    assert lst.engine == "parakeet (starting)"
    lst.start()
    deadline = time.time() + 60
    while not published and time.time() < deadline:
        time.sleep(0.1)
    lst.stop()
    assert lst.status == "listening" and lst.engine == "parakeet + silero VAD"
    assert [m["status"] for _, m in lst.statuses] == ["loading", "listening"]
    assert lst.statuses[-1][1]["engine"] == "parakeet + silero VAD"
    assert len(published) == 1, published
    msg = published[0][1]
    assert msg["lang"] == "en" and cer(msg["text"], REFERENCES["en"]) < 0.15, msg


@needs_model
def test_run_falls_back_to_energy_vad_without_vad_model(fake_mic, tmp_path):
    lst, _ = _listener()
    lst.vad_model = str(tmp_path / "missing.onnx")
    threading.Thread(target=lst._run, daemon=True).start()
    assert fake_mic.opened.wait(60)
    time.sleep(0.2)
    lst.stop()
    assert isinstance(lst.segmenter, EnergySegmenter) and lst.engine == "parakeet + energy VAD"


def test_run_without_model_reports_status(tmp_path):
    lst, published = _listener()
    lst.model_dir = str(tmp_path)
    lst._run()
    assert lst.status == "no model" and lst.engine == "parakeet (no model)" and not published
    assert [m["status"] for _, m in lst.statuses] == ["loading", "no model"]


@needs_model
def test_run_without_microphone_reports_status(monkeypatch, fake_mic):
    def no_device(*a, **k):
        raise OSError("Error querying device -1")
    monkeypatch.setattr(fake_mic, "__init__", no_device)
    lst, _ = _listener()
    lst._run()
    assert lst.status == "no mic"
    assert lst.statuses[-1] == ("stt", dict(status="no mic", engine="parakeet (no mic)"))
