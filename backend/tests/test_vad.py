"""Voice activity detection (Silero VAD v5 via sherpa-onnx, CPU) - QA suite.

Covers what makes a VAD feel glitchy in a car: triggering on engine/road noise, missing or
clipping speech, splitting a sentence at a short pause, dropping long speech, behaving
differently depending on how the audio is chunked, and crashing on bad model files.
"""
import time

import numpy as np
import pytest

from app.audio import vad
from app.audio.vad import (END_SILENCE_S, MAX_UTTERANCE_S, PAD_LEAD_S, PAD_TAIL_S, SR, EnergySegmenter,
                           SileroSegmenter, _Ring, make_segmenter)

from .conftest import (REFERENCES, VAD_MODEL, at_rms, cer, needs_model, needs_vad, noise, silence,
                       speech16k)


def run(seg, audio: np.ndarray, block: int = 800, flush: bool = True) -> list[np.ndarray]:
    out = []
    for i in range(0, len(audio), block):
        out += seg.accept(audio[i:i + block])
    return out + (seg.flush() if flush else [])


def scene(*parts) -> np.ndarray:
    return np.concatenate([p if isinstance(p, np.ndarray) else silence(p) for p in parts])


@pytest.fixture
def silero():
    if not VAD_MODEL:
        pytest.skip("VAD model not downloaded")
    return SileroSegmenter(VAD_MODEL)


# --------------------------------------------------------------------------- noise immunity
@needs_vad
@pytest.mark.parametrize("kind", ["white", "pink", "brown", "cabin"])
@pytest.mark.parametrize("rms", [0.01, 0.03, 0.1])
def test_no_false_triggers_on_noise(silero, kind, rms):
    assert run(silero, noise(kind, 30, rms)) == []


@needs_vad
def test_no_false_triggers_on_clicks_and_bumps(silero):
    rng = np.random.default_rng(3)
    audio = silence(10)
    for t in (1.0, 2.5, 4.0, 6.0, 8.5):                        # door clicks / indicator ticks
        i = int(t * SR)
        audio[i:i + 80] = rng.uniform(-0.8, 0.8, 80)
    bump = 0.5 * np.exp(-np.arange(int(0.12 * SR)) / 300) * np.sin(2 * np.pi * 60 * np.arange(int(0.12 * SR)) / SR)
    audio[int(5 * SR):int(5 * SR) + len(bump)] += bump.astype(np.float32)   # pothole thump
    assert run(silero, audio) == []


@needs_vad
def test_energy_vad_fails_in_cabin_noise_where_silero_works(silero):
    """Why Silero: steady engine noise above the energy threshold hides every pause."""
    clean = scene(1.0, at_rms(speech16k("en"), 0.1), 1.5, at_rms(speech16k("de"), 0.1), 2.0)
    audio = clean + noise("cabin", len(clean) / SR, 0.03)
    assert len(run(EnergySegmenter(), audio)) <= 1    # one endless "utterance", or none
    assert len(run(silero, audio)) == 2


# --------------------------------------------------------------------------- speech in noise
@needs_vad
@needs_model
@pytest.mark.parametrize("kind,snr_db", [("cabin", 10), ("cabin", 5), ("cabin", 0), ("pink", 10), ("white", 10)])
def test_finds_every_utterance_in_noise(silero, transcriber, kind, snr_db):
    langs = ["en"]
    parts, n_rms = [silence(2.0)], 0.03
    for lang in langs:
        parts += [at_rms(speech16k(lang), n_rms * 10 ** (snr_db / 20)), silence(1.5)]
    clean = np.concatenate(parts)
    audio = clean + noise(kind, len(clean) / SR, n_rms)
    segs = run(silero, audio)
    assert len(segs) == 1
    for lang, s in zip(langs, segs, strict=True):
        assert cer(transcriber.transcribe(s), REFERENCES[lang]) < (0.2 if snr_db >= 5 else 0.3), lang


# --------------------------------------------------------------------------- segmentation behaviour
@needs_vad
def test_output_does_not_depend_on_block_size(silero):
    audio = scene(1.0, speech16k("en"), 1.5, speech16k("en"), 1.5)
    counts = {b: len(run(SileroSegmenter(VAD_MODEL), audio, b)) for b in (160, 512, 777, 800, 1600, 4000)}
    assert set(counts.values()) == {2}, counts


@needs_vad
@pytest.mark.parametrize("pause,expected", [(0.3, 1), (0.6, 1), (1.2, 2), (2.0, 2)])
def test_short_pauses_stay_inside_one_utterance(silero, pause, expected):
    en = speech16k("en")
    half = len(en) // 2
    assert len(run(silero, scene(1.0, en[:half], pause, en[half:], 1.5))) == expected


@needs_vad
@needs_model
def test_short_word_is_caught(silero, transcriber):
    word = speech16k("en")[: int(0.6 * SR)]
    segs = run(silero, scene(1.0, word, 1.5))
    assert len(segs) == 1 and transcriber.transcribe(segs[0])


@needs_vad
@needs_model
def test_long_speech_is_split_without_losing_words(silero, transcriber):
    langs = ["en"] * 7
    speech = np.concatenate([speech16k(lang) for lang in langs])
    segs = run(silero, scene(1.0, speech, 1.5))
    assert len(segs) >= 2 and all(len(s) / SR <= MAX_UTTERANCE_S * 1.3 for s in segs)
    text = " ".join(transcriber.transcribe(s) for s in segs)
    ref = " ".join(REFERENCES[lang] for lang in langs)
    assert cer(text, ref) < 0.1


@needs_vad
def test_utterance_ends_promptly_after_speech(silero):
    en = speech16k("en")
    audio = scene(1.0, en, 3.0)
    for i in range(0, len(audio), 800):
        if silero.accept(audio[i:i + 800]):
            emitted = (i + 800) / SR
            break
    else:
        pytest.fail("no utterance")
    latency = emitted - (1.0 + len(en) / SR)   # the clip has a little trailing silence of its own
    assert 0 < latency <= END_SILENCE_S + 0.3, latency


@needs_vad
def test_segments_are_padded_with_context(silero):
    en = speech16k("en")
    seg = run(silero, scene(2.0, en, 2.0))[0]
    assert len(seg) / SR >= len(en) / SR - 0.5        # nothing substantial cut off
    assert len(seg) / SR <= len(en) / SR + PAD_LEAD_S + PAD_TAIL_S + 0.6


@needs_vad
def test_reset_drops_speech_in_progress(silero):
    """The companion starts talking mid-utterance: the half-heard speech must never surface."""
    assert run(silero, scene(1.0, speech16k("en")[: SR]), flush=False) == []
    silero.reset()
    assert run(silero, silence(2.0)) == []


@needs_vad
def test_cost_is_tiny(silero):
    audio = scene(1.0, speech16k("en"), 1.0, noise("cabin", 20, 0.03))
    t0 = time.perf_counter()
    run(silero, audio)
    assert (time.perf_counter() - t0) / (len(audio) / SR) < 0.05   # < 5 % of one core (measured ~0.7 %)


# --------------------------------------------------------------------------- robustness
def test_missing_model_falls_back_to_energy_vad(tmp_path):
    seg, note = make_segmenter(str(tmp_path / "missing.onnx"))
    assert isinstance(seg, EnergySegmenter) and "energy" in note
    seg, note = make_segmenter("")     # an empty path would make sherpa-onnx exit the process
    assert isinstance(seg, EnergySegmenter)


def test_corrupt_model_falls_back_to_energy_vad(tmp_path):
    bad = tmp_path / "bad.onnx"
    bad.write_bytes(np.random.default_rng(0).bytes(5000))
    seg, note = make_segmenter(str(bad))
    assert isinstance(seg, EnergySegmenter) and note


@needs_vad
@pytest.mark.parametrize("threshold", [0.0, 1.0, -1, 2])
def test_invalid_threshold_is_rejected(threshold):
    with pytest.raises(ValueError):
        SileroSegmenter(VAD_MODEL, threshold)


def test_energy_segmenter_still_works_in_a_quiet_room():
    tone = (0.1 * np.sin(2 * np.pi * 220 * np.arange(SR) / SR)).astype(np.float32)
    segs = run(EnergySegmenter(), scene(1.0, tone, 1.5))
    assert len(segs) == 1 and 1.0 <= len(segs[0]) / SR <= 1.0 + 0.3 + END_SILENCE_S + 0.1


def test_energy_segmenter_ignores_clicks():
    click = np.zeros(int(0.1 * SR), np.float32) + 0.2
    assert run(EnergySegmenter(), scene(1.0, click, 1.5)) == []


def test_ring_buffer_wraps_and_clips():
    r = _Ring(1.0)
    data = np.arange(3 * r.cap + 123, dtype=np.float32)
    for i in range(0, len(data), 777):
        r.push(data[i:i + 777])
    t = r.total
    assert np.array_equal(r.get(t - r.cap, t), data[t - r.cap:t])
    assert np.array_equal(r.get(t - 500, t - 100), data[t - 500:t - 100])
    assert np.array_equal(r.get(t - r.cap - 50, t + 50), data[t - r.cap:t])   # clipped to what is kept
    assert len(r.get(0, 10)) == 0
    big = _Ring(1.0)
    big.push(data)
    assert big.total == len(data) and np.array_equal(big.get(len(data) - big.cap, len(data)), data[-big.cap:])


def test_pad_constants_are_sane():
    assert 0 < PAD_TAIL_S < END_SILENCE_S and 0 < PAD_LEAD_S < 1 and vad.MIN_SPEECH_S < END_SILENCE_S
