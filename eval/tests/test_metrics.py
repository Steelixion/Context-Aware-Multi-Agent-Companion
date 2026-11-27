"""Unit tests for the scoring functions, on small hand-checked synthetic inputs."""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness import metrics as m  # noqa: E402


# ---------------------------------------------------------------- normalisation and WER
def test_normalize_text_rules():
    assert m.normalize_text("Hello, World!") == ["hello", "world"]
    assert m.normalize_text("It’s a well-known fact.") == ["its", "a", "well", "known", "fact"]
    assert m.normalize_text("I have 3 cats") == ["i", "have", "three", "cats"]
    assert m.normalize_text("") == []
    assert m.normalize_text("Check the tyre colour") == ["check", "the", "tire", "color"]


def test_wer_spelling_variants_are_not_errors():
    ref = "What is the tire pressure on my front left wheel right now?"
    assert m.word_error_rate(ref, "What is the tyre pressure on my front left wheel right now?").wer == 0


def test_wer_identical_and_punctuation_insensitive():
    r = m.word_error_rate("Tell me a fact about the moon.", "tell me a fact about the moon")
    assert r.wer == 0 and r.ref_words == 7


def test_wer_counts_sub_del_ins():
    r = m.word_error_rate("a b c d", "a x c d e")          # b->x = S, e = I (unambiguous)
    assert (r.substitutions, r.deletions, r.insertions) == (1, 0, 1)
    assert r.wer == pytest.approx(0.5)
    r = m.word_error_rate("a b c", "a c")                  # b missing = D
    assert (r.substitutions, r.deletions, r.insertions) == (0, 1, 0)
    # equally cheap alignments exist here (3 S, or S + D + I); WER is the same either way
    assert m.word_error_rate("the cat sat on the mat", "the bat sat on mat today").wer == pytest.approx(0.5)


def test_wer_empty_cases():
    assert m.word_error_rate("", "").wer == 0
    assert math.isinf(m.word_error_rate("", "hello").wer)
    r = m.word_error_rate("hello there", "")
    assert r.wer == 1.0 and r.deletions == 2


def test_wer_can_exceed_one():
    assert m.word_error_rate("hi", "oh hello there friend").wer == pytest.approx(4.0)


def test_corpus_wer_weights_by_reference_length():
    pairs = [("a b c d", "a b c d"), ("x y", "x z")]   # 0 errors / 4, 1 error / 2
    assert m.corpus_wer(pairs) == pytest.approx(1 / 6)
    assert m.corpus_wer([]) is None


def test_ngram_overlap_self_hearing():
    reply = "You seem a little tired. How about a short break at the next stop?"
    assert m.ngram_overlap("how about a short break", reply) == 1.0
    assert m.ngram_overlap("what is the weather in paris", reply) == 0.0
    assert m.ngram_overlap("", reply) == 0.0


# ---------------------------------------------------------------- percentiles
def test_percentile_matches_numpy_linear():
    xs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert m.percentile(xs, 50) == pytest.approx(5.5)
    assert m.percentile(xs, 95) == pytest.approx(9.55)
    assert m.percentile(xs, 99) == pytest.approx(9.91)
    assert m.percentile(xs, 0) == 1 and m.percentile(xs, 100) == 10


def test_percentile_edge_cases():
    assert m.percentile([], 50) is None
    assert m.percentile([4.2], 99) == 4.2
    assert m.percentile([None, float("nan"), 2.0], 50) == 2.0


def test_latency_summary():
    s = m.latency_summary([0.5, 1.0, 1.5, None])
    assert s["n"] == 3 and s["max"] == 1.5 and s["p50"] == 1.0
    assert m.latency_summary([])["p95"] is None


# ---------------------------------------------------------------- intervals and events
def test_intervals_from_labels():
    labels = [0, 0, 1, 1, 1, 0, 1, 1]
    assert m.intervals_from_labels(labels, fps=2) == [(1.0, 2.5), (3.0, 4.0)]
    assert m.intervals_from_labels([0, 0], fps=30) == []
    assert m.intervals_from_labels([1], fps=10) == [(0.0, 0.1)]


def test_merge_and_complement():
    assert m.merge_intervals([(5, 6), (0, 1), (1.2, 2)], gap=0.5) == [(0, 2), (5, 6)]
    assert m.intervals_complement([(2, 3), (5, 7)], 0, 10) == [(0, 2), (3, 5), (7, 10)]
    assert m.total_length([(0, 2), (5, 6)]) == 3


def test_step_value_and_compress():
    ch = m.compress_changes([(0, "ALERT"), (1, "ALERT"), (2, "HIGH"), (3, "HIGH"), (4, "ALERT")])
    assert ch == [(0, "ALERT"), (2, "HIGH"), (4, "ALERT")]
    assert m.step_value(ch, -1) is None
    assert m.step_value(ch, 2) == "HIGH" and m.step_value(ch, 3.99) == "HIGH"
    assert m.step_value(ch, 10) == "ALERT"


def test_detection_delay_and_missed_events():
    gt = [(10.0, 20.0), (40.0, 45.0), (60.0, 70.0)]
    changes = [(0, "ALERT"), (12.5, "MILD"), (14.0, "HIGH"), (22, "ALERT"),
               (39.0, "HIGH"), (41, "ALERT")]
    high = m.match_events(gt, changes, min_rank=2, early_s=2.0, late_s=3.0)
    assert [e.detected for e in high] == [True, True, False]
    assert high[0].delay == pytest.approx(4.0)
    assert high[1].delay == pytest.approx(-1.0)          # 1 s early, inside the early window
    mild = m.match_events(gt, changes, min_rank=1, early_s=0.0, late_s=0.0)
    assert mild[0].delay == pytest.approx(2.5)
    # the level is already HIGH at the 40 s onset: detected at the onset itself, delay 0
    assert mild[1].detected is True and mild[1].delay == 0.0
    assert mild[2].detected is False


def test_first_crossing_level_already_high_at_window_start():
    changes = [(0, "HIGH")]
    assert m.first_crossing(changes, 2, 5.0, 9.0) == 5.0


def test_false_alarms_outside_events():
    gt = [(10.0, 20.0)]
    warns = [5.0, 9.5, 15.0, 21.0, 30.0]
    assert m.warnings_outside(warns, gt, before_s=1.0, after_s=2.0) == [5.0, 30.0]


def test_frame_confusion_with_lag_and_valid_window():
    fps = 1.0
    gt = [0, 0, 1, 1, 1, 0, 0, None]
    changes = [(0.0, "ALERT"), (3.0, "HIGH"), (6.0, "ALERT")]
    c = m.frame_confusion(gt, fps, changes, min_rank=2)
    # frame centres 0.5..6.5 -> pred: F F F T T T F ; labels 0 0 1 1 1 0 0
    assert (c.tp, c.fp, c.fn, c.tn) == (2, 1, 1, 3)
    assert c.precision == pytest.approx(2 / 3) and c.recall == pytest.approx(2 / 3)
    assert c.f1 == pytest.approx(2 / 3)
    lagged = m.frame_confusion(gt, fps, changes, min_rank=2, lag_s=1.0)
    assert (lagged.tp, lagged.fp, lagged.fn, lagged.tn) == (3, 0, 0, 4)
    windowed = m.frame_confusion(gt, fps, changes, min_rank=2, valid=[(2.0, 5.0)])
    assert windowed.tp + windowed.fp + windowed.fn + windowed.tn == 3


def test_confusion_undefined_metrics_are_none():
    c = m.Confusion(tn=5)
    assert c.precision is None and c.recall is None and c.f1 is None
    assert (m.Confusion(tp=1) + m.Confusion(fn=1)).recall == 0.5


def test_repeat_intervals():
    assert m.repeat_intervals([12.0, 0.0, 6.0, 18.5]) == [6.0, 6.0, 6.5]
    assert m.repeat_intervals([3.0]) == []


def test_active_segments_from_energy():
    hop = 0.01
    db = [-90] * 10 + [-20] * 30 + [-90] * 5 + [-20] * 20 + [-90] * 50 + [-20] * 3
    seg = m.active_segments(db, hop, t0=100.0, threshold_db=-45, min_len_s=0.1, merge_gap_s=0.1)
    assert len(seg) == 1                                  # the 50 ms gap merges, 30 ms blip dropped
    assert seg[0][0] == pytest.approx(100.10) and seg[0][1] == pytest.approx(100.65)


# ---------------------------------------------------------------- judge helpers
def test_majority_and_agreement():
    assert m.majority(["yes", "yes", "no"]) == ("yes", pytest.approx(2 / 3))
    assert m.majority(["no", "no", "no"]) == ("no", 1.0)
    assert m.majority(["yes", "no", None]) == (None, pytest.approx(1 / 3))
    assert m.majority([None, None, None]) == (None, 0.0)


def test_cohen_kappa_known_values():
    a = ["yes"] * 20 + ["no"] * 30
    b = ["yes"] * 15 + ["no"] * 5 + ["yes"] * 10 + ["no"] * 20
    # po = (15 + 20)/50 = 0.7 ; pe = 0.4*0.5 + 0.6*0.5 = 0.5 ; kappa = 0.4
    assert m.cohen_kappa(a, b) == pytest.approx(0.4)
    assert m.cohen_kappa(["yes", "no"], ["yes", "no"]) == pytest.approx(1.0)
    assert m.cohen_kappa(["yes"] * 3, ["yes"] * 3) is None
    assert m.cohen_kappa([], []) is None


# ---------------------------------------------------------------- reply checks
def test_reply_checks():
    assert m.looks_english("Sure, here is a fun fact about the moon.")
    assert not m.looks_english("Это ответ на русском языке.")
    assert not m.looks_english("")
    assert m.repetition_ratio("take a break take a break take a break take a break") > 0.5
    assert m.repetition_ratio("How are you feeling today on this drive?") == 0.0
    assert m.word_count("Two words.") == 2
