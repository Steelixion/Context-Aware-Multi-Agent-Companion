"""Pure scoring functions. No I/O, no app knowledge: everything here is unit-tested on small
synthetic inputs (eval/tests/test_metrics.py).

Time is always seconds on one axis (the caller converts app timestamps to clip time first).
"""
from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable, Sequence

# --------------------------------------------------------------------------- text / WER
_NUM_WORDS = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
              "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten"}


# British -> American spellings (so a correct transcript in either spelling is not an error)
_SPELLING = {"tyre": "tire", "tyres": "tires", "colour": "color", "colours": "colors", "favourite": "favorite",
             "favourites": "favorites", "centre": "center", "metre": "meter", "metres": "meters",
             "kilometre": "kilometer", "kilometres": "kilometers", "litre": "liter", "litres": "liters",
             "travelling": "traveling", "travelled": "traveled", "neighbour": "neighbor", "programme": "program",
             "organise": "organize", "realise": "realize", "realised": "realized", "apologise": "apologize",
             "analyse": "analyze", "theatre": "theater", "grey": "gray", "humour": "humor", "labour": "labor",
             "honour": "honor", "behaviour": "behavior", "mum": "mom", "cheque": "check", "kerb": "curb",
             "aeroplane": "airplane", "jewellery": "jewelry", "pyjamas": "pajamas", "defence": "defense",
             "licence": "license", "offence": "offense", "practise": "practice", "towards": "toward"}


def normalize_text(text: str) -> list[str]:
    """Normalisation used for every WER number in the report (documented in README):
    Unicode NFKC, lowercase, curly quotes -> straight, hyphens and slashes -> spaces,
    every character that is not a letter, digit, apostrophe or space removed, apostrophes
    removed (so "it's" == "its"), standalone digits 0-10 spelled out, British
    spellings mapped to American (tyre -> tire), whitespace collapsed.
    """
    t = unicodedata.normalize("NFKC", text or "").lower()
    t = t.replace("’", "'").replace("‘", "'").replace("`", "'")
    t = re.sub(r"[-/_]", " ", t)
    t = re.sub(r"[^\w' ]+", " ", t)
    t = t.replace("'", "")
    return [_SPELLING.get(w, _NUM_WORDS.get(w, w)) for w in t.split()]


@dataclass
class WerResult:
    wer: float
    substitutions: int
    deletions: int
    insertions: int
    ref_words: int
    hyp_words: int


def word_error_rate(reference: str, hypothesis: str) -> WerResult:
    """Levenshtein alignment over normalised words. WER = (S + D + I) / N.
    Empty reference: WER is 0 if the hypothesis is also empty, else 1.0 per inserted word
    is undefined, so we report float('inf') to make it impossible to miss."""
    ref, hyp = normalize_text(reference), normalize_text(hypothesis)
    n, m = len(ref), len(hyp)
    if n == 0:
        return WerResult(0.0 if m == 0 else math.inf, 0, 0, m, 0, m)
    # dp[i][j] = (cost, S, D, I)
    prev = [(j, 0, 0, j) for j in range(m + 1)]
    for i in range(1, n + 1):
        cur = [(i, 0, i, 0)] + [None] * m  # type: ignore[list-item]
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                cur[j] = prev[j - 1]
                continue
            s, d, ins = prev[j - 1], prev[j], cur[j - 1]
            best = min((s[0], 0), (d[0], 1), (ins[0], 2))
            if best[1] == 0:
                cur[j] = (s[0] + 1, s[1] + 1, s[2], s[3])
            elif best[1] == 1:
                cur[j] = (d[0] + 1, d[1], d[2] + 1, d[3])
            else:
                cur[j] = (ins[0] + 1, ins[1], ins[2], ins[3] + 1)
        prev = cur
    cost, sub, dele, ins = prev[m]
    return WerResult(cost / n, sub, dele, ins, n, m)


def corpus_wer(pairs: Iterable[tuple[str, str]]) -> float | None:
    """Total errors / total reference words over many utterances (not a mean of WERs)."""
    errs = words = 0
    for ref, hyp in pairs:
        r = word_error_rate(ref, hyp)
        errs += r.substitutions + r.deletions + r.insertions
        words += r.ref_words
    return errs / words if words else None


def ngram_overlap(a: str, b: str, n: int = 3) -> float:
    """Share of a's word n-grams that also occur in b (0..1). Used for self-hearing: how much
    of a transcript is the companion's own recent speech. Falls back to unigrams for short a."""
    wa, wb = normalize_text(a), normalize_text(b)
    if not wa:
        return 0.0
    k = n if len(wa) >= n else 1
    ga = [tuple(wa[i:i + k]) for i in range(len(wa) - k + 1)]
    gb = {tuple(wb[i:i + k]) for i in range(len(wb) - k + 1)}
    return sum(g in gb for g in ga) / len(ga)


# --------------------------------------------------------------------------- percentiles
def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolated percentile (numpy's default 'linear' method), q in [0, 100].
    None for an empty input, so 'no data' can never be mistaken for 0."""
    xs = sorted(v for v in values if v is not None and not math.isnan(v))
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def latency_summary(values: Sequence[float]) -> dict:
    xs = [v for v in values if v is not None and not math.isnan(v)]
    return dict(n=len(xs), p50=percentile(xs, 50), p95=percentile(xs, 95),
                p99=percentile(xs, 99), max=max(xs) if xs else None,
                mean=(sum(xs) / len(xs)) if xs else None)


# --------------------------------------------------------------------------- intervals
def intervals_from_labels(labels: Sequence[int], fps: float, positive: int = 1) -> list[tuple[float, float]]:
    """Frame labels -> [(onset_s, offset_s)] of runs equal to `positive`. Frame i covers
    [i/fps, (i+1)/fps), so a run of frames a..b gives (a/fps, (b+1)/fps)."""
    out, start = [], None
    for i, v in enumerate(labels):
        if v == positive and start is None:
            start = i
        elif v != positive and start is not None:
            out.append((start / fps, i / fps))
            start = None
    if start is not None:
        out.append((start / fps, len(labels) / fps))
    return out


def merge_intervals(iv: Iterable[tuple[float, float]], gap: float = 0.0) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for a, b in sorted(iv):
        if out and a - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def in_any(t: float, iv: Sequence[tuple[float, float]], before: float = 0.0, after: float = 0.0) -> bool:
    return any(a - before <= t < b + after for a, b in iv)


# --------------------------------------------------------------------------- level timeline
LEVEL_RANK = {"ALERT": 0, "MILD": 1, "HIGH": 2}


def step_value(changes: Sequence[tuple[float, str]], t: float, default: str | None = None) -> str | None:
    """Value of a step function given sorted (time, value) change points, at time t."""
    lo, hi, val = 0, len(changes), default
    while lo < hi:                       # last change with time <= t
        mid = (lo + hi) // 2
        if changes[mid][0] <= t:
            lo = mid + 1
        else:
            hi = mid
    return changes[lo - 1][1] if lo else val


def compress_changes(samples: Iterable[tuple[float, str]]) -> list[tuple[float, str]]:
    """Sorted samples -> only the points where the value changes."""
    out: list[tuple[float, str]] = []
    for t, v in sorted(samples, key=lambda x: x[0]):
        if not out or out[-1][1] != v:
            out.append((t, v))
    return out


@dataclass
class Confusion:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    @property
    def precision(self) -> float | None:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else None

    @property
    def recall(self) -> float | None:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else None

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        if p is None or r is None or p + r == 0:
            return None if p is None or r is None else 0.0
        return 2 * p * r / (p + r)

    def __add__(self, o: "Confusion") -> "Confusion":
        return Confusion(self.tp + o.tp, self.fp + o.fp, self.fn + o.fn, self.tn + o.tn)


def frame_confusion(gt_labels: Sequence[int | None], fps: float, changes: Sequence[tuple[float, str]],
                    min_rank: int, lag_s: float = 0.0, valid: Sequence[tuple[float, float]] | None = None) -> Confusion:
    """Per ground-truth frame: positive if label == 1; predicted positive if the app's level
    rank (step function, evaluated at frame-centre time + lag_s) >= min_rank. Frames with a
    None label, before the first observed level, or outside `valid` windows are skipped."""
    c = Confusion()
    first = changes[0][0] if changes else math.inf
    for i, y in enumerate(gt_labels):
        if y is None:
            continue
        t = (i + 0.5) / fps
        if valid is not None and not in_any(t, valid):
            continue
        tq = t + lag_s
        if tq < first:
            continue
        lvl = step_value(changes, tq)
        pred = LEVEL_RANK.get(lvl or "ALERT", 0) >= min_rank
        if y == 1 and pred:
            c.tp += 1
        elif y == 1:
            c.fn += 1
        elif pred:
            c.fp += 1
        else:
            c.tn += 1
    return c


# --------------------------------------------------------------------------- events
@dataclass
class EventMatch:
    onset: float
    offset: float
    detected: bool
    first_time: float | None = None
    delay: float | None = None


def first_crossing(changes: Sequence[tuple[float, str]], min_rank: int, start: float, end: float) -> float | None:
    """First time in [start, end) at which the level rank is >= min_rank (including a level
    that was already high at `start`)."""
    v = step_value(changes, start)
    if v is not None and LEVEL_RANK.get(v, 0) >= min_rank:
        return start
    for t, lvl in changes:
        if start <= t < end and LEVEL_RANK.get(lvl, 0) >= min_rank:
            return t
    return None


def match_events(gt_events: Sequence[tuple[float, float]], changes: Sequence[tuple[float, str]],
                 min_rank: int, early_s: float, late_s: float) -> list[EventMatch]:
    """An event (onset, offset) is detected if the level reaches min_rank in
    [onset - early_s, offset + late_s). Delay = detection time - onset (negative = early,
    within the allowed early window)."""
    out = []
    for a, b in gt_events:
        t = first_crossing(changes, min_rank, a - early_s, b + late_s)
        out.append(EventMatch(a, b, t is not None, t, None if t is None else t - a))
    return out


def warnings_outside(warning_times: Sequence[float], gt_events: Sequence[tuple[float, float]],
                     before_s: float, after_s: float) -> list[float]:
    """Warnings not inside any GT drowsy event (widened by before_s / after_s): false alarms."""
    return [t for t in warning_times if not in_any(t, gt_events, before_s, after_s)]


def intervals_complement(iv: Sequence[tuple[float, float]], start: float, end: float) -> list[tuple[float, float]]:
    out, cur = [], start
    for a, b in merge_intervals(iv):
        if a > cur:
            out.append((cur, min(a, end)))
        cur = max(cur, b)
    if cur < end:
        out.append((cur, end))
    return [(a, b) for a, b in out if b > a]


def total_length(iv: Iterable[tuple[float, float]]) -> float:
    return sum(b - a for a, b in iv)


def repeat_intervals(times: Sequence[float]) -> list[float]:
    ts = sorted(times)
    return [b - a for a, b in zip(ts, ts[1:])]


# --------------------------------------------------------------------------- segments
def active_segments(levels_db: Sequence[float], hop_s: float, t0: float, threshold_db: float,
                    min_len_s: float, merge_gap_s: float) -> list[tuple[float, float]]:
    """Frame energies (dBFS per hop) -> sounding segments (start, end) in seconds."""
    raw, start = [], None
    for i, db in enumerate(levels_db):
        on = db >= threshold_db
        if on and start is None:
            start = i
        elif not on and start is not None:
            raw.append((t0 + start * hop_s, t0 + i * hop_s))
            start = None
    if start is not None:
        raw.append((t0 + start * hop_s, t0 + len(levels_db) * hop_s))
    merged = merge_intervals(raw, merge_gap_s)
    return [(a, b) for a, b in merged if b - a >= min_len_s]


# --------------------------------------------------------------------------- judge
def majority(verdicts: Sequence[str | None]) -> tuple[str | None, float]:
    """Majority verdict over runs and agreement (share of runs agreeing with it). Ties and
    all-None give (None, share)."""
    vs = [v for v in verdicts if v is not None]
    if not vs:
        return None, 0.0
    counts: dict[str, int] = {}
    for v in vs:
        counts[v] = counts.get(v, 0) + 1
    best = max(counts.values())
    winners = [k for k, n in counts.items() if n == best]
    share = best / len(verdicts)
    return (winners[0] if len(winners) == 1 else None), share


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float | None:
    """Cohen's kappa for two raters over the same items. None if undefined (no items, or
    both raters constant and identical: chance agreement 1)."""
    if len(a) != len(b) or not a:
        return None
    n = len(a)
    labels = sorted(set(a) | set(b))
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((list(a).count(l) / n) * (list(b).count(l) / n) for l in labels)
    if pe >= 1.0:
        return None
    return (po - pe) / (1 - pe)


# --------------------------------------------------------------------------- reply checks
def looks_english(text: str) -> bool:
    """Cheap, deterministic language check: mostly Latin letters and at least one common
    English function word (for texts of 3+ words)."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False
    latin = sum("a" <= ch.lower() <= "z" for ch in letters) / len(letters)
    words = normalize_text(text)
    common = {"the", "a", "an", "and", "or", "you", "i", "is", "are", "to", "of", "it", "that",
              "for", "on", "in", "with", "your", "how", "what", "can", "do", "be", "this", "me",
              "my", "we", "let", "sure", "yes", "no", "please", "take", "here", "about", "there"}
    return latin >= 0.9 and (len(words) < 3 or any(w in common for w in words))


def repetition_ratio(text: str, n: int = 3) -> float:
    """Share of repeated word n-grams (0 = none). High values mean looping output."""
    w = normalize_text(text)
    if len(w) < n * 2:
        return 0.0
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def word_count(text: str) -> int:
    return len(normalize_text(text))
