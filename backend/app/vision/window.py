"""Sliding window of per-frame spatial features -> sustained-fatigue statistics."""
from __future__ import annotations

from collections import deque

import numpy as np

# columns
TS, EAR, MAR, YAW, PITCH, ROLL, SLUMP, FACE = range(8)
N_FEATURES = 8


class SlidingWindow:
    def __init__(self, seconds: float = 30.0, ear_thr: float = 0.20):
        self.seconds = seconds
        self.ear_thr = ear_thr
        self.buf: deque[tuple] = deque()

    def push(self, ts, ear, mar, yaw, pitch, roll, slump, face) -> None:
        self.buf.append((ts, ear, mar, yaw, pitch, roll, slump, 1.0 if face else 0.0))
        while self.buf and ts - self.buf[0][TS] > self.seconds:
            self.buf.popleft()

    def array(self) -> np.ndarray:
        return np.array(self.buf, dtype=np.float32) if self.buf else np.zeros((0, N_FEATURES), np.float32)

    def stats(self) -> dict:
        a = self.array()
        a = a[a[:, FACE] > 0] if len(a) else a
        if len(a) < 5:
            return dict(perclos=0.0, yawns=0, head_down=0.0, slump=0.0, longest_closure=0.0)
        closed = a[:, EAR] < self.ear_thr
        perclos = float(closed.mean())
        longest = _longest_run(a[:, TS], closed)
        yawns = _count_runs(a[:, TS], a[:, MAR] > 0.6, min_len=1.0)
        pitch_dev = a[:, PITCH] - np.median(a[:, PITCH])
        head_down = float((pitch_dev < -15).mean())
        return dict(perclos=perclos, yawns=yawns, head_down=head_down,
                    slump=float(a[:, SLUMP].mean()), longest_closure=longest)


def _runs(ts, mask):
    start = None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            yield ts[i - 1] - ts[start]
            start = None
    if start is not None:
        yield ts[-1] - ts[start]


def _longest_run(ts, mask) -> float:
    return float(max(_runs(ts, mask), default=0.0))


def _count_runs(ts, mask, min_len: float) -> int:
    return sum(1 for d in _runs(ts, mask) if d >= min_len)
