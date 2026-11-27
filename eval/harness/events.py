"""One monotonic clock and one append-only event log per case run.

Every event is {"t": <time.monotonic() seconds>, "src": ..., "type": ..., "data": {...}}.
App timestamps (time.time() on the same host) are converted with wall_to_mono(), using the
offset between the two clocks sampled at start and every 10 s (also logged, so it can be
audited: see the 'clock' events).
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any


class Clock:
    def __init__(self) -> None:
        self.samples: list[tuple[float, float]] = []
        self.sample()

    def now(self) -> float:
        return time.monotonic()

    def sample(self) -> tuple[float, float]:
        # bracket time.time() between two monotonic reads; use the midpoint
        a = time.monotonic()
        w = time.time()
        b = time.monotonic()
        s = ((a + b) / 2, w)
        self.samples.append(s)
        return s

    @property
    def offset(self) -> float:
        """wall - mono, from the latest sample."""
        m, w = self.samples[-1]
        return w - m

    def wall_to_mono(self, ts: float) -> float:
        return ts - self.offset


class EventLog:
    def __init__(self, path: Path, clock: Clock):
        self.path, self.clock = path, clock
        self.events: list[dict] = []
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(path, "w", encoding="utf-8")
        self._last_clock = 0.0
        self.add("harness", "clock", mono=clock.samples[-1][0], wall=clock.samples[-1][1])

    def add(self, src: str, type_: str, t: float | None = None, **data: Any) -> dict:
        ev = {"t": self.clock.now() if t is None else t, "src": src, "type": type_, "data": data}
        with self._lock:
            self.events.append(ev)
            self._fh.write(json.dumps(ev, default=str) + "\n")
            now = ev["t"]
            if now - getattr(self, "_last_flush", 0.0) > 1.0:   # crash-safe: at most ~1 s lost
                self._fh.flush()
                self._last_flush = now
            if now - self._last_clock > 10.0:
                self._last_clock = now
                m, w = self.clock.sample()
                ce = {"t": m, "src": "harness", "type": "clock", "data": {"mono": m, "wall": w}}
                self.events.append(ce)
                self._fh.write(json.dumps(ce) + "\n")
        return ev

    def select(self, src: str | None = None, type_: str | None = None) -> list[dict]:
        with self._lock:
            return [e for e in self.events if (src is None or e["src"] == src)
                    and (type_ is None or e["type"] == type_)]

    def close(self) -> None:
        with self._lock:
            self._fh.flush()
            self._fh.close()


def load_events(path: Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
