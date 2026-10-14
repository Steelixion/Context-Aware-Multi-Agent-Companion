from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class Session:
    user_id: str
    record_id: str
    name: str = "Driver"
    lang: str = "en"
    telemetry: dict | None = None
    emotion: str = "neutral"
    history: list = field(default_factory=list)          # [{"role","content"}], recent turns
    turn_queue: deque = field(default_factory=lambda: deque(maxlen=200))  # short-term session queue
    chat_log: list = field(default_factory=list)         # for UI snapshot
    safety_hold_until: float = 0.0

    @property
    def safety_active(self) -> bool:
        return time.time() < self.safety_hold_until

    @property
    def level(self) -> str:
        return (self.telemetry or {}).get("drowsiness_level", "ALERT")

    @property
    def fatigue(self) -> float:
        return (self.telemetry or {}).get("fatigue_score", 0.0)
