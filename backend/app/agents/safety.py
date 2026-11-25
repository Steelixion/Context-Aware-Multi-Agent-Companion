"""Safety Agent: deterministic (no LLM). Consumes telemetry and preempts everything on danger.

Rules (Implementation ch. 6.3-6.4, Table 6.2):
  HIGH (R >= 0.85: microsleep, head drop or sustained fatigue) -> urgent warning; all chat is
      stopped and muted (preempt); repeated every warning_repeat_seconds while HIGH
  MILD (0.50 <= R < 0.85) held for 20 s -> spoken check-in, then none for 120 s
  eyes off the road for distraction_seconds -> spoken reminder (cooldown)
  posture unstable (D_t > 1 for 5 s)        -> spoken reminder (cooldown)
  face not visible for face_lost_seconds    -> spoken reminder
  no telemetry for watchdog_seconds         -> "camera signal lost" warning
Low-quality telemetry (head turned away, occlusion) never raises drowsiness alarms; the vision
pipeline already holds the risk, and nothing is spoken (6.4.3).
"""
from __future__ import annotations

import asyncio
import math
import time
from typing import Callable

from ..audio.arbiter import AudioArbiter
from ..bus import Bus
from ..config import Config
from ..db.relational import Database
from ..state import Session

WARNINGS = {
    "en": "Warning! You appear to be falling asleep. Please pull over safely and rest.",
    "es": "¡Atención! Parece que te estás durmiendo. Por favor, detente en un lugar seguro.",
    "fr": "Attention ! Vous semblez vous endormir. Arrêtez-vous en toute sécurité.",
    "de": "Achtung! Sie scheinen einzuschlafen. Bitte halten Sie sicher an und ruhen Sie sich aus.",
    "ur": "خبردار! آپ کو نیند آ رہی ہے۔ براہ کرم محفوظ جگہ گاڑی روک کر آرام کریں۔",
}
SENSOR_LOST = "Driver monitoring camera signal lost. Please stay alert."
FACE_LOST = "I can't see your face. Please keep your eyes on the road and face the camera."
CHECK_IN = "You seem a little tired. How are you feeling? A short break could help."
DISTRACTED = "Please keep your eyes on the road."
POSTURE = "Your posture has slumped. Try sitting upright, and take a break if you feel tired."
STATE_DEBOUNCE_S = 1.0   # a driver-state change is logged once it has held this long


def driver_state(t: dict, long_closure_ms: float) -> str:
    if not t.get("face_detected", False):
        return "NO_FACE"
    if t.get("eyes_closed_ms", 0) >= long_closure_ms:
        return "EYES_CLOSED"
    if t.get("head_drop"):
        return "HEAD_DROP"
    if t.get("yawning"):
        return "YAWNING"
    if (t.get("gaze") or {}).get("distracted"):
        return "DISTRACTED"
    if (t.get("posture") or {}).get("unstable"):
        return "POSTURE"
    return "ATTENTIVE"


class SafetyAgent:
    def __init__(self, cfg: Config, bus: Bus, session: Session, arbiter: AudioArbiter,
                 db: Database, cancel_companion: Callable[[], None]):
        self.cfg, self.bus, self.s, self.arbiter, self.db = cfg, bus, session, arbiter, db
        self.cancel_companion = cancel_companion
        never = -math.inf
        self.last_warn = self.last_checkin = self.last_distract = self.last_posture = never
        self.last_face_warn = never
        self.mild_since: float | None = None
        self.face_missing_since: float | None = None
        self.logged = ("ATTENTIVE", "ALERT")
        self.candidate, self.candidate_since = self.logged, 0.0
        self.last_tel = time.time()
        self.lost_notified = False

    async def run(self) -> None:
        q = self.bus.subscribe("telemetry", maxsize=8)
        while True:
            try:
                tel = await asyncio.wait_for(q.get(), timeout=1.0)
            except asyncio.TimeoutError:
                await self._watchdog()
                continue
            self.last_tel, self.lost_notified = time.time(), False
            self.s.telemetry = tel
            try:
                await self._evaluate(tel)
            except Exception as e:  # noqa: BLE001 - one bad message must not stop the safety loop
                print(f"[safety] evaluation failed: {e}", flush=True)

    async def _evaluate(self, t: dict) -> None:
        cfg, now = self.cfg, t["ts"]
        level = t["drowsiness_level"]
        quality_ok = t.get("quality", 1.0) >= cfg.quality_min
        long_closure_ms = cfg.long_closure_seconds * 1000

        if level == "HIGH" and quality_ok:
            if now - self.last_warn >= cfg.warning_repeat_seconds:
                self.last_warn = now
                reason = ("eyes_closed" if t.get("eyes_closed_ms", 0) >= long_closure_ms else
                          "head_drop" if t.get("head_drop") else "sustained_fatigue")
                await self._raise("CRITICAL", WARNINGS.get(self.s.lang, WARNINGS["en"]), t, reason)

        if level == "MILD" and quality_ok:
            self.mild_since = self.mild_since or now
            if (now - self.mild_since >= cfg.mild_checkin_seconds
                    and now - self.last_checkin >= cfg.mild_checkin_cooldown):
                self.last_checkin = now
                await self._raise("ADVISORY", CHECK_IN, t, "mild_fatigue", preempt=False)
        else:
            self.mild_since = None

        if (t.get("gaze") or {}).get("distracted") and now - self.last_distract >= cfg.distraction_cooldown:
            self.last_distract = now
            await self._raise("ADVISORY", DISTRACTED, t, "distracted", preempt=False)

        if (t.get("posture") or {}).get("unstable") and now - self.last_posture >= cfg.posture_cooldown:
            self.last_posture = now
            await self._raise("ADVISORY", POSTURE, t, "posture", preempt=False)

        if not t["face_detected"]:   # looking away for long / camera covered
            self.face_missing_since = self.face_missing_since or now
            if now - self.face_missing_since > cfg.face_lost_seconds and now - self.last_face_warn > 30:
                self.last_face_warn = now
                await self._raise("ADVISORY", FACE_LOST, t, "face_lost", preempt=False)
        else:
            self.face_missing_since = None

        await self._log_state(t, driver_state(t, long_closure_ms), level, now)
        self.bus.publish("state", dict(level=level, fatigue=t["fatigue_score"], risk=t.get("risk", 0.0),
                                       safety_active=self.s.safety_active))

    async def _log_state(self, t: dict, state: str, level: str, now: float) -> None:
        """Log a (state, level) change once it has held STATE_DEBOUNCE_S (no flicker spam)."""
        key = (state, level)
        if key != self.candidate:
            self.candidate, self.candidate_since = key, now
        elif key != self.logged and now - self.candidate_since >= STATE_DEBOUNCE_S:
            self.logged = key
            await self._log(t, None, state)

    async def _watchdog(self) -> None:
        if time.time() - self.last_tel > self.cfg.watchdog_seconds and not self.lost_notified:
            self.lost_notified = True
            await self._raise("CRITICAL", SENSOR_LOST, self.s.telemetry or {}, "sensor_lost")

    async def _raise(self, severity: str, text: str, t: dict, reason: str, preempt: bool = True):
        self.bus.publish("alert", dict(ts=time.time(), severity=severity, reason=reason, text=text))
        if preempt:
            self.cancel_companion()                        # stop generating chat
            await self.arbiter.preempt(text, self.s.lang)  # mute chat + warn
        else:
            await self.arbiter.say(text, self.s.lang)
        if reason == "sensor_lost" or not t:
            state = "SENSOR_LOST"
        else:
            state = driver_state(t, self.cfg.long_closure_seconds * 1000)
        await self._log(t, text, state)

    async def _log(self, t: dict, prompt: str | None, state: str) -> None:
        try:
            await self.db.log_event(
                self.s.record_id, state, t.get("drowsiness_level", "HIGH") if t else "HIGH",
                fatigue_score=t.get("risk", t.get("fatigue_score")) if t else None,
                ear=t.get("ear") if t else None, mar=t.get("mar") if t else None,
                safety_prompt=prompt, emotion=self.s.emotion)
        except Exception as e:  # noqa: BLE001
            print(f"[safety] log failed: {e}", flush=True)
