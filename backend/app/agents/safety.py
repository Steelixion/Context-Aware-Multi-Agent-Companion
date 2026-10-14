"""Safety Agent: deterministic (no LLM), consumes telemetry, preempts everything on danger."""
from __future__ import annotations

import asyncio
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
ADVISORY = "You seem tired. Consider taking a break soon."


class SafetyAgent:
    def __init__(self, cfg: Config, bus: Bus, session: Session, arbiter: AudioArbiter,
                 db: Database, cancel_companion: Callable[[], None]):
        self.cfg, self.bus, self.s, self.arbiter, self.db = cfg, bus, session, arbiter, db
        self.cancel_companion = cancel_companion
        self.closed_since: float | None = None
        self.last_warn = 0.0
        self.last_advisory = 0.0
        self.last_face_warn = 0.0
        self.face_missing_since: float | None = None
        self.last_level = "ALERT"
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
            await self._evaluate(tel)

    async def _evaluate(self, t: dict) -> None:
        now, cfg = t["ts"], self.cfg
        # --- spec rule: EAR < 0.20 continuously for > 1.5 s
        if t["face_detected"] and t["ear"] < cfg.ear_threshold:
            self.closed_since = self.closed_since or now
        else:
            self.closed_since = None
        eyes_closed = self.closed_since is not None and now - self.closed_since > cfg.ear_seconds
        critical = eyes_closed or t["fatigue_score"] >= cfg.fatigue_high

        if critical and now - self.last_warn >= cfg.warning_repeat_seconds:
            self.last_warn = now
            reason = "eyes_closed" if eyes_closed else "sustained_fatigue"
            await self._raise("CRITICAL", WARNINGS.get(self.s.lang, WARNINGS["en"]), t, reason)
        elif t["drowsiness_level"] == "MODERATE" and not critical and now - self.last_advisory > 60:
            self.last_advisory = now
            await self._raise("ADVISORY", ADVISORY, t, "moderate_fatigue", preempt=False)

        # face lost for a while (looking away / covered camera)
        if not t["face_detected"]:
            self.face_missing_since = self.face_missing_since or now
            if now - self.face_missing_since > cfg.face_lost_seconds and now - self.last_face_warn > 30:
                self.last_face_warn = now
                await self._raise("ADVISORY", FACE_LOST, t, "face_lost", preempt=False)
        else:
            self.face_missing_since = None

        if t["drowsiness_level"] != self.last_level:
            self.last_level = t["drowsiness_level"]
            await self._log(t, None)
        self.bus.publish("state", dict(level=t["drowsiness_level"], fatigue=t["fatigue_score"],
                                       safety_active=self.s.safety_active))

    async def _watchdog(self) -> None:
        if time.time() - self.last_tel > self.cfg.watchdog_seconds and not self.lost_notified:
            self.lost_notified = True
            await self._raise("CRITICAL", SENSOR_LOST, self.s.telemetry or {}, "sensor_lost")

    async def _raise(self, severity: str, text: str, t: dict, reason: str, preempt: bool = True):
        self.bus.publish("alert", dict(ts=time.time(), severity=severity, reason=reason, text=text))
        if preempt:
            self.cancel_companion()                    # stop generating chat
            await self.arbiter.preempt(text, self.s.lang)  # mute chat + warn
        else:
            await self.arbiter.say(text, self.s.lang)
        await self._log(t, text)

    async def _log(self, t: dict, prompt: str | None) -> None:
        try:
            await self.db.log_event(
                self.s.record_id,
                "EYES_CLOSED" if (t.get("eyes_closed_ms", 0) > 0) else
                ("YAWNING" if t.get("mar", 0) > 0.6 else
                 ("NO_FACE" if t and not t.get("face_detected", True) else "ATTENTIVE")),
                t.get("drowsiness_level", "HIGH"), fatigue_score=t.get("fatigue_score"),
                ear=t.get("ear"), mar=t.get("mar"), safety_prompt=prompt, emotion=self.s.emotion)
        except Exception as e:  # noqa: BLE001
            print(f"[safety] log failed: {e}")
