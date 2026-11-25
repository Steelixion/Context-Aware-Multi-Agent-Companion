"""Safety Agent: deterministic (no LLM). Consumes telemetry and speaks safety alerts.

Rules (Implementation ch. 6.3-6.4, Table 6.2):
  HIGH (R >= 0.85: microsleep, head drop or sustained fatigue) -> urgent warning, repeated every
      warning_repeat_seconds while HIGH
  MILD (0.50 <= R < 0.85) held for 20 s -> companion check-in, then none for 60 s
  eyes off the road for distraction_seconds -> companion reminder (cooldown)
  posture unstable (D_t > 1 for 2.5 s)      -> companion reminder (cooldown)
  face not visible for face_lost_seconds    -> spoken reminder
  no telemetry for watchdog_seconds         -> "camera signal lost" warning
Low-quality telemetry (head turned away, occlusion) never raises drowsiness alarms; the vision
pipeline already holds the risk, and nothing is spoken (6.4.3).
"""
from __future__ import annotations

import asyncio
import math
import time
import uuid

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


def all_alert_lines() -> list[tuple[str, str]]:
    """(text, lang) for every line the Safety Agent can speak, for prerendering."""
    return [(text, lang) for lang, text in WARNINGS.items()] + [
        (text, "en") for text in (SENSOR_LOST, FACE_LOST, CHECK_IN, DISTRACTED, POSTURE)]


class SafetyAgent:
    def __init__(self, cfg: Config, bus: Bus, session: Session, arbiter: AudioArbiter,
                 db: Database, _cancel_companion=None, nudge=None):
        self.cfg, self.bus, self.s, self.arbiter, self.db = cfg, bus, session, arbiter, db
        self.nudge = nudge
        never = -math.inf
        self.last_warn = self.last_checkin = self.last_distract = self.last_posture = never
        self.last_face_warn = self.last_visual_nudge = self.last_mild_nudge = never
        self.mild_active = False
        self.face_missing_since: float | None = None
        self.logged = ("ATTENTIVE", "ALERT")
        self.candidate, self.candidate_since = self.logged, 0.0
        self.last_tel = time.time()
        self.lost_notified = False
        self.got_telemetry = False

    async def run(self) -> None:
        q = self.bus.subscribe("telemetry", maxsize=8)
        speech_task = asyncio.create_task(self._watch_speech(), name="safety-speech")
        try:
            while True:
                try:
                    tel = await asyncio.wait_for(q.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    await self._watchdog()
                    continue
                self.last_tel, self.lost_notified, self.got_telemetry = time.time(), False, True
                self.s.telemetry = tel
                try:
                    await self._evaluate(tel)
                except Exception as e:  # noqa: BLE001 - one bad message must not stop the safety loop
                    print(f"[safety] evaluation failed: {e}", flush=True)
        finally:
            speech_task.cancel()

    async def _watch_speech(self) -> None:
        q = self.bus.subscribe("speech", maxsize=32)
        while True:
            msg = await q.get()
            if msg.get("source") != "safety":
                self.user_spoke()

    def user_spoke(self) -> None:
        now = time.time()
        self.last_warn = now
        self.last_checkin = now
        self.last_visual_nudge = now
        self.last_distract = now
        self.last_posture = now
        self.last_mild_nudge = now
        self.mild_active = True

    async def _evaluate(self, t: dict) -> None:
        cfg, now = self.cfg, t["ts"]
        level = t["drowsiness_level"]
        quality_ok = t.get("quality", 1.0) >= cfg.quality_min
        long_closure_ms = cfg.long_closure_seconds * 1000

        if level == "HIGH" and quality_ok:
            self.mild_active = False
            if now - self.last_checkin >= cfg.mild_checkin_cooldown:
                self.last_warn = now
                await self._advisory(
                    "drowsy", WARNINGS.get(self.s.lang, WARNINGS["en"]),
                    "The driver is showing strong signs of drowsiness. Ask where they are headed "
                    "or how they feel today, and gently suggest a safe break. One short question only.",
                    severity="CRITICAL")
        elif level == "MILD" and quality_ok:
            if not self.mild_active and now - self.last_mild_nudge >= min(cfg.mild_checkin_cooldown, 15.0):
                self.mild_active = True
                self.last_mild_nudge = now
                await self._nudge(
                    "The driver seems tired. Ask where they are headed or how they feel today, "
                    "and gently suggest a safe break.")
        else:
            self.mild_active = False

        state = driver_state(t, long_closure_ms)
        cue = None
        if t.get("eyes_closed_ms", 0) >= 600:
            cue = "closed their eyes"
        elif state == "YAWNING":
            cue = "yawned"
        elif t.get("head_drop_ms", 0) >= 600 or state == "HEAD_DROP":
            cue = "let their head drop"
        if (cue and level != "HIGH" and quality_ok
                and now - self.last_visual_nudge >= cfg.mild_checkin_cooldown
                and now - self.last_checkin >= cfg.mild_checkin_cooldown):
            await self._nudge(f"The driver just {cue}. Say one short caring sentence and suggest a safe break.")

        if (t.get("gaze") or {}).get("distracted") and now - self.last_distract >= cfg.distraction_cooldown:
            self.last_distract = now
            await self._advisory("distracted", DISTRACTED,
                                 "The driver looked away from the road. Say one short sentence to bring their attention back.")

        if (t.get("posture") or {}).get("unstable") and now - self.last_posture >= cfg.posture_cooldown:
            self.last_posture = now
            await self._advisory("posture", POSTURE,
                                 "The driver has slumped in their seat. Say one short sentence and suggest sitting upright.")

        if not t["face_detected"]:   # looking away for long / camera covered
            self.face_missing_since = self.face_missing_since or now
            if now - self.face_missing_since > cfg.face_lost_seconds and now - self.last_face_warn > 30:
                self.last_face_warn = now
                await self._raise("ADVISORY", FACE_LOST, t, "face_lost")
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
        if not self.got_telemetry:
            return
        if time.time() - self.last_tel > self.cfg.watchdog_seconds and not self.lost_notified:
            self.lost_notified = True
            await self._raise("CRITICAL", SENSOR_LOST, self.s.telemetry or {}, "sensor_lost")

    async def _nudge(self, prompt: str, fallback: str | None = None) -> None:
        self.last_checkin = time.time()
        self.last_visual_nudge = self.last_checkin
        if self.nudge:
            self.nudge(prompt)
        else:
            await self.arbiter.say(fallback or CHECK_IN, self.s.lang)

    async def _advisory(self, reason: str, text: str, prompt: str,
                        severity: str = "ADVISORY") -> None:
        self.bus.publish("alert", dict(ts=time.time(), severity=severity, reason=reason, text=text))
        await self._nudge(prompt, fallback=text)

    async def _raise(self, severity: str, text: str, t: dict, reason: str):
        self.bus.publish("alert", dict(ts=time.time(), severity=severity, reason=reason, text=text))
        m = dict(id=str(uuid.uuid4()), role="assistant", text=text,
                 final=True, ts=time.time(), source="safety")
        self.s.chat_log = (self.s.chat_log + [m])[-60:]
        self.bus.publish("chat", m)
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
