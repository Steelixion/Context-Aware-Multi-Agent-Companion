"""Safety Agent rules (Implementation Table 6.2, ch. 6.3-6.4)."""
import asyncio

import pytest

from app.agents import safety as S
from app.agents.safety import SafetyAgent
from app.bus import Bus
from app.config import Config
from app.contracts import Telemetry
from app.state import Session


class FakeArbiter:
    def __init__(self):
        self.preempts, self.says = [], []

    async def preempt(self, text, lang="en"):
        self.preempts.append(text)

    async def say(self, text, lang="en"):
        self.says.append(text)


class FakeDB:
    def __init__(self):
        self.logs = []

    async def log_event(self, record_id, state, level, **kw):
        self.logs.append((state, level, kw.get("safety_prompt")))


def make():
    cfg, arb, db = Config(), FakeArbiter(), FakeDB()
    cancelled = []
    agent = SafetyAgent(cfg, Bus(), Session(user_id="u", record_id="r"), arb, db, lambda: cancelled.append(1))
    return agent, arb, db, cancelled


def tel(ts, **kw) -> dict:
    t = Telemetry(ts=ts, face_detected=True).model_dump()
    for k, v in kw.items():
        if isinstance(v, dict):
            t[k].update(v)
        else:
            t[k] = v
    return t


def feed(agent, seq):
    async def go():
        for t in seq:
            await agent._evaluate(t)
    asyncio.run(go())


def frames(start, seconds, fps=15, **kw):
    return [tel(start + i / fps, **kw) for i in range(int(seconds * fps))]


# --------------------------------------------------------------------------- HIGH
def test_high_preempts_and_repeats_every_6_s():
    agent, arb, db, cancelled = make()
    feed(agent, frames(0, 13, drowsiness_level="HIGH", risk=0.95, quality=1.0))
    assert len(arb.preempts) == 3 and cancelled == [1, 1, 1]          # t = 0, 6, 12
    assert arb.preempts[0] == S.WARNINGS["en"]


@pytest.mark.parametrize("kw,reason", [
    (dict(eyes_closed_ms=1600), "eyes_closed"), (dict(head_drop=True), "head_drop"), ({}, "sustained_fatigue")])
def test_high_reason(kw, reason):
    agent, _, _, _ = make()
    alerts = agent.bus.subscribe("alert")
    feed(agent, [tel(0, drowsiness_level="HIGH", **kw)])
    assert alerts.get_nowait()["reason"] == reason


def test_low_quality_never_alarms():
    agent, arb, _, _ = make()
    feed(agent, frames(0, 10, drowsiness_level="HIGH", quality=0.3))
    feed(agent, frames(10, 30, drowsiness_level="MILD", quality=0.3))
    assert arb.preempts == [] and S.CHECK_IN not in arb.says


def test_warning_uses_the_session_language():
    agent, arb, _, _ = make()
    agent.s.lang = "ur"
    feed(agent, [tel(0, drowsiness_level="HIGH")])
    assert arb.preempts == [S.WARNINGS["ur"]]


# --------------------------------------------------------------------------- MILD check-in
def test_mild_check_in_after_20_s_then_120_s_quiet():
    agent, arb, _, _ = make()
    feed(agent, frames(0, 19.9, drowsiness_level="MILD"))
    assert S.CHECK_IN not in arb.says
    feed(agent, frames(20, 100, drowsiness_level="MILD"))
    assert arb.says.count(S.CHECK_IN) == 1                     # t = 20, nothing until 140
    feed(agent, frames(120, 21, drowsiness_level="MILD"))
    assert arb.says.count(S.CHECK_IN) == 2
    assert arb.preempts == []                                   # a check-in is not an alarm


def test_mild_timer_restarts_when_alert_in_between():
    agent, arb, _, _ = make()
    feed(agent, frames(0, 15, drowsiness_level="MILD") + frames(15, 1, drowsiness_level="ALERT")
         + frames(16, 15, drowsiness_level="MILD"))
    assert S.CHECK_IN not in arb.says


# --------------------------------------------------------------------------- distraction, posture, face
def test_distraction_reminder_with_cooldown():
    agent, arb, _, _ = make()
    feed(agent, frames(0, 20, gaze=dict(distracted=True, on_road=False)))
    assert arb.says.count(S.DISTRACTED) == 2                   # t = 0 and 15


def test_posture_reminder_with_cooldown():
    agent, arb, _, _ = make()
    feed(agent, frames(0, 60, posture=dict(visible=True, unstable=True)))
    assert arb.says.count(S.POSTURE) == 1


def test_face_lost_reminder():
    agent, arb, _, _ = make()
    feed(agent, frames(0, 5.9, face_detected=False))
    assert S.FACE_LOST not in arb.says
    feed(agent, frames(5.9, 1, face_detected=False))
    assert arb.says.count(S.FACE_LOST) == 1


# --------------------------------------------------------------------------- logging
def test_state_changes_are_logged_once_and_debounced():
    agent, _, db, _ = make()
    feed(agent, frames(0, 2) + frames(2, 0.5, yawning=True) + frames(2.5, 2)       # 0.5 s flicker
         + frames(4.5, 2, yawning=True) + frames(6.5, 3))
    states = [s for s, _, prompt in db.logs if prompt is None]
    assert states == ["YAWNING", "ATTENTIVE"]


@pytest.mark.parametrize("kw,state", [
    (dict(face_detected=False), "NO_FACE"), (dict(eyes_closed_ms=1600), "EYES_CLOSED"),
    (dict(head_drop=True), "HEAD_DROP"), (dict(yawning=True), "YAWNING"),
    (dict(gaze=dict(distracted=True)), "DISTRACTED"), (dict(posture=dict(unstable=True)), "POSTURE"),
    ({}, "ATTENTIVE")])
def test_driver_state_labels(kw, state):
    assert S.driver_state(tel(0, **kw), 1500) == state


def test_alerts_are_logged_with_their_prompt():
    agent, _, db, _ = make()
    feed(agent, [tel(0, drowsiness_level="HIGH", eyes_closed_ms=1600)])
    assert ("EYES_CLOSED", "HIGH", S.WARNINGS["en"]) in db.logs


# --------------------------------------------------------------------------- robustness
def test_a_bad_message_does_not_stop_the_agent():
    agent, arb, _, _ = make()

    async def go():
        task = asyncio.create_task(agent.run())
        await asyncio.sleep(0)
        agent.bus.publish("telemetry", {"ts": 0.0})                 # missing fields
        agent.bus.publish("telemetry", tel(0.1, drowsiness_level="HIGH"))
        for _ in range(50):
            await asyncio.sleep(0.01)
            if arb.preempts:
                break
        task.cancel()
    asyncio.run(go())
    assert arb.preempts == [S.WARNINGS["en"]]


def test_watchdog_reports_lost_camera_once():
    agent, arb, db, _ = make()
    agent.last_tel -= 10
    asyncio.run(agent._watchdog())
    asyncio.run(agent._watchdog())
    assert arb.preempts == [S.SENSOR_LOST] and db.logs[-1][0] == "SENSOR_LOST"
