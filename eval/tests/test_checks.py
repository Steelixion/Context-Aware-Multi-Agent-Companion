"""Observation building and scenario checks on small synthetic event logs."""
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness import checks as C  # noqa: E402
from harness.observe import build  # noqa: E402

TH = yaml.safe_load((Path(__file__).resolve().parents[1] / "config" / "thresholds.yaml").read_text(encoding="utf-8"))
T0 = 1000.0          # clip start on the monotonic clock
WALL = 5000.0        # wall - mono offset used in these synthetic logs


def ev(t, src, typ, **d):
    return {"t": T0 + t, "src": src, "type": typ, "data": d}


def base_events():
    return [{"t": T0 - 5, "src": "harness", "type": "clock", "data": {"mono": T0 - 5, "wall": T0 - 5 + WALL}}]


def gt(events=(), fps=10.0, dur=60.0):
    n = int(fps * dur)
    labels = [0] * n
    for a, b in events:
        for i in range(int(a * fps), int(b * fps)):
            labels[i] = 1
    return {"fps": fps, "frame_labels": labels, "events": [list(e) for e in events]}


def alert(t, reason="drowsy"):
    return ev(t, "ws", "alert", ts=T0 + t + WALL, severity="CRITICAL", reason=reason, text="x")


def test_turn_latency_chain():
    q = "Can you suggest a good podcast"
    e = base_events() + [
        ev(2.0, "harness", "speech_start", audio_id="U01_x", event_id="e1"),
        ev(4.0, "harness", "speech_end", audio_id="U01_x", event_id="e1"),
        # a safety nudge starts its own LLM turn just before the transcript arrives; its bubble is
        # then interrupted by the driver's turn: it must NOT be taken as the reply (real app flow)
        ev(5.15, "ws", "speech", text="The driver seems tired...", source="safety"),
        ev(5.16, "ws", "chat", id="nudge", role="assistant", text="", final=False),
        ev(5.17, "llm", "llm_request", rid=9, messages=[{"role": "system", "content": "s"},
                                                        {"role": "user", "content": "The driver seems tired..."}]),
        ev(5.2, "ws", "speech", text=q, source="voice", ts=T0 + 5.2 + WALL),
        ev(5.21, "ws", "chat", id="u1", role="user", text=q, final=True),
        ev(5.22, "ws", "chat", id="nudge", role="assistant", text="", final=True, interrupted=True),
        ev(5.23, "ws", "chat", id="r1", role="assistant", text="", final=False),
        ev(5.25, "llm", "llm_request", rid=10, messages=[{"role": "system", "content": "sys"},
                                                         {"role": "user", "content": q}]),
        ev(6.0, "llm", "llm_first_token", rid=10),
        ev(7.0, "llm", "llm_done", rid=10, text="Try a history podcast."),
        ev(7.0, "ws", "chat", id="r1", role="assistant", text="Try a history podcast.", final=True),
        ev(7.01, "ws", "audio", speaking=True, priority=5, text="Try a history podcast."),
        ev(9.0, "ws", "audio", speaking=False, priority=5, text=""),
    ]
    lb = {"available": True, "t0": T0 + 0.0, "hop_s": 0.01, "db": [-90.0] * 750 + [-20.0] * 100 + [-90.0] * 100}
    obs = build(e, T0, 60.0, lb, TH)
    u = obs.turns[0]
    assert u.transcript == q
    assert round(u.transcripts[0][0] - u.speech_end, 3) == 1.2
    assert round(u.llm_request_t, 2) == 5.25                # the driver's request, not the nudge's
    assert round(u.llm_first_token_t - u.transcripts[0][0], 3) == 0.8
    assert u.reply_text == "Try a history podcast." and not u.reply_interrupted
    assert round(u.first_audio_event_t, 2) == 7.01
    assert round(u.first_audible_t, 2) == 7.5              # loopback onset, not the audio event
    assert round(u.first_audible_t - u.speech_end, 2) == 3.5


def test_s02_warning_in_time_and_missed_event():
    g = gt([(10.0, 20.0), (40.0, 45.0)])
    e = base_events() + [ev(0, "ws", "state", level="ALERT"), ev(12.0, "ws", "state", level="HIGH"),
                         alert(12.1), ev(13.0, "ws", "audio", speaking=True, priority=5, text="w"),
                         ev(16.0, "ws", "audio", speaking=False, priority=5, text=""),
                         ev(21.0, "ws", "state", level="ALERT")]
    obs = build(e, T0, 60.0, None, TH)
    case = {"scenario": "S02", "clip": {"start_s": 0.0}, "timeline": []}
    res = C.score(case, obs, g, TH, {})
    lat = [c for c in res if c["check"].startswith("warning_in_time")]
    assert lat[0]["status"] == "PASS" and lat[0]["observed"] == 3.0      # audio 3 s after onset (no loopback)
    assert lat[1]["status"] == "FAIL" and lat[1]["severity"] == 1        # second event never warned
    assert C.case_status(res, "DONE") == "FAIL"
    assert any(c["check"] == "no_false_warning" and c["status"] == "PASS" for c in res)


def test_s01_false_alarm_detected():
    obs = build(base_events() + [alert(30.0)], T0, 60.0, None, TH)
    res = C.score({"scenario": "S01", "clip": {"start_s": 0.0}, "timeline": []}, obs, gt(), TH, {})
    assert res[0]["status"] == "FAIL" and res[0]["observed"] == 1 and res[0]["severity"] == 2


def test_s12_watchdog_timing():
    e = base_events() + [ev(10.0, "harness", "kill_vision", pid=1), alert(13.2, "sensor_lost"),
                         ev(13.3, "ws", "audio", speaking=True, priority=5, text="lost")]
    obs = build(e, T0, 25.0, None, TH)
    res = C.score({"scenario": "S12", "clip": {"start_s": 0.0}, "timeline": []}, obs, gt(dur=25), TH, {})
    w = next(c for c in res if c["check"] == "camera_signal_lost_warning")
    assert w["status"] == "PASS" and w["observed"] == 3.2


def test_s05_inconclusive_without_mid_reply():
    e = base_events() + [ev(0, "ws", "state", level="HIGH"), alert(1.0)]
    obs = build(e, T0, 30.0, None, TH)
    res = C.score({"scenario": "S05", "clip": {"start_s": 0.0}, "timeline": []}, obs, gt(dur=30), TH, {})
    assert res[0]["status"] == "INCONCLUSIVE"
    assert C.case_status(res, "DONE") == "INCONCLUSIVE"


def test_s05_preemption_fails_when_chat_keeps_playing():
    e = base_events() + [ev(0.0, "ws", "audio", speaking=True, priority=5, text="long chat"),
                         alert(1.0), ev(6.0, "ws", "audio", speaking=False, priority=5, text=""),
                         ev(6.1, "ws", "audio", speaking=True, priority=5, text="warning via chat")]
    obs = build(e, T0, 30.0, None, TH)
    res = {c["check"]: c for c in C.score({"scenario": "S05", "clip": {"start_s": 0.0}, "timeline": []}, obs,
                                          gt(dur=30), TH, {})}
    assert res["chat_audio_stopped"]["status"] == "FAIL" and res["chat_audio_stopped"]["observed"] == 5.0
    assert res["warning_played_with_safety_priority"]["status"] == "FAIL"
    assert res["chat_muted_window"]["status"] == "FAIL"


def test_s06_repeat_interval():
    e = base_events() + [ev(0.0, "ws", "state", level="HIGH")] + [alert(t) for t in (1.0, 7.0, 13.1, 19.0)]
    obs = build(e, T0, 25.0, None, TH)
    res = C.score({"scenario": "S06", "clip": {"start_s": 0.0}, "timeline": []}, obs, gt([(0, 25)], dur=25), TH, {})
    assert res[0]["status"] == "PASS"
    e2 = base_events() + [ev(0.0, "ws", "state", level="HIGH")] + [alert(1.0)]
    res2 = C.score({"scenario": "S06", "clip": {"start_s": 0.0}, "timeline": []}, build(e2, T0, 25.0, None, TH),
                   gt([(0, 25)], dur=25), TH, {})
    assert res2[0]["status"] == "FAIL"


def test_s13_self_hearing():
    reply = "You seem a little tired, how about a short break at the next stop"
    e = base_events() + [ev(1.0, "ws", "chat", role="assistant", text=reply, final=True),
                         ev(1.1, "ws", "audio", speaking=True, priority=5, text=reply),
                         ev(5.0, "ws", "audio", speaking=False, priority=5, text=""),
                         ev(6.0, "ws", "speech", text="how about a short break at the next stop", source="voice", ts=T0 + 6 + WALL)]
    obs = build(e, T0, 20.0, None, TH)
    res = {c["check"]: c for c in C.score({"scenario": "S13", "clip": {"start_s": 0.0}, "timeline": []}, obs,
                                          gt(dur=20), TH, {})}
    assert res["no_self_hearing"]["status"] == "FAIL"
    assert res["no_unprompted_transcripts"]["status"] == "FAIL"


def test_observe_only_case_never_fails():
    obs = build(base_events() + [alert(30.0)], T0, 60.0, None, TH)
    res = C.score({"scenario": "S01", "mode": "observe", "clip": {"start_s": 0.0}, "timeline": []}, obs, gt(), TH, {})
    assert res[0]["status"] == "OBSERVED" and "would be FAIL" in res[0]["detail"]
    assert C.case_status(res, "DONE") == "OBSERVED"


def test_utterance_lost_to_talk_over_is_inconclusive_not_asr_failure():
    e = base_events() + [
        ev(10.5, "harness", "speech_start", audio_id="U02_x", event_id="e1"),
        ev(12.0, "ws", "audio", speaking=True, priority=5, text="Are you feeling okay?"),   # companion talks over
        ev(12.6, "harness", "speech_end", audio_id="U02_x", event_id="e1"),
        ev(14.0, "ws", "audio", speaking=False, priority=5, text=""),
    ]
    obs = build(e, T0, 40.0, None, TH)
    case = {"scenario": "S03", "clip": {"start_s": 0.0},
            "timeline": [{"id": "e1", "at_s": 10.0, "action": "speak", "audio_id": "U02_x"}]}
    res = {c["check"]: c for c in C.score(case, obs, gt(dur=40), TH, {"U02_x": {"text": "Tell me a fact", "snr_key": "clean"}})}
    assert res["e1:transcript_received"]["status"] == "INCONCLUSIVE"
    assert "echo guard" in res["e1:transcript_received"]["detail"]
    assert "e1:reply_received" not in res                  # no double-counting


def test_utterance_without_transcript_and_no_talk_over_fails():
    e = base_events() + [ev(10.5, "harness", "speech_start", audio_id="U02_x", event_id="e1"),
                         ev(12.6, "harness", "speech_end", audio_id="U02_x", event_id="e1")]
    obs = build(e, T0, 40.0, None, TH)
    case = {"scenario": "S03", "clip": {"start_s": 0.0},
            "timeline": [{"id": "e1", "at_s": 10.0, "action": "speak", "audio_id": "U02_x"}]}
    res = {c["check"]: c for c in C.score(case, obs, gt(dur=40), TH, {"U02_x": {"text": "Tell me a fact", "snr_key": "clean"}})}
    assert res["e1:transcript_received"]["status"] == "FAIL"


def test_s13_own_transcript_is_not_unprompted():
    e = base_events() + [ev(8.0, "harness", "speech_start", audio_id="U01_x", event_id="e1"),
                         ev(10.0, "harness", "speech_end", audio_id="U01_x", event_id="e1"),
                         ev(11.5, "ws", "speech", text="Can you suggest a podcast", source="voice", ts=T0 + 11.5 + WALL)]
    obs = build(e, T0, 30.0, None, TH)
    case = {"scenario": "S13", "clip": {"start_s": 0.0},
            "timeline": [{"id": "e1", "at_s": 8.0, "action": "speak", "audio_id": "U01_x"}]}
    res = {c["check"]: c for c in C.score(case, obs, gt(dur=30), TH, {"U01_x": {"text": "Can you suggest a podcast"}})}
    assert res["no_unprompted_transcripts"]["status"] == "PASS"
    assert res["no_self_hearing"]["status"] == "PASS"


def test_s05_without_any_warning_is_inconclusive():
    obs = build(base_events() + [ev(0, "ws", "state", level="MILD")], T0, 30.0, None, TH)
    res = C.score({"scenario": "S05", "clip": {"start_s": 0.0}, "timeline": []}, obs, gt(dur=30), TH, {})
    assert res[0]["status"] == "INCONCLUSIVE"


def test_transcript_goes_to_the_utterance_that_just_ended():
    # B01 (barge-in) starts first and is dropped by the app; U02 is spoken later and transcribed.
    e = base_events() + [
        ev(9.0, "harness", "speech_start", audio_id="B01_x", event_id="e2"),
        ev(10.5, "harness", "speech_end", audio_id="B01_x", event_id="e2"),
        ev(12.0, "harness", "speech_start", audio_id="U02_x", event_id="e1"),
        ev(14.0, "harness", "speech_end", audio_id="U02_x", event_id="e1"),
        ev(15.3, "ws", "speech", text="Tell me an interesting fact", source="voice", ts=T0 + 15.3 + WALL),
    ]
    obs = build(e, T0, 30.0, None, TH)
    by = {u.event_id: u for u in obs.turns}
    assert by["e1"].transcript == "Tell me an interesting fact"
    assert by["e2"].transcripts == []
