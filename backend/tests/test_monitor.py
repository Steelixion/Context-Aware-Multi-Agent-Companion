"""DriverMonitor: events, PERCLOS, quality, temporal model, risk (Implementation 6.2-6.4).
Scenarios run on a fake 15 fps clock, so a minute of driving takes milliseconds."""
import json

import numpy as np
import pytest

from app.contracts import Telemetry, level_from_risk
from app.vision.calibration import Profile, default_profile
from app.vision.features import FrameFeatures
from app.vision.monitor import DriverMonitor
from app.vision.temporal_model import SEQ_FEATURES, OnnxTCN, TemporalModel, heuristic
from app.vision.worker import SIM_MODES, SimSource

from .conftest import vision_cfg

FPS = 15
BASE = dict(ear=0.30, mar=0.05, pitch=0.0, yaw=0.0, roll=0.0, gh_r=0.0, gv_r=0.0, gh_l=0.0, gv_l=0.0,
            theta=0.0, lam=0.0, delta=0.6)


def calibrated(**over) -> Profile:
    return Profile(base=dict(BASE, **over), frames=150, calibrated=True)


def neutral(**kw) -> FrameFeatures:
    d = dict(face=True, ear=0.30, mar=0.05, pitch=0.0, yaw=0.0, roll=0.0, gaze_r=(0.0, 0.0), gaze_l=(0.0, 0.0),
             posture=(0.0, 0.0, 0.6))
    d.update(kw)
    return FrameFeatures(**d)


def blinking(t, period=3.0, length=0.15, **kw):
    return neutral(ear=0.08, **kw) if t % period < length else neutral(**kw)


class Drive:
    def __init__(self, profile=None, model=None, **cfg):
        self.m = DriverMonitor(vision_cfg(**cfg), profile or calibrated(), model or TemporalModel())
        self.t, self.events = 1000.0, []

    def run(self, seconds: float, fn) -> list[dict]:
        out, t0 = [], self.t
        for _ in range(int(round(seconds * FPS))):
            tel, ev = self.m.step(fn(self.t - t0), self.t, self.t)
            out.append(tel)
            if ev:
                self.events.append(ev)
            self.t += 1 / FPS
        return out


class Constant:
    """Stub temporal model returning a fixed probability."""
    seq_len, name = 45, "stub"

    def __init__(self, p):
        self.p = p

    def __call__(self, seq, stats):
        return self.p


# --------------------------------------------------------------------------- eyes
def test_alert_driver_stays_alert():
    tel = Drive().run(70, blinking)
    assert {t["drowsiness_level"] for t in tel} == {"ALERT"} and max(t["risk"] for t in tel) < 0.5
    last = tel[-1]
    assert last["perclos"] < 0.08 and last["long_closures"] == 0 and 15 <= last["blinks"] <= 25


def test_microsleep_is_high_risk_immediately_at_1_5_s():
    d = Drive()
    d.run(5, blinking)
    tel = d.run(2.2, lambda t: neutral(ear=0.08))
    before = [t for t in tel if t["eyes_closed_ms"] < 1500]
    after = [t for t in tel if t["eyes_closed_ms"] >= 1500]
    assert before and after
    assert all(t["drowsiness_level"] != "HIGH" for t in before)
    assert all(t["drowsiness_level"] == "HIGH" and t["risk"] == 1.0 for t in after)   # r_t = 1 (6.17)
    assert after[-1]["long_closures"] == 1


def test_one_noisy_open_frame_does_not_end_a_closure():
    d = Drive()
    d.run(2, lambda t: neutral())
    tel = d.run(2.0, lambda t: neutral(ear=0.25 if abs(t - 0.8) < 0.03 else 0.08))
    assert tel[-1]["eyes_closed_ms"] >= 1900


def test_thresholds_are_personalised():
    narrow = lambda t: neutral(ear=0.19)          # this driver's eyes are simply narrow
    default = Drive(profile=default_profile()).run(5, narrow)
    personal = Drive(profile=calibrated(ear=0.20)).run(5, narrow)
    assert default[-1]["eyes_closed_ms"] > 4000       # tau 0.21: "eyes closed" all along
    assert personal[-1]["eyes_closed_ms"] == 0        # tau 0.14: open
    assert personal[-1]["calibration"]["ear_threshold"] == pytest.approx(0.14)


def test_perclos_uses_the_last_60_seconds():
    d = Drive()
    tel = d.run(20, lambda t: neutral(ear=0.08) if t % 1.0 < 0.4 else neutral())
    assert tel[-1]["perclos"] == pytest.approx(0.4, abs=0.04)
    tel = d.run(61, lambda t: neutral())
    assert tel[-1]["perclos"] == 0.0 and tel[-1]["blinks"] == 0


# --------------------------------------------------------------------------- mouth
def test_yawn_needs_the_full_duration():
    d = Drive()
    d.run(2, lambda t: neutral())
    tel = d.run(4.0, lambda t: neutral(mar=0.80))
    first = next(i for i, t in enumerate(tel) if t["yawning"])
    assert abs(first / FPS - 3.0) < 0.15 and tel[-1]["yawns"] == 1


def test_animated_speech_is_not_a_yawn():
    # mouth wide open on vowels with a single-frame closure between syllables, for 8 s
    tel = Drive().run(8, lambda t: neutral(mar=0.7) if t % 0.4 < 0.27 else neutral(mar=0.1))
    assert not any(t["yawning"] for t in tel) and tel[-1]["yawns"] == 0


# --------------------------------------------------------------------------- head
def test_head_drop_after_1_5_s():
    d = Drive()
    d.run(2, lambda t: neutral())
    tel = d.run(2.0, lambda t: neutral(pitch=-25))
    first = next(i for i, t in enumerate(tel) if t["head_drop"])
    assert abs(first / FPS - 1.5) < 0.15
    assert tel[-1]["drowsiness_level"] == "HIGH" and tel[-1]["head_drops"] == 1


def test_small_nod_is_not_a_head_drop():
    assert not any(t["head_drop"] for t in Drive().run(4, lambda t: neutral(pitch=-15)))


def test_extreme_head_drop_still_alarms():
    """Regression: a head dropped beyond the eye-tracking range must not count as bad tracking
    (which suppresses alarms)."""
    d = Drive()
    d.run(2, lambda t: neutral())
    tel = d.run(2.0, lambda t: neutral(pitch=-60, ear=0.05))
    assert tel[-1]["quality"] >= 0.5 and tel[-1]["head_drop"] and tel[-1]["drowsiness_level"] == "HIGH"


def test_head_pose_is_relative_to_the_calibrated_neutral():
    tel = Drive(profile=calibrated(pitch=12.0, yaw=-4.0)).run(1, lambda t: neutral(pitch=12.0, yaw=-4.0))
    assert tel[-1]["head_pose"] == {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}


# --------------------------------------------------------------------------- gaze / distraction
def test_looking_away_becomes_distraction_after_2_s():
    d = Drive()
    tel = d.run(3.0, lambda t: neutral(yaw=30))
    first = next(i for i, t in enumerate(tel) if t["gaze"]["distracted"])
    assert abs(first / FPS - 2.0) < 0.15 and not tel[0]["gaze"]["on_road"]


def test_mirror_glance_is_not_distraction():
    d = Drive()
    tel = d.run(1.0, lambda t: neutral(yaw=35)) + d.run(3, lambda t: neutral())
    assert not any(t["gaze"]["distracted"] for t in tel) and tel[-1]["gaze"]["on_road"]


def test_eyes_only_glance_counts_via_gaze():
    tel = Drive().run(3.0, lambda t: neutral(gaze_r=(0.25, 0.0), gaze_l=(-0.25, 0.0)))
    assert tel[-1]["gaze"]["h"] == pytest.approx(0.25) and tel[-1]["gaze"]["distracted"]


def test_gaze_is_relative_to_each_drivers_baseline():
    prof = calibrated(gh_r=0.1, gh_l=-0.1)
    tel = Drive(profile=prof).run(3.0, lambda t: neutral(gaze_r=(0.1, 0.0), gaze_l=(-0.1, 0.0)))
    assert tel[-1]["gaze"]["h"] == pytest.approx(0.0) and not tel[-1]["gaze"]["distracted"]


def test_far_turn_counts_as_distraction_but_not_as_closed_eyes():
    tel = Drive().run(3.0, lambda t: neutral(yaw=70, ear=0.05))
    assert tel[-1]["gaze"]["distracted"] and tel[-1]["eyes_closed_ms"] == 0
    assert tel[-1]["quality"] < 0.5 and tel[-1]["drowsiness_level"] != "HIGH"


# --------------------------------------------------------------------------- posture
def test_slumping_becomes_posture_instability_after_5_s():
    tel = Drive().run(6.0, lambda t: neutral(posture=(0.0, 0.0, 0.36)))
    assert tel[0]["posture"]["deviation"] == pytest.approx(1.6, abs=0.01)
    first = next(i for i, t in enumerate(tel) if t["posture"]["unstable"])
    assert abs(first / FPS - 5.0) < 0.6
    assert tel[-1]["posture"]["slump_score"] == pytest.approx(0.4)


def test_brief_shift_is_not_instability():
    d = Drive()
    tel = d.run(2.0, lambda t: neutral(posture=(15.0, 0.3, 0.6))) + d.run(2, lambda t: neutral())
    assert not any(t["posture"]["unstable"] for t in tel)


def test_posture_without_shoulders():
    tel = Drive().run(6.0, lambda t: neutral(posture=None))
    assert not tel[-1]["posture"]["visible"] and not tel[-1]["posture"]["unstable"]


def test_uncalibrated_posture_uses_tilt_and_lean_only():
    tel = Drive(profile=default_profile()).run(1.0, lambda t: neutral(posture=(0.0, 0.0, 0.2)))
    assert tel[-1]["posture"]["deviation"] == 0.0 and tel[-1]["posture"]["slump_score"] == 0.0


# --------------------------------------------------------------------------- quality and fusion
def test_lost_face_lowers_quality_and_holds_the_risk():
    d = Drive(model=Constant(0.6))
    tel = d.run(10, lambda t: neutral())
    held = tel[-1]["fatigue_score"]
    tel = d.run(10, lambda t: FrameFeatures())
    assert tel[-1]["quality"] == 0.0 and not tel[-1]["face_detected"]
    assert tel[-1]["fatigue_score"] == held           # no updates from frames we cannot see


def test_ema_matches_6_16():
    tel = Drive(model=Constant(1.0)).run(50 / FPS, lambda t: neutral())
    assert tel[-1]["fatigue_score"] == pytest.approx(1 - 0.7 ** 10, abs=1e-3)   # 10 inferences, alpha 0.3


@pytest.mark.parametrize("p,level", [(0.3, "ALERT"), (0.6, "MILD"), (0.9, "HIGH")])
def test_levels_follow_table_6_2(p, level):
    assert Drive(model=Constant(p)).run(20, lambda t: neutral())[-1]["drowsiness_level"] == level


@pytest.mark.parametrize("r,level", [(0.0, "ALERT"), (0.4999, "ALERT"), (0.5, "MILD"), (0.8499, "MILD"),
                                     (0.85, "HIGH"), (1.0, "HIGH")])
def test_level_boundaries(r, level):
    assert level_from_risk(r) == level


def test_heuristic_evidence():
    assert heuristic(0.04, 0, 0, 0) == 0.0                     # normal blinking
    assert 0.5 <= heuristic(0.20, 0, 0, 0) < 0.85              # clearly drowsy eyes: MILD
    assert heuristic(0.30, 0, 0, 0) >= 0.85                    # severe PERCLOS: HIGH
    assert heuristic(0.0, 2, 0, 0) == pytest.approx(0.85)      # two microsleeps in a minute: HIGH
    assert heuristic(0.0, 0, 1, 0) < 0.5 and 0.5 <= heuristic(0.12, 0, 3, 0) < 0.85


# --------------------------------------------------------------------------- calibration in the loop
def test_calibration_runs_in_the_loop():
    d = Drive(profile=default_profile())
    d.m.start_calibration(d.t)
    tel = d.run(11, lambda t: neutral(ear=0.08 if t % 3 < 0.15 else 0.26, pitch=10.0))
    assert tel[0]["calibration"]["state"] == "calibrating" and tel[0]["calibration"]["progress"] < 0.05
    assert tel[-1]["calibration"]["state"] == "calibrated"
    assert len(d.events) == 1 and d.events[0]["ok"] and d.events[0]["frames"] == 150
    base = d.events[0]["profile"]["base"]
    assert base["ear"] == pytest.approx(0.26, abs=0.002) and base["pitch"] == pytest.approx(10.0)
    assert tel[-1]["calibration"]["ear_threshold"] == pytest.approx(0.182, abs=0.002)
    assert tel[-1]["head_pose"]["pitch"] == 0.0                # now relative to the learnt neutral


def test_failed_calibration_keeps_the_old_profile():
    d = Drive(profile=calibrated(ear=0.27))
    d.m.start_calibration(d.t)
    tel = d.run(61, lambda t: FrameFeatures())
    assert len(d.events) == 1 and not d.events[0]["ok"]
    assert tel[-1]["calibration"]["state"] == "calibrated" and d.m.profile.base["ear"] == 0.27


# --------------------------------------------------------------------------- simulator scenarios
@pytest.mark.parametrize("mode,check", [
    ("normal", lambda tel: {t["drowsiness_level"] for t in tel} == {"ALERT"}),
    ("drowsy", lambda tel: tel[-1]["long_closures"] >= 1 and any(t["drowsiness_level"] == "HIGH" for t in tel)),
    ("yawning", lambda tel: tel[-1]["yawns"] >= 2),
    ("nodding", lambda tel: tel[-1]["head_drops"] >= 3 and any(t["drowsiness_level"] == "HIGH" for t in tel)),
    ("distracted", lambda tel: any(t["gaze"]["distracted"] for t in tel)),
    ("slouching", lambda tel: any(t["posture"]["unstable"] for t in tel)),
    ("no_face", lambda tel: not tel[-1]["face_detected"] and tel[-1]["quality"] == 0.0),
])
def test_simulator_scenarios_from_a_fresh_start(mode, check):
    d = Drive(profile=default_profile())
    sim = SimSource(seed=1, clock=lambda: d.t)        # the simulator shares the fake clock
    d.m.start_calibration(d.t)
    frames = lambda t: sim.read()[0]
    d.run(11, frames)                                  # calibrate on the neutral driver
    assert d.events and d.events[0]["ok"]
    sim.set_mode(mode)
    assert check(d.run(40, frames)), mode


def test_simulator_knows_every_dashboard_scenario():
    assert set(SIM_MODES) == {"normal", "drowsy", "yawning", "nodding", "distracted", "slouching", "no_face"}


# --------------------------------------------------------------------------- telemetry contract
def test_telemetry_is_valid_json_and_schema():
    tel = Drive().run(1, lambda t: neutral(gaze_r=None, gaze_l=None))[-1]
    Telemetry.model_validate(tel)
    assert json.loads(json.dumps(tel)) == tel and tel["gaze"]["h"] is None


# --------------------------------------------------------------------------- trained TCN via ONNX Runtime
def _tcn(path, n_features=7):
    """Tiny stand-in TCN: y = sigmoid(mean(x)) over time and features."""
    import onnx
    from onnx import TensorProto, helper
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 45, n_features])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1])
    axes = helper.make_tensor("axes", TensorProto.INT64, [2], [1, 2])
    shape = helper.make_tensor("shape", TensorProto.INT64, [2], [1, 1])
    graph = helper.make_graph([helper.make_node("ReduceMean", ["x", "axes"], ["m"], keepdims=1),
                               helper.make_node("Reshape", ["m", "shape"], ["r"]),
                               helper.make_node("Sigmoid", ["r"], ["y"])],
                              "tcn", [x], [y], initializer=[axes, shape])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    onnx.save(model, str(path))
    return path


def test_onnx_tcn_contract(tmp_path):
    pytest.importorskip("onnx")
    path = _tcn(tmp_path / "tcn.onnx")
    tcn = OnnxTCN(str(path))
    assert tcn.seq_len == 45
    assert tcn(np.zeros((45, 7))) == pytest.approx(0.5)
    path.with_suffix(".json").write_text(json.dumps({"mean": [1.0] * 7, "std": [2.0] * 7}))
    assert OnnxTCN(str(path))(np.zeros((45, 7))) == pytest.approx(1 / (1 + np.exp(0.5)), abs=1e-6)
    path.with_suffix(".json").write_text(json.dumps({"mean": [1.0] * 7, "std": [2.0] * 7, "output": "logit"}))
    p = 1 / (1 + np.exp(0.5))
    assert OnnxTCN(str(path))(np.zeros((45, 7))) == pytest.approx(1 / (1 + np.exp(-p)), abs=1e-6)


def test_onnx_tcn_rejects_wrong_feature_count_and_falls_back(tmp_path):
    pytest.importorskip("onnx")
    path = _tcn(tmp_path / "bad.onnx", n_features=5)
    with pytest.raises(ValueError):
        OnnxTCN(str(path))
    assert TemporalModel(str(path)).name == "heuristic"
    assert TemporalModel(str(tmp_path / "missing.onnx")).name == "heuristic"
    assert len(SEQ_FEATURES) == 7


def test_monitor_uses_the_tcn_once_45_frames_are_in(tmp_path):
    pytest.importorskip("onnx")
    tm = TemporalModel(str(_tcn(tmp_path / "tcn.onnx")))
    assert tm.name == "tcn"
    tel = Drive(model=tm).run(20, lambda t: neutral())
    # neutral frames -> features (1.0 ear_norm, 0...) -> mean 1/7 -> sigmoid ~0.536 -> MILD
    assert tel[-1]["fatigue_score"] == pytest.approx(1 / (1 + np.exp(-1 / 7)), abs=0.01)
    assert tel[-1]["drowsiness_level"] == "MILD"


@pytest.mark.parametrize("p", [0.49996, 0.5, 0.84996, 0.8499, 0.85, 0.123])
def test_published_risk_and_level_always_agree(p):
    """Regression: R = 0.84996 was published as 0.85 (HIGH on the gauge) with level MILD."""
    for t in Drive(model=Constant(p)).run(30, lambda t: neutral()):
        assert t["drowsiness_level"] == level_from_risk(t["risk"]), t["risk"]
