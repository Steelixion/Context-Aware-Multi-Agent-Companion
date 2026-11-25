"""Driver calibration (Implementation 6.2): robust baselines and personalised thresholds."""
import math

import numpy as np
import pytest

from app.vision.calibration import MAD_K, Calibrator, Profile, default_profile, robust_stats
from app.vision.features import FrameFeatures


def frame(ear=0.30, mar=0.05, pitch=8.0, yaw=-3.0, roll=1.0, gaze=(0.02, 0.05), posture=(1.5, 0.02, 0.58), face=True):
    return FrameFeatures(face=face, ear=ear, mar=mar, pitch=pitch, yaw=yaw, roll=roll,
                         gaze_r=gaze, gaze_l=(-gaze[0], gaze[1]) if gaze else None, posture=posture)


def test_robust_stats_formula():           # (6.1), (6.2)
    x = [1.0, 2.0, 3.0, 4.0, 100.0]
    med, sigma = robust_stats(x)
    assert med == 3.0 and sigma == pytest.approx(MAD_K * 1.0)


def test_scaled_mad_matches_gaussian_sigma():
    x = np.random.default_rng(0).normal(5.0, 2.0, 20000)
    med, sigma = robust_stats(x)
    assert med == pytest.approx(5.0, abs=0.05) and sigma == pytest.approx(2.0, rel=0.03)


def test_baseline_ignores_blinks_where_a_mean_would_not():
    rng = np.random.default_rng(1)
    ear = np.where(rng.random(300) < 0.12, 0.08, rng.normal(0.31, 0.008, 300))   # 12 % blink frames
    med, _ = robust_stats(ear)
    assert med == pytest.approx(0.31, abs=0.005)
    assert np.mean(ear) < 0.29          # the mean is dragged down; the median is not


def test_personalised_thresholds():        # (6.3), (6.4)
    p = Profile(base=dict(default_profile().base, ear=0.32, mar=0.06))
    assert p.tau_ear(0.70) == pytest.approx(0.224) and p.tau_mar(0.40) == pytest.approx(0.46)


@pytest.mark.parametrize("ear,expect", [(0.05, 0.7 * 0.15), (0.9, 0.7 * 0.45)])
def test_implausible_ear_baseline_is_clamped(ear, expect):
    assert Profile(base=dict(default_profile().base, ear=ear)).tau_ear(0.7) == pytest.approx(expect)


def test_calibrator_learns_the_driver():
    cal = Calibrator(seconds=10, fps=15, timeout=60, started=0.0)
    rng = np.random.default_rng(2)
    for i in range(150):
        cal.add(frame(ear=0.08 if i % 25 == 0 else 0.26 + rng.normal(0, 0.005), pitch=12 + rng.normal(0, 1)))
    assert cal.finished(10.0) and cal.progress == 1.0
    p = cal.result(default_profile())
    assert p.calibrated and p.frames == 150
    assert p.base["ear"] == pytest.approx(0.26, abs=0.005) and p.base["pitch"] == pytest.approx(12, abs=0.5)
    assert p.base["delta"] == pytest.approx(0.58) and p.base["gh_l"] == pytest.approx(-0.02)
    assert p.sigma["pitch"] == pytest.approx(1.0, rel=0.25)


def test_calibrator_skips_unusable_frames():
    cal = Calibrator(seconds=2, fps=15, timeout=60, started=0.0)
    assert not cal.add(frame(face=False))
    assert not cal.add(frame(yaw=50))        # looking at the side mirror is not "neutral"
    assert not cal.add(frame(pitch=-60))
    assert not cal.add(FrameFeatures(face=True, ear=0.3, mar=0.05))   # head pose unknown
    assert cal.add(frame()) and cal.n == 1


def test_missing_posture_keeps_prior_value():
    prior = default_profile()
    cal = Calibrator(seconds=2, fps=15, timeout=60, started=0.0)
    for _ in range(30):
        cal.add(frame(posture=None, gaze=None))
    p = cal.result(prior)
    assert "delta" not in p.base and p.base["theta"] == prior.base["theta"] and p.base["gh_r"] == 0.0


def test_too_few_frames_fails_and_timeout_finishes():
    cal = Calibrator(seconds=10, fps=15, timeout=60, started=0.0)
    for _ in range(10):
        cal.add(frame())
    assert not cal.finished(30.0) and cal.finished(60.0)
    assert cal.result(default_profile()) is None


def test_profile_roundtrip_and_validation():
    p = Profile(base=dict(default_profile().base, ear=0.27, delta=0.6), sigma=dict(ear=0.01), frames=150,
                created_at=1.0, calibrated=True)
    q = Profile.from_dict(p.to_dict())
    assert q.to_dict() == p.to_dict()
    bad = {"base": {"ear": float("nan"), "mar": "x", "bogus": 1.0, "pitch": 5}, "frames": 0, "calibrated": True}
    r = Profile.from_dict(bad)
    assert r.base["ear"] == 0.30 and r.base["mar"] == 0.05 and r.base["pitch"] == 5.0
    assert "bogus" not in r.base and not r.calibrated          # zero frames cannot be "calibrated"
    for junk in (None, [], "x", 3):
        assert Profile.from_dict(junk).to_dict() == default_profile().to_dict()


def test_default_profile_matches_population_values():
    p = default_profile()
    assert p.tau_ear(0.7) == pytest.approx(0.21) and p.tau_mar(0.4) == pytest.approx(0.45)
    assert not p.calibrated and "delta" not in p.base and all(math.isfinite(v) for v in p.base.values())
