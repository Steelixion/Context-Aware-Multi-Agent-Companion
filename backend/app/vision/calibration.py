"""Driver calibration (Implementation ch. 6.2).

Collects neutral frames at the start of a session and computes, per feature, a robust baseline
x_base = median(x_1..x_n)                         (6.1)
sigma  = 1.4826 * median(|x_i - x_base|)          (6.2)
Medians ignore blinks and landmark jitter that would skew a mean. Thresholds are then
personalised: tau_EAR = kappa * EAR_base (6.3), tau_MAR = MAR_base + delta_m (6.4).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .features import FrameFeatures

MAD_K = 1.4826
FEATURES = ("ear", "mar", "pitch", "yaw", "roll", "gh_r", "gv_r", "gh_l", "gv_l", "theta", "lam", "delta")
POSTURE = ("theta", "lam", "delta")
EAR_BASE_RANGE = (0.15, 0.45)   # outside this the eye landmarks are unreliable (e.g. sunglasses)
NEUTRAL_YAW, NEUTRAL_PITCH = 35.0, 40.0


@dataclass
class Profile:
    """Baselines (median) and spreads (scaled MAD) of a driver's neutral state."""
    base: dict = field(default_factory=dict)
    sigma: dict = field(default_factory=dict)
    frames: int = 0
    created_at: float = 0.0
    calibrated: bool = False

    def tau_ear(self, kappa: float) -> float:
        lo, hi = EAR_BASE_RANGE
        return kappa * min(max(self.base["ear"], lo), hi)

    def tau_mar(self, delta: float) -> float:
        return self.base["mar"] + delta

    def to_dict(self) -> dict:
        return dict(base=self.base, sigma=self.sigma, frames=self.frames,
                    created_at=self.created_at, calibrated=self.calibrated)

    @classmethod
    def from_dict(cls, d: dict | None) -> "Profile":
        """Stored profile -> Profile; anything missing or malformed falls back to the defaults."""
        p = default_profile()
        if not isinstance(d, dict):
            return p
        for key in ("base", "sigma"):
            for k, v in (d.get(key) or {}).items():
                if k in FEATURES and isinstance(v, (int, float)) and np.isfinite(v):
                    getattr(p, key)[k] = float(v)
        p.frames = int(d.get("frames") or 0)
        p.created_at = float(d.get("created_at") or 0.0)
        p.calibrated = bool(d.get("calibrated")) and p.frames > 0
        return p


def default_profile() -> Profile:
    """Population defaults until the driver is calibrated: open-eye EAR ~0.30 (tau 0.21),
    closed-mouth MAR ~0.05, neutral head/gaze at 0. No posture baseline: the slump term of
    D_t needs a calibrated head height."""
    return Profile(base=dict(ear=0.30, mar=0.05, pitch=0.0, yaw=0.0, roll=0.0,
                             gh_r=0.0, gv_r=0.0, gh_l=0.0, gv_l=0.0, theta=0.0, lam=0.0))


def robust_stats(x) -> tuple[float, float]:
    """(median, 1.4826 * MAD)."""
    a = np.asarray(x, dtype=np.float64)
    med = float(np.median(a))
    return med, float(MAD_K * np.median(np.abs(a - med)))


class Calibrator:
    """Accumulates neutral frames; result() gives the new profile once enough were seen."""

    def __init__(self, seconds: float, fps: float, timeout: float, started: float | None = None):
        self.target = max(30, int(round(seconds * fps)))
        self.timeout = timeout
        self.started = time.monotonic() if started is None else started
        self.samples: dict[str, list] = {k: [] for k in FEATURES}
        self.n = 0

    @property
    def progress(self) -> float:
        return min(1.0, self.n / self.target)

    def add(self, f: FrameFeatures) -> bool:
        """Use the frame if it is a usable neutral frame. Returns True if it was used."""
        if not f.face or f.ear is None or f.mar is None or f.pitch is None:
            return False
        if abs(f.yaw) > NEUTRAL_YAW or abs(f.pitch) > NEUTRAL_PITCH:
            return False
        self.n += 1
        s = self.samples
        for k, v in (("ear", f.ear), ("mar", f.mar), ("pitch", f.pitch), ("yaw", f.yaw), ("roll", f.roll)):
            s[k].append(v)
        if f.gaze_r is not None:
            s["gh_r"].append(f.gaze_r[0]); s["gv_r"].append(f.gaze_r[1])
        if f.gaze_l is not None:
            s["gh_l"].append(f.gaze_l[0]); s["gv_l"].append(f.gaze_l[1])
        if f.posture is not None:
            for k, v in zip(POSTURE, f.posture, strict=True):
                s[k].append(v)
        return True

    def finished(self, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        return self.n >= self.target or now - self.started >= self.timeout

    def result(self, prior: Profile) -> Profile | None:
        """New profile, or None if too few usable frames were seen (the prior stays in use).
        A feature with too few samples (e.g. shoulders never visible) keeps the prior's value."""
        if self.n < max(15, int(0.3 * self.target)):
            return None
        p = Profile(base=dict(prior.base), sigma=dict(prior.sigma), frames=self.n,
                    created_at=time.time(), calibrated=True)
        need = max(10, int(0.3 * self.n))
        for k, xs in self.samples.items():
            if len(xs) >= need:
                p.base[k], p.sigma[k] = robust_stats(xs)
        return p
