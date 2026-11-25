"""Telemetry contract between the vision process and the orchestrator (one message per frame)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Level = Literal["ALERT", "MILD", "HIGH"]


def level_from_risk(risk: float, mild: float = 0.50, high: float = 0.85) -> Level:
    """Table 6.2: R < 0.50 alert, 0.50 <= R < 0.85 mild drowsiness, R >= 0.85 high drowsiness."""
    if risk >= high:
        return "HIGH"
    if risk >= mild:
        return "MILD"
    return "ALERT"


class HeadPose(BaseModel):
    """Degrees relative to the driver's calibrated neutral pose (pitch < 0 = head down)."""
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0


class Gaze(BaseModel):
    """Iris offsets relative to the calibrated baseline (h > 0 = towards the image's right)."""
    h: float | None = None
    v: float | None = None
    on_road: bool = True
    off_road_ms: int = 0
    distracted: bool = False


class Posture(BaseModel):
    visible: bool = False
    shoulder_tilt: float = 0.0     # degrees, 0 = level
    lean: float = 0.0              # lambda: nose offset from the shoulder midpoint / shoulder width
    head_height: float = 0.0       # delta: nose height above the shoulder line / shoulder width
    slump_score: float = 0.0       # max(0, 1 - delta / delta_base)
    deviation: float = 0.0         # D_t (6.14); > 1 = outside the driver's tolerance
    unstable: bool = False         # D_t > 1 for longer than posture_seconds


class Calibration(BaseModel):
    state: Literal["default", "calibrating", "calibrated"] = "default"
    progress: float = 0.0
    ear_threshold: float = 0.21
    mar_threshold: float = 0.45


class Telemetry(BaseModel):
    ts: float
    face_detected: bool
    quality: float = 1.0           # share of usable frames over the last ~2 s
    ear: float = 0.0
    ear_norm: float = 0.0          # EAR / EAR_base (1.0 = eyes as open as at calibration)
    mar: float = 0.0
    mar_rel: float = 0.0           # MAR - MAR_base
    head_pose: HeadPose = Field(default_factory=HeadPose)
    gaze: Gaze = Field(default_factory=Gaze)
    posture: Posture = Field(default_factory=Posture)
    window_s: float = 60.0
    perclos: float = 0.0
    eyes_closed_ms: int = 0
    head_drop_ms: int = 0
    blinks: int = 0                # counts within the last window_s seconds
    long_closures: int = 0
    yawns: int = 0
    head_drops: int = 0
    yawning: bool = False
    head_drop: bool = False
    fatigue_score: float = 0.0     # smoothed temporal-model probability p_hat
    risk: float = 0.0              # R = max(p_hat, r)
    drowsiness_level: Level = "ALERT"
    calibration: Calibration = Field(default_factory=Calibration)
    source: str = "camera"
    note: str = ""
