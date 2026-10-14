from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Level = Literal["ALERT", "MILD", "MODERATE", "HIGH"]


def level_from_score(score: float) -> Level:
    if score >= 0.80:
        return "HIGH"
    if score >= 0.55:
        return "MODERATE"
    if score >= 0.30:
        return "MILD"
    return "ALERT"


class HeadPose(BaseModel):
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0


class Posture(BaseModel):
    slump_score: float = 0.0
    shoulder_tilt: float = 0.0


class Telemetry(BaseModel):
    """JSON payload emitted by the vision process for the orchestrator."""
    ts: float
    face_detected: bool
    ear: float = 0.0
    mar: float = 0.0
    head_pose: HeadPose = Field(default_factory=HeadPose)
    posture: Posture = Field(default_factory=Posture)
    perclos_30s: float = 0.0
    eyes_closed_ms: int = 0
    yawns_30s: int = 0
    fatigue_score: float = 0.0
    drowsiness_level: Level = "ALERT"
    source: str = "camera"
    note: str = ""
