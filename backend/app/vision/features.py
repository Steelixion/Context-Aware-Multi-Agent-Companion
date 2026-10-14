"""Geometric feature extraction from MediaPipe landmarks (pure numpy/cv2, no mediapipe import)."""
from __future__ import annotations

import numpy as np

# MediaPipe FaceMesh indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [78, 81, 13, 311, 308, 402, 14, 178]
# nose, chin, image-left eye outer, image-right eye outer, image-left mouth, image-right mouth
POSE_IDX = [1, 152, 33, 263, 61, 291]
MODEL_3D = np.array([
    (0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0),
], dtype=np.float64)


def _d(a, b) -> float:
    return float(np.linalg.norm(a - b))


def eye_aspect_ratio(pts: np.ndarray, idx: list[int]) -> float:
    p = pts[idx]
    return (_d(p[1], p[5]) + _d(p[2], p[4])) / (2.0 * _d(p[0], p[3]) + 1e-6)


def mouth_aspect_ratio(pts: np.ndarray) -> float:
    p = pts[MOUTH]
    return (_d(p[1], p[7]) + _d(p[2], p[6]) + _d(p[3], p[5])) / (2.0 * _d(p[0], p[4]) + 1e-6)


def head_pose(pts: np.ndarray, w: int, h: int) -> tuple[float, float, float]:
    """Return (yaw, pitch, roll) in degrees via solvePnP."""
    import cv2
    img = pts[POSE_IDX].astype(np.float64)
    cam = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(MODEL_3D, img, cam, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0, 0.0, 0.0
    rmat, _ = cv2.Rodrigues(rvec)
    proj = np.hstack((rmat, tvec))
    euler = cv2.decomposeProjectionMatrix(proj)[6].flatten()
    pitch, yaw, roll = (float(x) for x in euler)
    # fold into [-90, 90]
    if pitch > 90: pitch -= 180
    if pitch < -90: pitch += 180
    if roll > 90: roll -= 180
    if roll < -90: roll += 180
    return yaw, pitch, roll


def posture_from_pose(landmarks) -> tuple[float, float] | None:
    """(slump_score 0..1, shoulder_tilt deg) from MediaPipe Pose landmarks, or None if not visible."""
    nose, ls, rs = landmarks[0], landmarks[11], landmarks[12]
    if min(ls.visibility, rs.visibility) < 0.5:
        return None
    width = abs(ls.x - rs.x) + 1e-6
    mid_y = (ls.y + rs.y) / 2
    neck_ratio = (mid_y - nose.y) / width      # large when sitting tall, small when slumped
    slump = float(np.clip(1.0 - neck_ratio / 0.9, 0.0, 1.0))
    tilt = float(np.degrees(np.arctan2(rs.y - ls.y, rs.x - ls.x)))
    tilt = tilt - 180 if tilt > 90 else tilt + 180 if tilt < -90 else tilt
    return slump, tilt
