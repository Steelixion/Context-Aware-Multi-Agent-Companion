"""Per-frame geometric features from MediaPipe landmarks (Implementation ch. 6.3).

Pure numpy (+ OpenCV for PnP); no MediaPipe import, so it is unit-testable with synthetic points.
All inputs are pixel coordinates of the unmirrored camera frame (x right, y down). Head
rotations are 3x3 matrices in the OpenCV camera frame (x right, y down, z away from the camera).

Conventions used in telemetry:
  pitch < 0  head down          yaw > 0  head turned towards the image's right (driver's left)
  roll  > 0  head tilted clockwise in the image
  gaze h > 0 eyes towards the image's right, v > 0 eyes looking down
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# ---- Face Landmarker indices (478-point mesh incl. iris). "R" = driver's right eye (image left).
EYE_R = (33, 160, 158, 133, 153, 144)    # p1 outer, p2, p3 upper lid, p4 inner, p5, p6 lower lid
EYE_L = (362, 385, 387, 263, 373, 380)   # p1 inner, p2, p3 upper lid, p4 outer, p5, p6 lower lid
MOUTH = (78, 308, 81, 178, 13, 14, 311, 402)  # q1, q2 inner corners; (q3,q4) (q5,q6) (q7,q8) vertical pairs
# gaze: outer corner, inner corner, upper lid, lower lid, iris centre
GAZE_R = (33, 133, 159, 145, 468)
GAZE_L = (263, 362, 386, 374, 473)
# PnP (6.9): nose tip, chin, outer eye corners, mouth corners
PNP_IDX = (1, 152, 33, 263, 61, 291)
# 3D reference points: MediaPipe's canonical face model (face_geometry/data/canonical_face_model.obj,
# cm), the same geometry the Face Landmarker is trained on, so a frontal face solves to ~0 deg.
# (A generic anthropometric model biases pitch by ~20 deg with these landmarks.) Converted to the
# camera frame of a face looking at the camera (x right, y down, z away) and centred on the nose tip.
PNP_MODEL = np.array([
    (0.0, 0.0, 0.0),                       # 1   nose tip
    (0.0, 8.276513, 3.211112),             # 152 chin
    (-4.445859, -3.790856, 4.302182),      # 33  outer corner of the eye on the image left
    (4.445859, -3.790856, 4.302182),       # 263 outer corner of the eye on the image right
    (-2.456206, 3.215756, 3.191720),       # 61  mouth corner on the image left
    (2.456206, 3.215756, 3.191720),        # 291 mouth corner on the image right
], dtype=np.float64)
# ---- Pose Landmarker indices
NOSE, SHOULDER_L, SHOULDER_R = 0, 11, 12  # L/R = driver's left/right (image right/left)
MIN_VISIBILITY = 0.5


def _d(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[:2] - b[:2]))


def eye_aspect_ratio(pts: np.ndarray, idx: tuple = EYE_R) -> float:
    """(6.5) EAR = (|p2-p6| + |p3-p5|) / (2 |p1-p4|)."""
    p1, p2, p3, p4, p5, p6 = (pts[i] for i in idx)
    return (_d(p2, p6) + _d(p3, p5)) / (2.0 * _d(p1, p4) + 1e-9)


def mouth_aspect_ratio(pts: np.ndarray) -> float:
    """(6.7) MAR = (|q3-q4| + |q5-q6| + |q7-q8|) / (3 |q1-q2|)."""
    q1, q2, q3, q4, q5, q6, q7, q8 = (pts[i] for i in MOUTH)
    return (_d(q3, q4) + _d(q5, q6) + _d(q7, q8)) / (3.0 * _d(q1, q2) + 1e-9)


def euler_from_rotation(r: np.ndarray) -> tuple[float, float, float]:
    """(6.10) pitch = atan2(R32, R33), yaw = atan2(-R31, sqrt(R32^2 + R33^2)), roll = atan2(R21, R11),
    in degrees, mapped to the telemetry conventions (pitch < 0 = head down)."""
    pitch = math.degrees(math.atan2(r[2, 1], r[2, 2]))
    yaw = math.degrees(math.atan2(-r[2, 0], math.hypot(r[2, 1], r[2, 2])))
    roll = math.degrees(math.atan2(r[1, 0], r[0, 0]))
    # camera frame: +pitch = chin away from the camera (head down); +yaw = face turned towards the
    # image left. Flip both so that down is negative and the image's right is positive.
    return -pitch, -yaw, roll


class HeadPose:
    """(6.9) PnP from 6 face landmarks with a pinhole camera (f = image width, no distortion).
    SQPnP is a global solver, so there are no flipped solutions to fold back.

    Fallback only: the live pipeline uses the Face Landmarker's own head rotation (its fit of the
    canonical face to all 478 landmarks). Measured against it, 6-point PnP was 13-20 deg off in
    pitch on frontal faces and moved ~20 deg when the face shifted in the frame, which would
    cause false head-drop alarms."""

    def __init__(self):
        import cv2
        self.cv2 = cv2

    def __call__(self, pts: np.ndarray, w: int, h: int) -> tuple[float, float, float] | None:
        cv2 = self.cv2
        img = np.ascontiguousarray([pts[i][:2] for i in PNP_IDX], dtype=np.float64)
        cam = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
        ok, rvec, _ = cv2.solvePnP(PNP_MODEL, img, cam, None, flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            return None
        return euler_from_rotation(cv2.Rodrigues(rvec)[0])


def eye_gaze(pts: np.ndarray, idx: tuple) -> tuple[float, float] | None:
    """(6.11) g_h = (c_x - co_x) / (ci_x - co_x) - 0.5,  g_v = (c_y - eu_y) / (el_y - eu_y) - 0.5.
    g_h > 0 = iris towards the inner corner, g_v > 0 = iris towards the lower lid."""
    co, ci, eu, el, c = (pts[i] for i in idx)
    width, height = ci[0] - co[0], el[1] - eu[1]
    eye_w = _d(co, ci)
    if eye_w < 1e-6 or height < 0.05 * eye_w:
        return None                      # (nearly) closed eye
    if abs(width) < 0.2 * eye_w or abs(width) < 0.5 * height:
        return None                      # eye seen edge-on or rolled far over (open eye: width ~3x height)
    return float((c[0] - co[0]) / width - 0.5), float((c[1] - eu[1]) / height - 0.5)


def posture(pose: np.ndarray) -> tuple[float, float, float] | None:
    """(6.12-6.13) shoulder tilt theta (deg), lateral lean lambda, head height delta, from pose
    landmarks in pixels (x, y, z, visibility). None unless nose and both shoulders are visible.

    theta is measured from the shoulder on the image left to the one on the image right, so it
    reads 0 deg when level (the PDF's s_R - s_L differs by a constant 180 deg)."""
    nose, sl, sr = pose[NOSE], pose[SHOULDER_L], pose[SHOULDER_R]
    if min(nose[3], sl[3], sr[3]) < MIN_VISIBILITY:
        return None
    left, right = (sr, sl) if sr[0] <= sl[0] else (sl, sr)   # image order
    w = _d(sl, sr)
    if w < 1e-6:
        return None
    theta = math.degrees(math.atan2(right[1] - left[1], right[0] - left[0]))
    mx, my = (sl[0] + sr[0]) / 2, (sl[1] + sr[1]) / 2
    return theta, float((nose[0] - mx) / w), float((my - nose[1]) / w)


@dataclass
class FrameFeatures:
    """Raw per-frame measurements. None = not measurable in this frame."""
    face: bool = False
    ear: float | None = None
    mar: float | None = None
    pitch: float | None = None
    yaw: float | None = None
    roll: float | None = None
    gaze_r: tuple | None = None      # (g_h, g_v) of the driver's right eye
    gaze_l: tuple | None = None
    posture: tuple | None = None     # (theta, lambda, delta)


def extract(face_pts: np.ndarray | None, pose_pts: np.ndarray | None, w: int, h: int,
            head_pose: HeadPose | None = None, rotation: np.ndarray | None = None) -> FrameFeatures:
    """Landmarks -> FrameFeatures. `rotation` (3x3, camera frame) is the Face Landmarker's head
    rotation; without it the angles come from PnP (`head_pose`)."""
    f = FrameFeatures(posture=posture(pose_pts) if pose_pts is not None else None)
    if face_pts is None:
        return f
    f.face = True
    f.ear = (eye_aspect_ratio(face_pts, EYE_R) + eye_aspect_ratio(face_pts, EYE_L)) / 2
    f.mar = mouth_aspect_ratio(face_pts)
    angles = euler_from_rotation(rotation) if rotation is not None else         head_pose(face_pts, w, h) if head_pose is not None else None
    if angles is not None:
        f.pitch, f.yaw, f.roll = angles
    if len(face_pts) > max(GAZE_L):   # iris points are only in the 478-point mesh
        f.gaze_r, f.gaze_l = eye_gaze(face_pts, GAZE_R), eye_gaze(face_pts, GAZE_L)
    return f
