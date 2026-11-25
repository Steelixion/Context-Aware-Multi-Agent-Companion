"""Per-frame feature maths (Implementation 6.3) on exact synthetic geometry."""
import math

import numpy as np
import pytest

from app.vision.features import (EYE_L, EYE_R, GAZE_L, GAZE_R, MOUTH, NOSE, PNP_IDX, PNP_MODEL, SHOULDER_L,
                                 SHOULDER_R, HeadPose, euler_from_rotation, extract, eye_aspect_ratio, eye_gaze,
                                 mouth_aspect_ratio, posture)


def Rx(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def mesh() -> np.ndarray:
    return np.zeros((478, 3))


def eye(pts, idx, cx, cy, width, opening):
    """Place the 6 EAR points of an eye: corners at +-width/2, lids at +-opening/2."""
    p1, p2, p3, p4, p5, p6 = idx
    pts[p1, :2] = (cx - width / 2, cy)
    pts[p4, :2] = (cx + width / 2, cy)
    pts[p2, :2] = (cx - width / 6, cy - opening / 2)
    pts[p3, :2] = (cx + width / 6, cy - opening / 2)
    pts[p6, :2] = (cx - width / 6, cy + opening / 2)
    pts[p5, :2] = (cx + width / 6, cy + opening / 2)


# --------------------------------------------------------------------------- EAR / MAR
@pytest.mark.parametrize("opening", [0.0, 3.0, 9.0, 12.0])
def test_ear_formula(opening):
    pts = mesh()
    eye(pts, EYE_R, 100, 100, 30, opening)
    assert eye_aspect_ratio(pts, EYE_R) == pytest.approx((opening + opening) / (2 * 30), abs=1e-6)


def test_ear_is_scale_and_translation_invariant():
    a, b = mesh(), mesh()
    eye(a, EYE_L, 100, 100, 30, 9)
    eye(b, EYE_L, 640, 20, 90, 27)
    assert eye_aspect_ratio(a, EYE_L) == pytest.approx(eye_aspect_ratio(b, EYE_L))


def test_mar_formula_uses_three_vertical_pairs_over_three_widths():
    pts = mesh()
    q1, q2, q3, q4, q5, q6, q7, q8 = MOUTH
    pts[q1, :2], pts[q2, :2] = (100, 200), (160, 200)          # inner corners, width 60
    for (up, lo), gap in zip(((q3, q4), (q5, q6), (q7, q8)), (10, 20, 12), strict=True):
        pts[up, :2], pts[lo, :2] = (130, 200 - gap / 2), (130, 200 + gap / 2)
    assert mouth_aspect_ratio(pts) == pytest.approx((10 + 20 + 12) / (3 * 60))   # (6.7)


# --------------------------------------------------------------------------- head pose
@pytest.mark.parametrize("name,r,expect", [
    ("frontal", np.eye(3), (0, 0, 0)),
    ("head down 25", Rx(math.radians(25)), (-25, 0, 0)),
    ("head up 15", Rx(math.radians(-15)), (15, 0, 0)),
    ("turned to image right 30", Ry(math.radians(-30)), (0, 30, 0)),
    ("turned to image left 30", Ry(math.radians(30)), (0, -30, 0)),
    ("tilted clockwise 20", Rz(math.radians(20)), (0, 0, 20)),
])
def test_euler_conventions(name, r, expect):
    assert euler_from_rotation(r) == pytest.approx(expect, abs=1e-6)


def test_euler_decomposition_recovers_combined_rotation():   # (6.10) is the ZYX decomposition
    r = Rz(math.radians(10)) @ Ry(math.radians(-20)) @ Rx(math.radians(15))
    assert euler_from_rotation(r) == pytest.approx((-15, 20, 10), abs=1e-6)


def _project(r, w=640, h=480, z=60.0, shift=(0, 0)):
    cam = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1.0]])
    p = (r @ PNP_MODEL.T).T + [shift[0], shift[1], z]
    uv = (cam @ p.T).T
    pts = mesh()
    for k, i in enumerate(PNP_IDX):
        pts[i, :2] = uv[k, :2] / uv[k, 2]
    return pts


@pytest.mark.parametrize("seed", range(20))
def test_pnp_fallback_recovers_random_poses(seed):
    rng = np.random.default_rng(seed)
    pitch, yaw, roll = rng.uniform(-30, 20), rng.uniform(-40, 40), rng.uniform(-30, 30)
    r = Rz(math.radians(roll)) @ Ry(math.radians(-yaw)) @ Rx(math.radians(-pitch))
    got = HeadPose()(_project(r), 640, 480)
    assert got == pytest.approx((pitch, yaw, roll), abs=0.5)


def test_extract_prefers_the_landmarker_rotation_over_pnp():
    pts = _project(np.eye(3))     # (eye corners 33/263 are PnP points too: leave them alone)
    f = extract(pts, None, 640, 480, HeadPose(), rotation=Rx(math.radians(30)))
    assert f.pitch == pytest.approx(-30, abs=1e-6)
    f = extract(pts, None, 640, 480, HeadPose())
    assert f.pitch == pytest.approx(0, abs=0.5)
    f = extract(pts, None, 640, 480)                 # neither: angles unknown, not zero
    assert f.pitch is None and f.face


# --------------------------------------------------------------------------- gaze
def _gaze_eye(pts, idx, outer_x, inner_x, top_y, bottom_y, iris):
    co, ci, eu, el, c = idx
    pts[co, :2], pts[ci, :2] = (outer_x, (top_y + bottom_y) / 2), (inner_x, (top_y + bottom_y) / 2)
    pts[eu, :2], pts[el, :2] = ((outer_x + inner_x) / 2, top_y), ((outer_x + inner_x) / 2, bottom_y)
    pts[c, :2] = iris


def test_gaze_formula():
    pts = mesh()
    _gaze_eye(pts, GAZE_R, 100, 140, 90, 100, (120, 95))       # centred
    assert eye_gaze(pts, GAZE_R) == pytest.approx((0.0, 0.0))
    _gaze_eye(pts, GAZE_R, 100, 140, 90, 100, (130, 98))       # towards inner corner and lower lid
    assert eye_gaze(pts, GAZE_R) == pytest.approx((0.25, 0.3))


def test_gaze_sign_is_relative_to_each_eyes_inner_corner():
    pts = mesh()
    _gaze_eye(pts, GAZE_L, 240, 200, 90, 100, (210, 95))       # left eye: inner corner is on the image left
    gh, _ = eye_gaze(pts, GAZE_L)
    assert gh == pytest.approx(0.25)


@pytest.mark.parametrize("top,bottom,outer,inner", [(95, 95.2, 100, 140), (90, 100, 100, 101)])
def test_gaze_undefined_for_closed_or_edge_on_eye(top, bottom, outer, inner):
    pts = mesh()
    _gaze_eye(pts, GAZE_R, outer, inner, top, bottom, (120, 95))
    assert eye_gaze(pts, GAZE_R) is None


# --------------------------------------------------------------------------- posture
def _pose(nose, left_img, right_img, vis=1.0):
    """left_img / right_img: shoulders as they appear in the image (driver's right / left)."""
    p = np.zeros((33, 4))
    p[NOSE] = (*nose, 0, vis)
    p[SHOULDER_R] = (*left_img, 0, vis)
    p[SHOULDER_L] = (*right_img, 0, vis)
    return p


def test_posture_level_and_centred():
    theta, lam, delta = posture(_pose((320, 100), (220, 220), (420, 220)))
    assert theta == pytest.approx(0) and lam == pytest.approx(0) and delta == pytest.approx(120 / 200)


def test_posture_tilt_lean_and_height():
    theta, lam, delta = posture(_pose((370, 160), (220, 200), (420, 240)))
    assert theta == pytest.approx(math.degrees(math.atan2(40, 200)))   # image-right shoulder lower
    w = math.hypot(200, 40)
    assert lam == pytest.approx(50 / w) and delta == pytest.approx(60 / w)


def test_posture_is_scale_and_translation_invariant():
    a = posture(_pose((320, 100), (220, 220), (420, 230)))
    b = posture(_pose((320 * 3 + 50, 100 * 3 + 7), (220 * 3 + 50, 220 * 3 + 7), (420 * 3 + 50, 230 * 3 + 7)))
    assert a == pytest.approx(b)


def test_posture_needs_visible_landmarks():
    assert posture(_pose((320, 100), (220, 220), (420, 220), vis=0.3)) is None


def test_posture_independent_of_landmark_label_order():
    a = posture(_pose((320, 100), (220, 220), (420, 230)))
    p = _pose((320, 100), (220, 220), (420, 230))
    p[[SHOULDER_L, SHOULDER_R]] = p[[SHOULDER_R, SHOULDER_L]]   # e.g. a mirrored image
    assert posture(p) == pytest.approx(a)


def test_extract_without_face_keeps_posture():
    f = extract(None, _pose((320, 100), (220, 220), (420, 220)), 640, 480, HeadPose())
    assert not f.face and f.ear is None and f.posture is not None
