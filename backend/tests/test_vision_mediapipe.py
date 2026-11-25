"""Real MediaPipe Face/Pose Landmarker on MediaPipe's own sample photos.

Assets: python scripts/download_models.py --test-assets
"""
import base64

import cv2
import numpy as np
import pytest

from app.vision.features import GAZE_L, GAZE_R, HeadPose, extract
from app.vision.worker import CameraSource

from .conftest import ASSETS, CFG, needs_vision, vision_cfg

pytestmark = needs_vision


@pytest.fixture(scope="module")
def extractor():
    from app.vision.landmarks import LandmarkExtractor
    ex = LandmarkExtractor(CFG.face_model, CFG.pose_model, video=False)
    yield ex
    ex.close()


def img(name: str) -> np.ndarray:
    return cv2.imread(str(ASSETS / name))


def features(extractor, bgr):
    h, w = bgr.shape[:2]
    lm = extractor.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return lm, extract(lm.face, lm.pose, w, h, HeadPose(), lm.rotation)


def rotate(bgr, degrees_ccw):
    h, w = bgr.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), degrees_ccw, 1.0)
    return cv2.warpAffine(bgr, m, (w, h), borderValue=(255, 255, 255))


# --------------------------------------------------------------------------- face
@pytest.mark.parametrize("name", ["portrait.jpg", "business-person.png"])
def test_frontal_faces(extractor, name):
    lm, f = features(extractor, img(name))
    assert lm.face.shape == (478, 3) and lm.rotation.shape == (3, 3)
    assert np.allclose(lm.rotation @ lm.rotation.T, np.eye(3), atol=1e-6)
    assert 0.15 < f.ear < 0.45                         # open eyes
    assert abs(f.yaw) < 10 and abs(f.roll) < 10


def test_mouth_closed_vs_smiling(extractor):
    _, closed = features(extractor, img("business-person.png"))
    _, smile = features(extractor, img("portrait.jpg"))
    assert closed.mar < 0.05 < smile.mar < 0.45        # teeth showing, but nowhere near a yawn


def test_iris_points_sit_between_the_eye_corners(extractor):
    lm, _ = features(extractor, img("business-person.png"))
    for outer, inner, _, _, iris in (GAZE_R, GAZE_L):
        lo, hi = sorted((lm.face[outer][0], lm.face[inner][0]))
        assert lo < lm.face[iris][0] < hi


def test_looking_at_the_camera_is_gaze_near_centre(extractor):
    _, f = features(extractor, img("business-person.png"))
    assert all(abs(g) < 0.15 for g in (*f.gaze_r, *f.gaze_l))


# --------------------------------------------------------------------------- head pose
def test_frontal_portrait_pitch_is_near_zero(extractor):
    """Regression: 6-point PnP put this straight-to-camera portrait at -17..-23 deg pitch."""
    _, f = features(extractor, img("portrait.jpg"))
    assert abs(f.pitch) < 8


def test_mirroring_flips_yaw_roll_and_gaze(extractor):
    im = img("business-person.png")
    _, a = features(extractor, im)
    _, b = features(extractor, cv2.flip(im, 1))
    assert b.yaw == pytest.approx(-a.yaw, abs=1.5) and b.roll == pytest.approx(-a.roll, abs=1.5)
    assert b.pitch == pytest.approx(a.pitch, abs=1.5)
    assert np.sign(b.yaw) == -np.sign(a.yaw)


@pytest.mark.parametrize("ccw", [-30, -15, 15, 30])
def test_in_plane_rotation_shows_up_as_roll_only(extractor, ccw):
    im = img("business-person.png")
    _, a = features(extractor, im)
    _, b = features(extractor, rotate(im, ccw))
    assert b.roll - a.roll == pytest.approx(-ccw, abs=3)      # counter-clockwise image = negative roll
    assert b.pitch == pytest.approx(a.pitch, abs=2) and b.yaw == pytest.approx(a.yaw, abs=2)


def test_pose_does_not_change_with_scale_or_position_in_frame(extractor):
    im = img("business-person.png")
    _, a = features(extractor, im)
    _, small = features(extractor, cv2.resize(im, None, fx=0.4, fy=0.4))
    _, high = features(extractor, cv2.copyMakeBorder(im, 0, 900, 0, 0, cv2.BORDER_CONSTANT, value=(255,) * 3))
    for other in (small, high):
        assert (other.pitch, other.yaw, other.roll) == pytest.approx((a.pitch, a.yaw, a.roll), abs=3)
    assert small.ear == pytest.approx(a.ear, abs=0.03)


# --------------------------------------------------------------------------- body
@pytest.mark.parametrize("name", ["portrait.jpg", "business-person.png"])
def test_upright_posture(extractor, name):
    _, f = features(extractor, img(name))
    theta, lam, delta = f.posture
    assert abs(theta) < 10 and abs(lam) < 0.15 and 0.4 < delta < 0.8


def test_profile_view_has_no_face_but_has_posture(extractor):
    _, f = features(extractor, img("pose.jpg"))       # side-on yoga pose, facing the image left
    assert not f.face and f.posture is not None and f.posture[1] < -0.2


def test_body_rotation_shows_up_in_shoulder_tilt(extractor):
    im = img("business-person.png")
    _, a = features(extractor, im)
    _, b = features(extractor, rotate(im, 15))
    assert b.posture[0] < a.posture[0] - 8               # counter-clockwise: the image-right shoulder rises


# --------------------------------------------------------------------------- video mode
def test_video_mode_timestamps_and_pose_cache():
    from app.vision.landmarks import LandmarkExtractor
    ex = LandmarkExtractor(CFG.face_model, CFG.pose_model, pose_every=5, video=True)
    rgb = cv2.cvtColor(img("business-person.png"), cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, None, fx=0.5, fy=0.5)
    try:
        results = [ex.process(rgb, ts) for ts in (0, 0, 10, 5, 66, 133)]   # repeats / going backwards
        assert all(r.face is not None and r.pose is not None for r in results)
        assert ex._last_ts > 133 - 1
        late = ex.process(rgb, 5000)       # frame 7: no pose run (every 5th), cache is 4.9 s old
        assert late.face is not None and late.pose is None
    finally:
        ex.close()


@pytest.fixture
def clip(tmp_path):
    """3 s video of the photo slowly rolling (+-10 deg), 640 wide."""
    im = cv2.resize(img("business-person.png"), (640, 906))
    path = str(tmp_path / "drive.avi")
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 15, (640, 906))
    for i in range(45):
        out.write(rotate(im, 10 * np.sin(i / 45 * 2 * np.pi)))
    out.release()
    return path


def test_camera_source_on_a_video_file(clip):
    src = CameraSource(vision_cfg(video_file=clip))
    try:
        frames = [src.read() for _ in range(60)]        # more frames than the clip: it loops
        feats = [f for f, frame in frames if frame is not None]
        assert len(feats) == 60 and all(f.face and f.posture is not None for f in feats)
        rolls = [f.roll for f in feats[:45]]
        assert max(rolls) - min(rolls) == pytest.approx(20, abs=4)   # follows the +-10 deg roll
        jpg = base64.b64decode(src.preview(frames[0][1]))
        assert jpg[:2] == b"\xff\xd8" and len(jpg) > 1000            # a JPEG
    finally:
        src.close()


def test_camera_source_reports_unreadable_video(tmp_path):
    src = CameraSource(vision_cfg(video_file=str(tmp_path / "missing.avi")))
    try:
        f, frame = src.read()
        assert frame is None and not f.face and "could not be opened" in src.note
    finally:
        src.close()
