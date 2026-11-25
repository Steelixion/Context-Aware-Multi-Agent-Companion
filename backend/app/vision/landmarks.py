"""MediaPipe Tasks: Face Landmarker (478 points incl. iris) + Pose Landmarker (33 points).

Returns landmarks as pixel arrays of the unmirrored frame:
  face (478, 3): x, y, z (z in the same scale as x)
  pose (33, 4):  x, y, z, visibility
plus the head rotation from the Face Landmarker's facial transformation matrix (MediaPipe's fit of
its canonical 3D face to all landmarks), converted to the OpenCV camera frame.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple

import numpy as np

# MediaPipe/TFLite log a page of warnings per model at start-up; keep the server console readable
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

POSE_MAX_AGE_MS = 1000   # a cached pose older than this is not reused
GL_TO_CV = np.diag([1.0, -1.0, -1.0])   # MediaPipe geometry is y-up / z-to-viewer; OpenCV is y-down / z-away


class Landmarks(NamedTuple):
    face: np.ndarray | None
    pose: np.ndarray | None
    rotation: np.ndarray | None   # 3x3 head rotation in the camera frame


class LandmarkExtractor:
    def __init__(self, face_model: str, pose_model: str = "", pose_every: int = 5, video: bool = True):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        if not face_model or not Path(face_model).is_file():
            raise FileNotFoundError(f"Face Landmarker model not found: '{face_model}' "
                                    "(run: python scripts/download_models.py)")
        self.mp, self.video = mp, video
        mode = vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE
        self.face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(face_model)), running_mode=mode, num_faces=1,
            min_face_detection_confidence=0.5, min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5, output_facial_transformation_matrixes=True))
        self.pose = None
        if pose_model and Path(pose_model).is_file():
            self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(pose_model)), running_mode=mode,
                num_poses=1, min_pose_detection_confidence=0.5, min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5))
        self.pose_every = max(1, pose_every)
        self._n, self._last_ts = 0, -1
        self._pose: np.ndarray | None = None
        self._pose_ts = -10**9

    def process(self, rgb: np.ndarray, ts_ms: int = 0) -> Landmarks:
        """RGB uint8 frame -> landmarks. VIDEO mode needs strictly increasing timestamps."""
        h, w = rgb.shape[:2]
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        ts = max(int(ts_ms), self._last_ts + 1)
        self._last_ts = ts
        r = self.face.detect_for_video(img, ts) if self.video else self.face.detect(img)
        face = rotation = None
        if r.face_landmarks:
            face = np.array([(p.x * w, p.y * h, p.z * w) for p in r.face_landmarks[0]], dtype=np.float64)
            if r.facial_transformation_matrixes:
                m = np.asarray(r.facial_transformation_matrixes[0], dtype=np.float64)[:3, :3]
                u, _, vt = np.linalg.svd(m)            # nearest pure rotation (drop any scale)
                rotation = GL_TO_CV @ (u @ vt) @ GL_TO_CV
        if self.pose is not None and (self._n % self.pose_every == 0 or not self.video):
            pr = self.pose.detect_for_video(img, ts) if self.video else self.pose.detect(img)
            self._pose = None
            if pr.pose_landmarks:
                self._pose = np.array([(p.x * w, p.y * h, p.z * w, p.visibility if p.visibility is not None else 0.0)
                                       for p in pr.pose_landmarks[0]], dtype=np.float64)
            self._pose_ts = ts
        self._n += 1
        pose = self._pose if ts - self._pose_ts <= POSE_MAX_AGE_MS else None
        return Landmarks(face, pose, rotation)

    def close(self) -> None:
        for m in (self.face, self.pose):
            if m is not None:
                m.close()
