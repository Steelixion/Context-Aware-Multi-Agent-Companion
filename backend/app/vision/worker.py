"""Vision process. Runs in its own OS process so LLM/STT/TTS load can never starve it.

Pipeline: source (camera | simulator) -> per-frame features -> sliding window -> temporal
model -> Telemetry JSON -> out_q (latest-wins).
"""
from __future__ import annotations

import base64
import math
import queue
import random
import time

from ..contracts import Telemetry, HeadPose, Posture, level_from_score
from .temporal_model import TemporalFatigueModel
from .window import SlidingWindow


def put_latest(q, item) -> None:
    """Never block the vision loop: drop the oldest item if the queue is full."""
    try:
        q.put_nowait(item)
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(item)
        except queue.Full:
            pass


# --------------------------------------------------------------------------- sources
class SimSource:
    """Synthetic driver. Modes: normal | drowsy | yawning | nodding | no_face (set via ctrl_q)."""

    def __init__(self, ctrl_q):
        self.ctrl_q = ctrl_q
        self.mode = "normal"
        self.t0 = time.time()
        self.next_blink = 0.0
        self.blink_until = 0.0
        self.drowse_until = 0.0

    def _poll(self):
        try:
            while True:
                cmd = self.ctrl_q.get_nowait()
                if cmd.get("type") == "sim":
                    self.mode = cmd["mode"]
                    self.t0 = time.time()
        except queue.Empty:
            pass

    def read(self):
        self._poll()
        now, t = time.time(), time.time() - self.t0
        if self.mode == "no_face":
            return None, None
        ear = 0.31 + random.gauss(0, 0.01)
        mar, pitch, slump = 0.08 + random.gauss(0, 0.01), random.gauss(0, 1.5), 0.15
        if now >= self.next_blink:  # normal blink
            self.blink_until = now + 0.15
            self.next_blink = now + random.uniform(2.0, 5.0 if self.mode == "normal" else 2.5)
        if now < self.blink_until:
            ear = 0.12
        if self.mode == "drowsy":
            slump, pitch = 0.5, -8 + random.gauss(0, 2)
            if now >= self.drowse_until and random.random() < 0.02:
                self.drowse_until = now + 2.2        # microsleep > 1.5 s
            if now < self.drowse_until:
                ear = 0.10 + random.gauss(0, 0.01)
            else:
                ear = 0.23 + random.gauss(0, 0.015)
        elif self.mode == "yawning":
            mar = 0.75 if (t % 6) < 2.5 else 0.1
            ear = min(ear, 0.22) if mar > 0.6 else ear
        elif self.mode == "nodding":
            pitch = -22 * (0.5 + 0.5 * math.sin(t * 1.2))
            slump = 0.4
        return dict(ear=ear, mar=mar, yaw=random.gauss(0, 2), pitch=pitch,
                    roll=random.gauss(0, 1), slump=slump, tilt=1.0), None


class CameraSource:
    def __init__(self, cfg: dict):
        import cv2
        import mediapipe as mp
        self.cv2, self.mp = cv2, mp
        self.cap = cv2.VideoCapture(cfg["camera_index"])
        self.ok = self.cap.isOpened()
        self.face = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.pose = mp.solutions.pose.Pose(model_complexity=0, min_detection_confidence=0.5)
        self.n = 0
        self.posture = (0.0, 0.0)

    def read(self):
        from . import features as F
        import numpy as np
        if not self.ok:
            return None, None
        ok, frame = self.cap.read()
        if not ok:
            return None, None
        frame = self.cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
        self.n += 1
        if self.n % 5 == 0:  # posture is slow-changing: run pose at 1/5 rate
            res = self.pose.process(rgb)
            if res.pose_landmarks:
                p = F.posture_from_pose(res.pose_landmarks.landmark)
                if p:
                    self.posture = p
        res = self.face.process(rgb)
        if not res.multi_face_landmarks:
            return None, frame
        lm = res.multi_face_landmarks[0].landmark
        pts = np.array([(l.x * w, l.y * h) for l in lm], dtype=np.float64)
        ear = (F.eye_aspect_ratio(pts, F.LEFT_EYE) + F.eye_aspect_ratio(pts, F.RIGHT_EYE)) / 2
        yaw, pitch, roll = F.head_pose(pts, w, h)
        return dict(ear=ear, mar=F.mouth_aspect_ratio(pts), yaw=yaw, pitch=pitch, roll=roll,
                    slump=self.posture[0], tilt=self.posture[1]), frame


# --------------------------------------------------------------------------- pipeline
class Pipeline:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.win = SlidingWindow(cfg["window_seconds"], cfg["ear_threshold"])
        self.model = TemporalFatigueModel(cfg.get("temporal_weights", ""))
        self.closed_since: float | None = None

    def step(self, feat: dict | None, source: str, note: str = "") -> dict:
        now = time.time()
        if feat is None:
            self.closed_since = None
            self.win.push(now, 0, 0, 0, 0, 0, 0, False)
            score, st = self.model.predict(self.win)
            return Telemetry(ts=now, face_detected=False, fatigue_score=score,
                             drowsiness_level=level_from_score(score), source=source,
                             note=note or "no face").model_dump()
        self.win.push(now, feat["ear"], feat["mar"], feat["yaw"], feat["pitch"],
                      feat["roll"], feat["slump"], True)
        if feat["ear"] < self.cfg["ear_threshold"]:
            self.closed_since = self.closed_since or now
        else:
            self.closed_since = None
        score, st = self.model.predict(self.win)
        return Telemetry(
            ts=now, face_detected=True, ear=round(feat["ear"], 4), mar=round(feat["mar"], 4),
            head_pose=HeadPose(yaw=round(feat["yaw"], 1), pitch=round(feat["pitch"], 1),
                               roll=round(feat["roll"], 1)),
            posture=Posture(slump_score=round(feat["slump"], 2), shoulder_tilt=round(feat["tilt"], 1)),
            perclos_30s=round(st["perclos"], 3),
            eyes_closed_ms=int((now - self.closed_since) * 1000) if self.closed_since else 0,
            yawns_30s=st["yawns"], fatigue_score=round(score, 3),
            drowsiness_level=level_from_score(score), source=source,
        ).model_dump()


def main(out_q, prev_q, ctrl_q, cfg: dict) -> None:
    mode, note = cfg["vision_mode"], ""
    source = None
    if mode == "camera":
        try:
            source = CameraSource(cfg)
            if not source.ok:
                note = f"camera {cfg['camera_index']} could not be opened"
        except Exception as e:  # noqa: BLE001
            note = f"camera pipeline unavailable: {e}"
            print("[vision]", note)
    if source is None and mode == "camera":
        source = SimSource(ctrl_q)
        source.mode = "no_face"
        mode = "camera-error"
    if source is None:
        source = SimSource(ctrl_q)
    pipe = Pipeline(cfg)
    period, n = 1.0 / cfg["vision_fps"], 0
    print(f"[vision] running, mode={mode}")
    while True:
        t0 = time.time()
        feat, frame = source.read()
        put_latest(out_q, pipe.step(feat, "sim" if isinstance(source, SimSource) else "camera", note))
        n += 1
        if frame is not None and n % 3 == 0:
            import cv2
            small = cv2.resize(frame, (320, int(frame.shape[0] * 320 / frame.shape[1])))
            ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 55])
            if ok:
                put_latest(prev_q, base64.b64encode(jpg.tobytes()).decode())
        time.sleep(max(0.0, period - (time.time() - t0)))
