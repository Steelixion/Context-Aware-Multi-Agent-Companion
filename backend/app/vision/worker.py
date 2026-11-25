"""Vision process. Runs in its own OS process so LLM/STT/TTS load can never starve it.

Pipeline: source (camera | video file | simulator) -> landmarks (MediaPipe) -> per-frame
features -> DriverMonitor (calibration, events, PERCLOS, temporal model, risk) -> telemetry.

Queues: out_q telemetry (latest wins), prev_q JPEG previews (latest wins), ctrl_q commands in
({"type": "sim"|"calibrate"}), evt_q events out ({"type": "calibration", ...}).
"""
from __future__ import annotations

import base64
import queue
import random
import threading
import time
from types import MappingProxyType

from .calibration import Profile
from .features import FrameFeatures, HeadPose, extract
from .monitor import DriverMonitor

SIM_MODES = ("normal", "drowsy", "yawning", "nodding", "distracted", "slouching", "no_face")
PREVIEW_EVERY, PREVIEW_WIDTH = 3, 320


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
    """Synthetic driver producing raw features like the camera path does. Its neutral pose is
    deliberately not the population default (camera below the face, slightly wide eyes), so
    calibration has something to learn."""
    name = "sim"
    NEUTRAL = MappingProxyType(dict(ear=0.32, mar=0.06, pitch=8.0, yaw=-3.0, roll=1.0, gh=0.02, gv=0.05,
                                    theta=1.5, lam=0.02, delta=0.58))

    def __init__(self, seed: int | None = None, clock=time.monotonic):
        self.rng, self.clock = random.Random(seed), clock
        self.mode = "normal"
        self.t0 = clock()
        self.next_blink = self.blink_until = self.closure_until = 0.0

    def set_mode(self, mode: str) -> None:
        if mode in SIM_MODES:
            self.mode, self.t0 = mode, self.clock()
            self.closure_until = 0.0

    def read(self) -> tuple[FrameFeatures, None]:
        n, g = self.NEUTRAL, self.rng.gauss
        now = self.clock()
        t = now - self.t0
        if self.mode == "no_face":
            return FrameFeatures(), None
        ear, mar = n["ear"] + g(0, 0.008), n["mar"] + g(0, 0.008)
        pitch, yaw, roll = n["pitch"] + g(0, 1.0), n["yaw"] + g(0, 1.2), n["roll"] + g(0, 0.8)
        gh, gv = n["gh"] + g(0, 0.02), n["gv"] + g(0, 0.02)
        theta, lam, delta = n["theta"] + g(0, 0.5), n["lam"] + g(0, 0.01), n["delta"] + g(0, 0.01)
        if now >= self.next_blink:                       # ordinary blink, ~150 ms
            self.blink_until = now + 0.15
            self.next_blink = now + self.rng.uniform(2.0, 5.0)
        if now < self.blink_until:
            ear = 0.09
        if self.mode == "drowsy":                        # heavy lids, slow blinks, microsleeps
            ear = min(ear, 0.25 + g(0, 0.01)) if now >= self.blink_until else 0.09
            pitch -= 6.0
            if now >= self.closure_until and self.rng.random() < 0.015:
                self.closure_until = now + self.rng.uniform(1.8, 2.6)
            if now < self.closure_until:
                ear = 0.08 + g(0, 0.006)
        elif self.mode == "yawning":                     # 4 s yawn every 9 s
            if t % 9.0 < 4.0:
                mar, ear = 0.80 + g(0, 0.03), min(ear, 0.26)
        elif self.mode == "nodding":                     # head drops 25 deg for 2.2 s every 6 s
            if t % 6.0 < 2.2:
                pitch = n["pitch"] - 25.0 + g(0, 1.5)
        elif self.mode == "distracted":                  # looks at the side window 3.5 s of every 5 s
            if t % 5.0 < 3.5:
                yaw, gh = n["yaw"] + 30.0 + g(0, 2.0), n["gh"] + 0.2
        elif self.mode == "slouching":                   # head sinks and leans to one side
            delta, lam, theta = n["delta"] * 0.62, n["lam"] + 0.18, n["theta"] + 6.0
        return FrameFeatures(face=True, ear=ear, mar=mar, pitch=pitch, yaw=yaw, roll=roll,
                             gaze_r=(gh, gv), gaze_l=(-gh, gv), posture=(theta, lam, delta)), None


class LatestFrameReader:
    """Webcam reader thread: always hands out the newest frame (no stale buffered frames) and
    reopens the camera if it disappears."""

    def __init__(self, index: int, width: int = 640, height: int = 480):
        import cv2
        self.cv2, self.index, self.size = cv2, index, (width, height)
        self.frame = None
        self.error = ""
        self._lock, self._stop = threading.Lock(), threading.Event()
        threading.Thread(target=self._run, name="camera", daemon=True).start()

    def _open(self):
        cv2 = self.cv2
        backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if hasattr(cv2, "CAP_DSHOW") else [cv2.CAP_ANY]
        for api in backends:
            cap = cv2.VideoCapture(self.index, api)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return cap
            cap.release()
        return None

    def _run(self) -> None:
        cap = None
        while not self._stop.is_set():
            if cap is None:
                cap = self._open()
                if cap is None:
                    self.error = f"camera {self.index} could not be opened"
                    with self._lock:
                        self.frame = None
                    self._stop.wait(2.0)
                    continue
            ok, frame = cap.read()
            if not ok:
                cap.release()
                cap, self.error = None, f"camera {self.index} stopped delivering frames"
                continue
            self.error = ""
            with self._lock:
                self.frame = frame
        if cap is not None:
            cap.release()

    def read(self):
        with self._lock:
            return self.frame

    def close(self) -> None:
        self._stop.set()


class VideoFileReader:
    """Plays a video file frame by frame, looping at the end (demos and repeatable tests)."""

    def __init__(self, path: str):
        import cv2
        self.cv2, self.cap = cv2, cv2.VideoCapture(path)
        self.error = "" if self.cap.isOpened() else f"video file could not be opened: {path}"

    def read(self):
        if self.error:
            return None
        ok, frame = self.cap.read()
        if not ok:
            self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        return frame if ok else None

    def close(self) -> None:
        self.cap.release()


class CameraSource:
    name = "camera"

    def __init__(self, cfg: dict):
        import cv2

        from .landmarks import LandmarkExtractor
        self.cv2 = cv2
        self.extractor = LandmarkExtractor(cfg["face_model"], cfg["pose_model"], cfg["pose_every"])
        self.head = HeadPose()
        self.reader = VideoFileReader(cfg["video_file"]) if cfg.get("video_file") else \
            LatestFrameReader(cfg["camera_index"])
        self.t0 = time.monotonic()
        self.face = self.pose = None

    @property
    def note(self) -> str:
        return self.reader.error

    def read(self):
        frame = self.reader.read()
        if frame is None:
            self.face = self.pose = None
            return FrameFeatures(), None
        h, w = frame.shape[:2]
        rgb = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
        lm = self.extractor.process(rgb, int((time.monotonic() - self.t0) * 1000))
        self.face, self.pose = lm.face, lm.pose
        return extract(lm.face, lm.pose, w, h, self.head, lm.rotation), frame

    def preview(self, frame) -> str | None:
        """Small mirrored (selfie-view) JPEG with the tracked points drawn on it."""
        cv2 = self.cv2
        img = frame.copy()
        if self.face is not None:
            for i in (33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380, 78, 308, 13, 14, 468, 473):
                cv2.circle(img, (int(self.face[i][0]), int(self.face[i][1])), 2, (80, 220, 120), -1)
        if self.pose is not None and min(self.pose[11][3], self.pose[12][3]) >= 0.5:
            a, b = self.pose[11], self.pose[12]
            cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (240, 180, 60), 2)
        img = cv2.flip(img, 1)
        h, w = img.shape[:2]
        img = cv2.resize(img, (PREVIEW_WIDTH, int(h * PREVIEW_WIDTH / w)))
        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 60])
        return base64.b64encode(jpg.tobytes()).decode() if ok else None

    def close(self) -> None:
        self.reader.close()
        self.extractor.close()


# --------------------------------------------------------------------------- main loop
def _drain(ctrl_q) -> list[dict]:
    cmds = []
    try:
        while True:
            cmds.append(ctrl_q.get_nowait())
    except queue.Empty:
        return cmds


def main(out_q, prev_q, ctrl_q, evt_q, cfg: dict) -> None:
    mode, note = cfg["vision_mode"], ""
    source = None
    if mode == "camera":
        try:
            source = CameraSource(cfg)
        except Exception as e:  # noqa: BLE001
            note = f"camera pipeline unavailable: {e}"
            print("[vision]", note, flush=True)
    if source is None:
        source = SimSource()
        if mode == "camera":          # camera requested but broken: report "no face", never fake data
            source.set_mode("no_face")
    monitor = DriverMonitor(cfg, Profile.from_dict(cfg.get("profile")))
    monitor.start_calibration(time.monotonic())
    print(f"[vision] running: source={source.name}, temporal model={monitor.model.name}, "
          f"profile={'stored' if monitor.profile.calibrated else 'default'}", flush=True)
    period, n = 1.0 / cfg["vision_fps"], 0
    while True:
        t0 = time.monotonic()
        for cmd in _drain(ctrl_q):
            if cmd.get("type") == "calibrate":
                monitor.start_calibration(time.monotonic())
            elif cmd.get("type") == "sim" and isinstance(source, SimSource) and mode == "sim":
                source.set_mode(cmd.get("mode", "normal"))
        feats, frame = source.read()
        tel, event = monitor.step(feats, time.monotonic(), time.time(), source.name,
                                  note or getattr(source, "note", ""))
        put_latest(out_q, tel)
        if event:
            evt_q.put(event)
        n += 1
        if frame is not None and n % PREVIEW_EVERY == 0:
            jpg = source.preview(frame)
            if jpg:
                put_latest(prev_q, jpg)
        time.sleep(max(0.0, period - (time.monotonic() - t0)))
