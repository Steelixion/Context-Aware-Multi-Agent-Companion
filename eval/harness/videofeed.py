"""Wall-clock video source: pushes frames to the OBS Virtual Camera at the clip's original FPS
on the monotonic clock. If the harness falls behind, frames are skipped (never slowed down),
so clip time and wall time stay locked; skipped frames are counted.

Phases: 'calibration' (a neutral segment, looped until the app reports calibrated), then
'clip' (the test clip, once), then 'hold' (last frame frozen while late events are recorded).
Overrides during the clip: blank (black frames, no face) and stop (camera closed).
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np

from .events import EventLog


@dataclass
class Segment:
    path: str
    start_s: float
    end_s: float | None
    loop: bool = False
    phase: str = "clip"
    blanks: list = field(default_factory=list)   # [(start_s, end_s)] relative to segment start


class VideoFeed:
    def __init__(self, log: EventLog, width: int, height: int, fps: float, backend: str = "obs"):
        self.log, self.w, self.h, self.fps, self.backend = log, width, height, fps, backend
        self._lock = threading.Lock()
        self._seg: Segment | None = None
        self._next: Segment | None = None
        self._stop = threading.Event()
        self._closed = threading.Event()
        self.clip_started = threading.Event()
        self.clip_finished = threading.Event()
        self.clip_t0: float | None = None
        self.sent = self.skipped = 0
        self.frame_times: list[tuple[str, int, float]] = []   # (phase, source frame, mono)
        self.error: str | None = None
        self.device = ""
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, name="video-feed", daemon=True)

    def start(self, first: Segment, timeout: float = 15.0) -> bool:
        self._seg = first
        self._thread.start()
        self._ready.wait(timeout)
        return self.error is None

    def queue_next(self, seg: Segment) -> None:
        """Switch to seg at the next frame (ends a looping calibration segment)."""
        with self._lock:
            self._next = seg

    def stop_camera(self) -> None:
        """S12a: close the virtual camera (the app's reader keeps running)."""
        self._closed.set()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(10)

    def _open(self, cv2, seg: Segment):
        cap = cv2.VideoCapture(seg.path)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video {seg.path}")
        src_fps = cap.get(cv2.CAP_PROP_FPS) or self.fps
        start_f = int(round(seg.start_s * src_fps))
        end_f = int(round(seg.end_s * src_fps)) if seg.end_s is not None else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        return cap, src_fps, start_f, end_f

    def _run(self) -> None:
        import cv2
        import pyvirtualcam
        black = np.zeros((self.h, self.w, 3), np.uint8)
        try:
            cam = pyvirtualcam.Camera(width=self.w, height=self.h, fps=self.fps,
                                      fmt=pyvirtualcam.PixelFormat.BGR, backend=self.backend)
        except Exception as e:  # noqa: BLE001
            self.error = f"virtual camera unavailable: {e!r}"
            self.log.add("harness", "video_error", error=self.error)
            self._ready.set()
            return
        self.device = cam.device
        self.log.add("harness", "video_device", device=cam.device, width=self.w, height=self.h, fps=self.fps)
        self._ready.set()
        last = black
        try:
            seg = self._seg
            while seg is not None and not self._stop.is_set():
                cap, src_fps, start_f, end_f = self._open(cv2, seg)
                n_frames = max(0, end_f - start_f)
                t0 = time.monotonic()
                if seg.phase == "clip":
                    self.clip_t0 = t0
                    self.log.add("harness", "clip_start", t=t0, path=seg.path, start_s=seg.start_s,
                                 end_s=seg.end_s, fps=src_fps, frames=n_frames)
                    self.clip_started.set()
                else:
                    self.log.add("harness", f"{seg.phase}_start", t=t0, path=seg.path,
                                 start_s=seg.start_s, end_s=seg.end_s)
                pos = 0                      # next source frame index to decode (relative)
                switched = False
                while not self._stop.is_set():
                    if self._closed.is_set():
                        break
                    with self._lock:
                        nxt = self._next
                    if nxt is not None and seg.phase != "clip":
                        with self._lock:
                            self._next = None
                        seg, switched = nxt, True
                        break
                    idx = int((time.monotonic() - t0) * src_fps)
                    if idx >= n_frames:
                        if seg.loop:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
                            t0, pos = time.monotonic(), 0
                            continue
                        break
                    while pos < idx:          # behind schedule: skip, never slow down
                        cap.grab()
                        pos += 1
                        self.skipped += 1
                    ok, frame = cap.read()
                    pos += 1
                    if not ok:
                        break
                    if frame.shape[1] != self.w or frame.shape[0] != self.h:
                        frame = cv2.resize(frame, (self.w, self.h))
                    rel = idx / src_fps
                    if any(a <= rel < b for a, b in seg.blanks):
                        frame = black
                    cam.send(frame)
                    last = frame
                    self.sent += 1
                    self.frame_times.append((seg.phase, start_f + idx, time.monotonic()))
                    due = t0 + (idx + 1) / src_fps
                    time.sleep(max(0.0, due - time.monotonic()))
                cap.release()
                if switched:
                    continue
                if self._closed.is_set():
                    break                      # camera stopped by the scenario: the clip did NOT finish
                if seg.phase == "clip":
                    self.log.add("harness", "clip_end", sent=self.sent, skipped=self.skipped)
                    self.clip_finished.set()
                    break
                seg = None
            if self._closed.is_set():
                self.log.add("harness", "video_stopped")
                cam.close()
                return
            # hold: keep the last frame on screen until stopped (no fake 'face lost')
            while not self._stop.is_set():
                cam.send(last)
                cam.sleep_until_next_frame()
            cam.close()
        except Exception as e:  # noqa: BLE001
            self.error = repr(e)
            self.log.add("harness", "video_error", error=repr(e))
            try:
                cam.close()
            except Exception:  # noqa: BLE001
                pass
