"""Driver state from per-frame features (Implementation ch. 6.2-6.4).

Each frame is normalised against the driver's calibrated profile, then:
  * events: blinks (< 0.5 s), long closures (>= 1.5 s), yawns, head drops (pitch <= -20 deg for 1.5 s),
    eyes-off-road (gaze / yaw outside the attentive bounds), posture instability (D_t > 1 for 5 s)
  * PERCLOS over the last 60 s with the personalised threshold                       (6.15)
  * telemetry quality: share of usable frames; low quality suppresses alarms           (6.4.2)
  * p_hat = EMA of the temporal model output, R = max(p_hat, r), level from R    (6.16, 6.17)
All per-frame work is O(1): windows keep running sums.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from ..contracts import Calibration, Gaze, HeadPose, Posture, Telemetry, level_from_risk
from .calibration import EAR_BASE_RANGE, Calibrator, Profile, default_profile
from .features import FrameFeatures
from .temporal_model import TemporalModel

MAX_TRACK_YAW = 45.0     # head turned further: tracking quality is poor (6.4.2) and eyes are unreliable
MAX_EYE_PITCH = 50.0     # head pitched further: eye/mouth landmarks are unreliable, but this is a
                         # drowsiness sign (head drop), not bad tracking, so it does not lower quality
INFERENCE_STRIDE = 5                          # run the temporal model every 5 frames (6.4.2)


def wrap180(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


class _Run:
    """How long a condition has held, tolerating dropouts shorter than `gap` seconds."""

    def __init__(self, gap: float):
        self.gap = gap
        self.start: float | None = None
        self.last = 0.0
        self.fired = False   # caller's flag: event already counted for this run

    def update(self, now: float, cond: bool) -> float | None:
        """Returns the length of a run that has just ended, else None."""
        if cond:
            if self.start is None:
                self.start, self.fired = now, False
            self.last = now
        elif self.start is not None and now - self.last > self.gap:
            length, self.start = self.last - self.start, None
            return length
        return None

    @property
    def duration(self) -> float:
        return 0.0 if self.start is None else self.last - self.start


class _Window:
    """Values over the last `seconds`, with a running sum."""

    def __init__(self, seconds: float):
        self.seconds = seconds
        self.q: deque = deque()
        self.sum = 0.0

    def push(self, now: float, value: float) -> None:
        self.q.append((now, value))
        self.sum += value
        while now - self.q[0][0] > self.seconds:
            self.sum -= self.q.popleft()[1]

    def mean(self) -> float:
        return self.sum / len(self.q) if self.q else 0.0

    def __len__(self) -> int:
        return len(self.q)


class _Events:
    """Event timestamps within the last `seconds`."""

    def __init__(self, seconds: float):
        self.seconds = seconds
        self.q: deque = deque()

    def add(self, now: float) -> None:
        self.q.append(now)

    def count(self, now: float) -> int:
        while self.q and now - self.q[0] > self.seconds:
            self.q.popleft()
        return len(self.q)


class DriverMonitor:
    def __init__(self, cfg: dict, profile: Profile | None = None, model: TemporalModel | None = None):
        self.cfg = cfg
        self.profile = profile or default_profile()
        self.model = model or TemporalModel(cfg.get("temporal_model", ""))
        self.calibrator: Calibrator | None = None
        fps = float(cfg["vision_fps"])
        # eye closures and head drops tolerate one dropped / noisy frame. Yawns must be strictly
        # continuous: a yawn's MAR (~1.0) sits far above the threshold (~0.45), while speech dips
        # below it for a frame or two between syllables and must never merge into a "yawn".
        frame_gap = 1.5 / fps
        self.closure, self.drop, self.yawn = _Run(frame_gap), _Run(frame_gap), _Run(0.5 / fps)
        self.offroad, self.slouch = _Run(0.25), _Run(0.5)
        w = cfg["perclos_seconds"]
        self.perclos = _Window(w)
        self.quality = _Window(cfg["quality_seconds"])
        self.blinks, self.long_closures, self.yawns, self.drops = (_Events(w) for _ in range(4))
        self.seq: deque = deque(maxlen=self.model.seq_len)
        self.min_perclos_frames = int(2 * fps)
        self.p_hat = 0.0
        self.n = 0

    # ---- calibration ----
    def start_calibration(self, now: float) -> None:
        c = self.cfg
        self.calibrator = Calibrator(c["calibration_seconds"], c["vision_fps"], c["calibration_timeout"], now)

    def _calibrate(self, f: FrameFeatures, now: float) -> dict | None:
        cal = self.calibrator
        cal.add(f)
        if not cal.finished(now):
            return None
        new = cal.result(self.profile)
        self.calibrator = None
        if new is not None:
            self.profile = new
        return dict(type="calibration", ok=new is not None, frames=cal.n,
                    profile=self.profile.to_dict())

    # ---- per frame ----
    def step(self, f: FrameFeatures, now: float, ts: float, source: str = "camera",
             note: str = "") -> tuple[dict, dict | None]:
        """One frame. `now` is monotonic seconds (durations), `ts` wall-clock (telemetry).
        Returns (telemetry dict, calibration event or None)."""
        c = self.cfg
        event = self._calibrate(f, now) if self.calibrator is not None else None
        p, b = self.profile, self.profile.base
        tau_ear, tau_mar = p.tau_ear(c["ear_kappa"]), p.tau_mar(c["mar_delta"])
        ear_base = min(max(b["ear"], EAR_BASE_RANGE[0]), EAR_BASE_RANGE[1])

        face = f.face and f.ear is not None and f.mar is not None
        has_pose = face and f.pitch is not None
        pitch = f.pitch - b["pitch"] if has_pose else 0.0
        yaw = f.yaw - b["yaw"] if has_pose else 0.0
        roll = wrap180(f.roll - b["roll"]) if has_pose else 0.0
        tracked = has_pose and abs(yaw) <= MAX_TRACK_YAW
        valid = tracked and abs(pitch) <= MAX_EYE_PITCH          # eye and mouth metrics usable
        self.quality.push(now, 1.0 if tracked else 0.0)
        quality = self.quality.mean()
        quality_ok = quality >= c["quality_min"]

        # ---- eyes: blinks, long closures, PERCLOS (only on usable frames) ----
        closed = valid and f.ear < tau_ear
        if valid:
            ended = self.closure.update(now, closed)
            if ended is not None and ended < c["blink_max_seconds"]:
                self.blinks.add(now)
            if self.closure.duration >= c["long_closure_seconds"] and not self.closure.fired:
                self.closure.fired = True
                self.long_closures.add(now)
            self.perclos.push(now, 1.0 if closed else 0.0)
            # ---- yawns (6.3.3: MAR above the personalised threshold for yawn_seconds) ----
            self.yawn.update(now, f.mar > tau_mar)
            if self.yawn.duration >= c["yawn_seconds"] and not self.yawn.fired:
                self.yawn.fired = True
                self.yawns.add(now)
        if tracked:   # ---- head drops: any tracked frame, however far the head has dropped ----
            self.drop.update(now, pitch <= -c["head_drop_degrees"])
            if self.drop.duration >= c["head_drop_seconds"] and not self.drop.fired:
                self.drop.fired = True
                self.drops.add(now)
        eyes_closed_s = self.closure.duration if self.closure.start is not None else 0.0
        long_closure = eyes_closed_s >= c["long_closure_seconds"]
        head_drop = self.drop.duration >= c["head_drop_seconds"]
        yawning = self.yawn.duration >= c["yawn_seconds"]

        # ---- gaze (6.11), relative to the calibrated baseline; only with open eyes ----
        gaze_h = gaze_v = None
        if valid and not closed:
            gh, gv = [], []
            if f.gaze_r is not None:      # driver's right eye: + g_h = towards the image's right
                gh.append(f.gaze_r[0] - b["gh_r"]); gv.append(f.gaze_r[1] - b["gv_r"])
            if f.gaze_l is not None:      # driver's left eye: + g_h = towards the image's left
                gh.append(-(f.gaze_l[0] - b["gh_l"])); gv.append(f.gaze_l[1] - b["gv_l"])
            if gh:
                gaze_h, gaze_v = float(np.mean(gh)), float(np.mean(gv))
        if has_pose:   # extreme head turns count as off-road even though the frame is not "valid"
            off_road = abs(yaw) >= c["yaw_tolerance"] or (gaze_h is not None and abs(gaze_h) >= c["gaze_tolerance"])
            self.offroad.update(now, off_road)
        else:
            off_road = False
        off_road_s = self.offroad.duration if self.offroad.start is not None else 0.0
        distracted = off_road_s >= c["distraction_seconds"]

        # ---- posture (6.12-6.14) ----
        posture = Posture()
        if f.posture is not None:
            theta, lam, delta = f.posture
            t_theta, t_lam, t_delta = c["posture_tolerances"]
            tilt, lean = wrap180(theta - b["theta"]), lam - b["lam"]
            terms = [abs(tilt) / t_theta, abs(lean) / t_lam]
            slump = 0.0
            if b.get("delta", 0.0) > 1e-6:
                slump = max(0.0, 1.0 - delta / b["delta"])
                terms.append(slump / t_delta)
            dev = max(terms)
            self.slouch.update(now, dev > 1.0)
            posture = Posture(visible=True, shoulder_tilt=round(tilt, 1), lean=round(lean, 3),
                              head_height=round(delta, 3), slump_score=round(min(1.0, slump), 3),
                              deviation=round(dev, 2),
                              unstable=self.slouch.duration >= c["posture_seconds"])
        else:
            self.slouch.update(now, False)

        # ---- temporal model, EMA (6.16) and max-fusion (6.17) ----
        mar_rel = f.mar - b["mar"] if face else 0.0
        if valid:
            self.seq.append((f.ear / ear_base, mar_rel, pitch, yaw, roll, gaze_h or 0.0, gaze_v or 0.0))
        perclos = self.perclos.mean() if len(self.perclos) >= self.min_perclos_frames else 0.0
        stats = dict(perclos=perclos, long_closures=self.long_closures.count(now),
                     yawns=self.yawns.count(now), head_drops=self.drops.count(now))
        self.n += 1
        if quality_ok and self.n % INFERENCE_STRIDE == 0:   # low quality: hold p_hat, no false alarms
            seq = np.asarray(self.seq, np.float32) if self.seq else None
            pt = self.model(seq, stats)
            self.p_hat = c["ema_alpha"] * pt + (1.0 - c["ema_alpha"]) * self.p_hat
        r = 1.0 if quality_ok and (long_closure or head_drop) else 0.0
        risk = round(max(self.p_hat, r), 3)   # level and published value must agree at 0.5 / 0.85

        cal_state = "calibrating" if self.calibrator else ("calibrated" if p.calibrated else "default")
        tel = Telemetry(
            ts=ts, face_detected=bool(f.face), quality=round(quality, 2),
            ear=round(f.ear, 4) if face else 0.0, ear_norm=round(f.ear / ear_base, 3) if face else 0.0,
            mar=round(f.mar, 4) if face else 0.0, mar_rel=round(mar_rel, 4),
            head_pose=HeadPose(yaw=round(yaw, 1), pitch=round(pitch, 1), roll=round(roll, 1)),
            gaze=Gaze(h=None if gaze_h is None else round(gaze_h, 3),
                      v=None if gaze_v is None else round(gaze_v, 3),
                      on_road=not off_road, off_road_ms=int(off_road_s * 1000), distracted=distracted),
            posture=posture, window_s=c["perclos_seconds"], perclos=round(perclos, 3),
            eyes_closed_ms=int(eyes_closed_s * 1000), blinks=self.blinks.count(now),
            long_closures=stats["long_closures"], yawns=stats["yawns"], head_drops=stats["head_drops"],
            yawning=yawning, head_drop=head_drop, fatigue_score=round(self.p_hat, 3), risk=risk,
            drowsiness_level=level_from_risk(risk, c["risk_mild"], c["risk_high"]),
            calibration=Calibration(state=cal_state,
                                    progress=round(self.calibrator.progress if self.calibrator else 1.0, 2),
                                    ear_threshold=round(tau_ear, 3), mar_threshold=round(tau_mar, 3)),
            source=source, note=note,
        ).model_dump()
        return tel, event
