"""Temporal drowsiness probability p_t (Implementation ch. 6.4).

With DC_TEMPORAL_MODEL=path/to/tcn.onnx a trained TCN is run with ONNX Runtime on the last
SEQ_LEN frames (3 s at 15 fps). Model contract:
  input  float32 [1, T, 7]: ear_norm, mar_rel, pitch, yaw, roll, gaze_h, gaze_v
         (head angles in degrees and gaze relative to the driver's calibrated baseline)
  output the drowsiness probability (any shape; the last value is used)
  optional sidecar tcn.json: {"mean": [7 values], "std": [7 values], "output": "probability"|"logit"}
         for the training-set normalisation (6.4.1).
Without a model, p_t comes from a transparent heuristic over the PERCLOS window.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

SEQ_FEATURES = ("ear_norm", "mar_rel", "pitch", "yaw", "roll", "gaze_h", "gaze_v")
SEQ_LEN = 45


class OnnxTCN:
    def __init__(self, path: str):
        import onnxruntime as ort
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"temporal model not found: '{path}'")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = opts.inter_op_num_threads = 1   # leave the cores to MediaPipe
        self.sess = ort.InferenceSession(str(p), opts, providers=["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.input = inp.name
        if len(inp.shape) != 3:
            raise ValueError(f"TCN input must be [batch, time, features], got {inp.shape}")
        self.seq_len = inp.shape[1] if isinstance(inp.shape[1], int) else SEQ_LEN
        if isinstance(inp.shape[2], int) and inp.shape[2] != len(SEQ_FEATURES):
            raise ValueError(f"TCN expects {inp.shape[2]} features; this pipeline provides {len(SEQ_FEATURES)}")
        meta_path = p.with_suffix(".json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        nf = len(SEQ_FEATURES)
        self.mean = np.asarray(meta.get("mean", [0.0] * nf), dtype=np.float32)
        self.std = np.asarray(meta.get("std", [1.0] * nf), dtype=np.float32)
        if self.mean.shape != (nf,) or self.std.shape != (nf,):
            raise ValueError(f"{meta_path.name}: mean/std need {nf} values")
        self.std[self.std == 0] = 1.0
        self.logit = meta.get("output") == "logit"

    def __call__(self, seq: np.ndarray) -> float:
        x = ((np.asarray(seq, np.float32) - self.mean) / self.std)[None]
        y = float(np.ravel(self.sess.run(None, {self.input: x})[0])[-1])
        if self.logit:
            y = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, y))))
        return float(min(1.0, max(0.0, y)))


# PERCLOS -> evidence anchors: ordinary blinking (~3-6 %) is no evidence, 15 % (the usual drowsiness
# criterion, Wierwille et al.) reaches the MILD threshold, 30 % is severe and reaches HIGH
_PERCLOS_X, _PERCLOS_Y = (0.06, 0.15, 0.30, 0.40), (0.0, 0.50, 0.90, 0.95)


def heuristic(perclos: float, long_closures: int, yawns: int, head_drops: int) -> float:
    """Noisy-OR of the window's evidence: PERCLOS, repeated microsleeps, yawns and head drops."""
    evidence = (
        float(np.interp(perclos, _PERCLOS_X, _PERCLOS_Y)),
        min(1.0, long_closures / 2.0) * 0.85,
        min(1.0, yawns / 3.0) * 0.60,
        min(1.0, head_drops / 2.0) * 0.85,
    )
    return 1.0 - math.prod(1.0 - e for e in evidence)


class TemporalModel:
    def __init__(self, path: str = ""):
        self.tcn: OnnxTCN | None = None
        self.name = "heuristic"
        if path:
            try:
                self.tcn = OnnxTCN(path)
                self.name = "tcn"
                print(f"[vision] TCN loaded: {path}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[vision] TCN unavailable ({e}); using the heuristic", flush=True)

    @property
    def seq_len(self) -> int:
        return self.tcn.seq_len if self.tcn else SEQ_LEN

    def __call__(self, seq: np.ndarray | None, stats: dict) -> float:
        if self.tcn is not None and seq is not None and len(seq) >= self.seq_len:
            return self.tcn(seq[-self.seq_len:])
        return heuristic(stats["perclos"], stats["long_closures"], stats["yawns"], stats["head_drops"])
