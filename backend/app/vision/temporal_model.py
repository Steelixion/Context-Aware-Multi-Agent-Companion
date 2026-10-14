"""Temporal fatigue model.

If a trained GRU checkpoint is supplied (DC_TEMPORAL_WEIGHTS=path.pt) it is used.
Otherwise a transparent window-statistics heuristic (PERCLOS, microsleeps, yawns, head-down,
slump) produces the score. Either way the score reflects the whole 30 s window, not one frame.
"""
from __future__ import annotations

import numpy as np

from .window import SlidingWindow, EAR, MAR, PITCH, SLUMP, FACE

STEPS = 60  # window resampled to 60 steps


class _GRU:
    def __init__(self, path: str):
        import torch
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.gru = nn.GRU(5, 32, batch_first=True)
                self.head = nn.Linear(32, 1)

            def forward(self, x):
                o, _ = self.gru(x)
                return torch.sigmoid(self.head(o[:, -1]))

        self.torch = torch
        self.net = Net()
        self.net.load_state_dict(torch.load(path, map_location="cpu"))
        self.net.eval()

    def __call__(self, a: np.ndarray) -> float:
        a = a[a[:, FACE] > 0]
        if len(a) < 10:
            return 0.0
        idx = np.linspace(0, len(a) - 1, STEPS).astype(int)
        x = a[idx][:, [EAR, MAR, PITCH, SLUMP, FACE]].copy()
        x[:, 2] = (x[:, 2] - np.median(x[:, 2])) / 30.0
        with self.torch.no_grad():
            return float(self.net(self.torch.from_numpy(x).float()[None]).item())


class TemporalFatigueModel:
    def __init__(self, weights_path: str = ""):
        self.gru = None
        self.ema = 0.0
        if weights_path:
            try:
                self.gru = _GRU(weights_path)
                print(f"[vision] GRU temporal model loaded: {weights_path}")
            except Exception as e:  # noqa: BLE001
                print(f"[vision] GRU unavailable ({e}); using heuristic")

    def predict(self, win: SlidingWindow) -> tuple[float, dict]:
        s = win.stats()
        if self.gru is not None:
            raw = self.gru(win.array())
        else:
            raw = (min(s["perclos"] / 0.35, 1.0) * 0.45
                   + min(s["longest_closure"] / 2.0, 1.0) * 0.30
                   + min(s["yawns"] / 3.0, 1.0) * 0.12
                   + min(s["head_down"] / 0.3, 1.0) * 0.08
                   + s["slump"] * 0.05)
        # rise fast on evidence, decay slowly -> "sustained", not flickering
        k = 0.35 if raw > self.ema else 0.05
        self.ema += k * (raw - self.ema)
        return float(np.clip(self.ema, 0.0, 1.0)), s
