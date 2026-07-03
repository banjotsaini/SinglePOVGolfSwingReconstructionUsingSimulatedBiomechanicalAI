"""Interventions benchmarked by coaching_accuracy_benchmark.py.

Each is a function (pred_xyz (T,17,3), ctx) -> smoothed/corrected (T,17,3),
applied to the lifted 3D before alignment + measurement. These are the
candidate fixes the iteration loop tests against the ground-truth benchmark.

Smoothing here mirrors what the live coaching pipeline does: temporal filtering
of the lifted 3D trajectory to kill frame-to-frame jitter. It is model-agnostic
(unlike Austin's MediaPipe-trained z-MLP), so it is valid on any lifter's output.
"""
from __future__ import annotations

import numpy as np
from scipy import signal


# ---------------------------------------------------------------------------
def _sg(pred, window=11, poly=2):
    """Savitzky-Golay on every joint/axis trajectory."""
    out = pred.copy()
    T = pred.shape[0]
    w = window if window <= T else (T if T % 2 == 1 else T - 1)
    if w < poly + 2:
        return out
    for j in range(pred.shape[1]):
        for a in range(3):
            out[:, j, a] = signal.savgol_filter(pred[:, j, a], w, poly)
    return out


def _moving_avg(pred, k=5):
    out = pred.copy()
    pad = k // 2
    for j in range(pred.shape[1]):
        for a in range(3):
            x = np.pad(pred[:, j, a], pad, mode="edge")
            out[:, j, a] = np.convolve(x, np.ones(k) / k, mode="valid")
    return out


def _one_euro(pred, fps=30.0, min_cutoff=1.0, beta=0.3, d_cutoff=1.0):
    """1-Euro filter per joint/axis (Casiez et al. 2012). Low lag, jitter-killing."""
    def alpha(cutoff):
        tau = 1.0 / (2 * np.pi * cutoff)
        te = 1.0 / fps
        return 1.0 / (1.0 + tau / te)

    T, J, _ = pred.shape
    out = pred.copy()
    for j in range(J):
        for a in range(3):
            x = pred[:, j, a]
            x_hat = np.empty(T); dx_hat = 0.0
            x_hat[0] = x[0]
            for t in range(1, T):
                dx = (x[t] - x_hat[t - 1]) * fps
                ad = alpha(d_cutoff)
                dx_hat = ad * dx + (1 - ad) * dx_hat
                cutoff = min_cutoff + beta * abs(dx_hat)
                a_ = alpha(cutoff)
                x_hat[t] = a_ * x[t] + (1 - a_) * x_hat[t - 1]
            out[:, j, a] = x_hat
    return out


def _bone_lock(pred, ctx=None):
    """Bone-length lock: per non-root joint, slide along parent->child so the
    bone equals the clip-wide median length (kills depth scale drift)."""
    PARENT = {1: 0, 2: 1, 3: 2, 4: 0, 5: 4, 6: 5, 7: 0, 8: 7, 10: 8,
              11: 8, 12: 11, 13: 12, 14: 8, 15: 14, 16: 15}
    out = pred.copy()
    for child, parent in PARENT.items():
        vec = pred[:, child] - pred[:, parent]
        L = np.linalg.norm(vec, axis=1, keepdims=True)
        med = np.median(L)
        unit = vec / (L + 1e-9)
        out[:, child] = out[:, parent] + unit * med
    return out


# ---------------------------------------------------------------------------
INTERVENTIONS = {
    "sg": lambda p, ctx: _sg(p),
    "sg_w7": lambda p, ctx: _sg(p, window=7),
    "sg_w15": lambda p, ctx: _sg(p, window=15),
    "movavg": lambda p, ctx: _moving_avg(p),
    "oneeuro": lambda p, ctx: _one_euro(p),
    "oneeuro_tight": lambda p, ctx: _one_euro(p, min_cutoff=0.5, beta=0.2),
    "bonelock": lambda p, ctx: _bone_lock(p),
    "sg_bonelock": lambda p, ctx: _bone_lock(_sg(p)),
    "oneeuro_bonelock": lambda p, ctx: _bone_lock(_one_euro(p)),
}
