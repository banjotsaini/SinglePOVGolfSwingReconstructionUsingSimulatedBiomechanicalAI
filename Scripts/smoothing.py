"""Temporal smoothing for 3D pose sequences before Unreal Engine handoff.

Operates on (T, 17, 3) H36M-17 arrays (the lifter output loaded by
export_ue5.load_3d_parquet_as_h36m). Insert one call to `smooth_sequence`
between lifting and export to de-jitter the CSV/JSON/BVH and the HTML preview
at once.

Golf constraint: impact is 2-3 frames of very fast wrist/club motion, so the
default filter is the One Euro filter, which backs off its smoothing as joint
speed rises and therefore preserves the impact snap. Savitzky-Golay is offered
as an offline alternative that preserves derivatives well on slower clips.

Pipeline order inside `smooth_sequence`:
  1. interpolate_gaps  - linear-fill short low-confidence runs (needs conf)
  2. temporal filter   - one_euro_filter | savgol_smooth
  3. enforce_bone_lengths (optional) - rigidify the skeleton frame-to-frame
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from export_ue5 import H36M17_PARENTS


# ---------------------------------------------------------------------------
# 1. Gap handling
# ---------------------------------------------------------------------------

def interpolate_gaps(xyz: np.ndarray,
                     conf: np.ndarray | None,
                     min_conf: float = 0.3,
                     max_gap: int = 5) -> np.ndarray:
    """Linearly interpolate joint positions across short low-confidence runs.

    A "gap" is a maximal run of consecutive frames where conf < min_conf for a
    given joint. Gaps no longer than `max_gap` and bounded by valid frames on
    both sides are filled by linear interpolation; longer gaps and gaps at the
    sequence ends are left untouched (no reliable anchor to interpolate from).

    If `conf` is None this is a no-op (the caller has no per-joint confidence).
    """
    if conf is None:
        return xyz.astype(np.float32, copy=True)

    out = xyz.astype(np.float32, copy=True)
    T, J, _ = out.shape
    for j in range(J):
        valid = conf[:, j] >= min_conf
        if valid.all() or valid.sum() < 2:
            continue
        t = 0
        while t < T:
            if valid[t]:
                t += 1
                continue
            start = t
            while t < T and not valid[t]:
                t += 1
            end = t  # first valid frame after the gap (or T)
            gap_len = end - start
            if start == 0 or end == T or gap_len > max_gap:
                continue  # unbounded or too long -> leave as-is
            lo, hi = start - 1, end
            for k in range(start, end):
                w = (k - lo) / (hi - lo)
                out[k, j] = (1.0 - w) * out[lo, j] + w * out[hi, j]
    return out


# ---------------------------------------------------------------------------
# 2a. One Euro filter (adaptive low-pass, default)
# ---------------------------------------------------------------------------

def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


def one_euro_filter(xyz: np.ndarray,
                    fps: float = 30.0,
                    min_cutoff: float = 1.0,
                    beta: float = 0.3,
                    d_cutoff: float = 1.0) -> np.ndarray:
    """One Euro filter (Casiez et al. 2012) applied per joint and axis.

    Adaptive low-pass: at low speeds the cutoff approaches `min_cutoff`
    (heavy smoothing, kills tremor); as speed rises the cutoff grows by
    `beta * |velocity|` (light smoothing, low lag). Tuning:
      - lower min_cutoff -> smoother slow phases (more lag)
      - higher beta      -> snappier fast phases (less lag at impact)
    """
    T = xyz.shape[0]
    if T < 2:
        return xyz.astype(np.float32, copy=True)

    dt = 1.0 / float(fps) if fps and fps > 0 else 1.0 / 30.0
    out = np.empty_like(xyz, dtype=np.float32)
    x_prev = xyz[0].astype(np.float32)
    dx_prev = np.zeros_like(x_prev)
    out[0] = x_prev
    a_d = _alpha(d_cutoff, dt)
    for t in range(1, T):
        x = xyz[t].astype(np.float32)
        dx = (x - x_prev) / dt
        dx_hat = a_d * dx + (1.0 - a_d) * dx_prev
        cutoff = min_cutoff + beta * np.abs(dx_hat)
        # vectorized alpha per joint/axis
        tau = 1.0 / (2.0 * math.pi * cutoff)
        a = 1.0 / (1.0 + tau / dt)
        x_hat = a * x + (1.0 - a) * x_prev
        out[t] = x_hat
        x_prev = x_hat
        dx_prev = dx_hat
    return out


# ---------------------------------------------------------------------------
# 2b. Savitzky-Golay (offline alternative)
# ---------------------------------------------------------------------------

def savgol_smooth(xyz: np.ndarray,
                  window: int = 7,
                  polyorder: int = 2) -> np.ndarray:
    """Savitzky-Golay polynomial smoothing along the time axis.

    Window is forced odd and clamped to the sequence length; polyorder is
    clamped below the window. Offline (uses future frames) and preserves
    peaks/derivatives better than a moving average.
    """
    from scipy.signal import savgol_filter

    T = xyz.shape[0]
    win = int(window)
    if win % 2 == 0:
        win += 1
    if win > T:
        win = T if T % 2 == 1 else T - 1
    if win < 3:
        return xyz.astype(np.float32, copy=True)
    poly = min(int(polyorder), win - 1)
    return savgol_filter(xyz, window_length=win, polyorder=poly,
                         axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# 3. Bone-length stabilization
# ---------------------------------------------------------------------------

def enforce_bone_lengths(xyz: np.ndarray,
                         parents: tuple[int, ...] = H36M17_PARENTS,
                         ref: str = "median") -> np.ndarray:
    """Rigidify the skeleton: re-project each joint so every bone keeps a
    constant length across all frames.

    For each bone (child -> parent) a reference length is taken as the
    median (or mean) of its per-frame lengths. Then, walking the skeleton
    root->leaf, each child is placed along its current bone direction at the
    reference length from the already-corrected parent. The root joint is
    untouched, so global translation is preserved.

    Relies on H36M17_PARENTS having parent index < child index (true for the
    H36M-17 ordering), so a single forward pass processes parents first.
    """
    T, J, _ = xyz.shape
    out = xyz.astype(np.float32, copy=True)

    # reference length per joint (0 for root)
    ref_len = np.zeros(J, dtype=np.float32)
    for j in range(J):
        p = parents[j]
        if p < 0:
            continue
        lengths = np.linalg.norm(xyz[:, j] - xyz[:, p], axis=1)
        ref_len[j] = np.median(lengths) if ref == "median" else lengths.mean()

    for t in range(T):
        for j in range(J):
            p = parents[j]
            if p < 0:
                continue
            vec = out[t, j] - out[t, p]
            norm = np.linalg.norm(vec)
            if norm < 1e-8:
                # degenerate frame: keep parent position
                out[t, j] = out[t, p]
            else:
                out[t, j] = out[t, p] + vec / norm * ref_len[j]
    return out


# ---------------------------------------------------------------------------
# 4. Floor leveling + grounding
# ---------------------------------------------------------------------------

def level_and_ground(xyz: np.ndarray,
                     l_ankle: int = 6,
                     r_ankle: int = 3,
                     still_quantile: float = 0.3,
                     max_tilt_deg: float = 20.0) -> tuple[np.ndarray, dict]:
    """Level the stance and pin the feet to y=0 (camera coords, y DOWN).

    The lifter output is hip-relative in CAMERA orientation, so a pitched or
    rolled phone leaks depth/side offsets into apparent height — measured on
    GolfDB as a median 1.3%-of-height ankle gap that makes one leg float above
    the viewer's floor plane. Both feet are planted at address, so the
    ankle-to-ankle line during the stillest frames is the one reliable
    horizontal in the scene:

      1. find the stillest `still_quantile` of frames (median joint speed),
      2. rotate the whole trajectory (about the mid-ankle pivot) by the
         minimal rotation that zeroes the y-component of the median
         ankle-to-ankle vector — corrects camera roll on face-on views and
         camera pitch on down-the-line views in one step,
      3. translate so the address ankle level sits exactly at y = 0.

    Rotations beyond `max_tilt_deg` are refused (bad tracking / real slope).
    Default ankle indices are H36M-17; pass COCO or custom indices for other
    orderings. Returns (xyz_out, info) where info records what was applied —
    downstream replay JSON should carry {"grounded": true, "floor_y": 0.0}
    so viewers can place the floor plane exactly instead of guessing.
    """
    T = xyz.shape[0]
    out = xyz.astype(np.float32, copy=True)
    info = {"applied": False, "tilt_deg": 0.0, "floor_shift": 0.0}
    if T < 2:
        return out, info

    # stillest frames by median joint speed (address / post-finish)
    v = np.median(np.linalg.norm(np.diff(out, axis=0), axis=2), axis=1)
    v = np.concatenate([v[:1], v])
    still = v <= np.quantile(v, still_quantile)
    if not still.any():
        still = np.ones(T, dtype=bool)

    ankle_vec = np.median(out[still, l_ankle] - out[still, r_ankle], axis=0)
    norm = float(np.linalg.norm(ankle_vec))
    if norm > 1e-6:
        tilt = math.degrees(math.asin(float(np.clip(ankle_vec[1] / norm, -1, 1))))
        target = ankle_vec.copy()
        target[1] = 0.0
        tnorm = float(np.linalg.norm(target))
        if 0.5 < abs(tilt) <= max_tilt_deg and tnorm > 1e-6:
            a = ankle_vec / norm
            b = target / tnorm
            axis = np.cross(a, b)
            axis_n = float(np.linalg.norm(axis))
            if axis_n > 1e-8:
                axis = axis / axis_n
                angle = math.atan2(axis_n, float(np.dot(a, b)))
                # Rodrigues rotation matrix
                K = np.array([[0, -axis[2], axis[1]],
                              [axis[2], 0, -axis[0]],
                              [-axis[1], axis[0], 0]], dtype=np.float32)
                R = (np.eye(3, dtype=np.float32) + math.sin(angle) * K
                     + (1 - math.cos(angle)) * (K @ K))
                pivot = np.median((out[still, l_ankle] + out[still, r_ankle]) / 2,
                                  axis=0).astype(np.float32)
                out = (out - pivot) @ R.T + pivot
                info["applied"] = True
                info["tilt_deg"] = float(tilt)

    # ground: address ankle level (the lower foot) -> y = 0
    floor_y = float(np.median(np.maximum(out[still, l_ankle, 1],
                                         out[still, r_ankle, 1])))
    out[..., 1] -= floor_y
    info["floor_shift"] = floor_y
    return out, info


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def smooth_sequence(xyz: np.ndarray,
                    conf: np.ndarray | None = None,
                    method: str = "oneeuro",
                    fps: float = 30.0,
                    bone_lock: bool = True,
                    min_conf: float = 0.3,
                    max_gap: int = 5,
                    min_cutoff: float = 1.0,
                    beta: float = 0.3,
                    window: int = 7,
                    polyorder: int = 2) -> np.ndarray:
    """Smooth a (T, 17, 3) H36M-17 sequence.

    method: "none" | "oneeuro" | "savgol". Gap interpolation runs first when
    `conf` is provided; bone-length stabilization runs last when `bone_lock`.
    """
    if xyz.ndim != 3 or xyz.shape[1:] != (17, 3):
        raise ValueError(f"expected (T,17,3) H36M array, got {xyz.shape}")

    out = interpolate_gaps(xyz, conf, min_conf=min_conf, max_gap=max_gap)

    if method == "none":
        pass
    elif method == "oneeuro":
        out = one_euro_filter(out, fps=fps, min_cutoff=min_cutoff, beta=beta)
    elif method == "savgol":
        out = savgol_smooth(out, window=window, polyorder=polyorder)
    else:
        raise ValueError(f"unknown smoothing method '{method}'")

    if bone_lock:
        out = enforce_bone_lengths(out)
    return out
