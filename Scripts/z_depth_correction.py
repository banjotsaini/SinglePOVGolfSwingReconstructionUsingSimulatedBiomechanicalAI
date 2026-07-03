"""Reusable z-depth correction stage (Austin Lee's GolfPose optimization).

This wraps the *trained artifacts* from Austin's `golfpose_eval.py` (repo root)
into a small, dependency-light inference class so the rest of the pipeline can
apply the learned z-depth correction without re-running the whole eval harness.

What it does
------------
Given raw **MediaPipe world landmarks**, it reproduces Austin's best-performing
correction stack on the depth (z) axis:

    MediaPipe world landmarks (root-centered)
      -> Procrustes alignment into GolfPose lab space   (R, s)
      -> Savitzky-Golay temporal smoothing on z         (optional, marginal)
      -> learned residual MLP  (ΔZ per joint)           (the dominant gain)

Measured on the GolfPose Vicon test set (S5/S6), this took z-MPJPE from
**106.0 mm -> 32.7 mm** (~69% / 3.2x). See `outputs/comparison/FINDINGS.md`
and `outputs/comparison/mpjpe_summary.csv` for the full per-method / per-joint
breakdown.

IMPORTANT — where this is valid
-------------------------------
The Procrustes fit and the MLP were trained on **MediaPipe world-landmark depth
vs Vicon ground truth**, on the 12 GolfPose body joints. They are valid for the
MediaPipe-world-landmark path *only*.

Our production coaching pipeline lifts depth with **MotionBERT-Full**, whose 3D
output lives in a different (normalized, COCO-17) space. Applying this corrector
directly to MotionBERT output would be an out-of-distribution misuse and could
corrupt the scorecard — so it is deliberately NOT wired into the live
MotionBERT -> scorecard path. Re-deriving the Procrustes+MLP on MotionBERT 3D
(Austin's documented "swap pred_list" next step) is the clean way to bring this
gain to the coaching numbers.

Artifacts (produced by golfpose_eval.py, committed under outputs/):
  - outputs/procrustes_params.json   R (3x3), s (scalar)
  - outputs/mlp_model.joblib         sklearn MLPRegressor (ΔZ head)
  - outputs/mlp_scaler.joblib        StandardScaler for MLP features

The three numpy/scipy correction functions below are copied verbatim from
`golfpose_eval.py` to keep this module light (no cv2 / mediapipe import needed
just to run the correction). golfpose_eval.py remains the source of truth that
*trains* the artifacts; if its math changes, re-sync these.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import joblib
from scipy import signal

# --- Skeleton constants (verbatim from golfpose_eval.py) ----------------------
# GolfPose body-joint index -> MediaPipe landmark index (12 joints with a direct
# MediaPipe equivalent).
GP_TO_MP = {
    1: 24, 2: 26, 3: 28,    # right leg
    4: 23, 5: 25, 6: 27,    # left leg
    11: 11, 12: 13, 13: 15,  # left arm
    14: 12, 15: 14, 16: 16,  # right arm
}
GP_IDX = sorted(GP_TO_MP.keys())          # [1,2,3,4,5,6,11,12,13,14,15,16]
MP_IDX = [GP_TO_MP[g] for g in GP_IDX]    # corresponding MediaPipe indices
N_JOINTS = len(GP_IDX)

GP_NAMES = {
    1: "r_hip", 2: "r_knee", 3: "r_ankle", 4: "l_hip", 5: "l_knee", 6: "l_ankle",
    11: "l_shoulder", 12: "l_elbow", 13: "l_wrist",
    14: "r_shoulder", 15: "r_elbow", 16: "r_wrist",
}
JOINT_NAMES = [GP_NAMES[g] for g in GP_IDX]

_DEFAULT_OUT = Path(__file__).resolve().parent.parent / "outputs"


# --- Pure correction functions (verbatim from golfpose_eval.py) ---------------
def apply_alignment(mp_arr: np.ndarray, R: np.ndarray, s: float) -> np.ndarray:
    """Apply Procrustes alignment to (n,12,3) MP array -> GP lab space."""
    shape = mp_arr.shape
    flat = mp_arr.reshape(-1, 3)
    aligned = s * (flat @ R)
    return aligned.reshape(shape)


def smooth_sg(arr: np.ndarray, window: int, poly: int) -> np.ndarray:
    """Savitzky-Golay on each joint's z trajectory. arr: (n,12,3)."""
    out = arr.copy()
    n = arr.shape[0]
    if n < window:
        window = n if n % 2 == 1 else n - 1
        if window < poly + 1:
            return out
    for ji in range(N_JOINTS):
        valid = ~np.isnan(arr[:, ji, 2])
        if valid.sum() < window:
            continue
        z = arr[:, ji, 2].copy()
        z[~valid] = np.interp(np.where(~valid)[0], np.where(valid)[0], z[valid])
        out[:, ji, 2] = signal.savgol_filter(z, window, poly)
    return out


def apply_mlp(pred_list, mlp, scaler, temporal_k: int = 3):
    """Apply MLP residual ΔZ correction to a list of (n,12,3) arrays."""
    out = []
    for pred in pred_list:
        corrected = pred.copy()
        n = pred.shape[0]
        for fi in range(n):
            if np.isnan(pred[fi]).any():
                continue
            feats = []
            for k in range(-temporal_k, temporal_k + 1):
                idx = max(0, min(n - 1, fi + k))
                if np.isnan(pred[idx]).any():
                    feats.append(pred[fi].flatten())
                else:
                    feats.append(pred[idx].flatten())
            x = np.concatenate(feats).reshape(1, -1)
            x_s = scaler.transform(x)
            delta_z = mlp.predict(x_s)[0]  # (12,)
            corrected[fi, :, 2] += delta_z
        out.append(corrected)
    return out


# --- The reusable stage -------------------------------------------------------
class ZDepthCorrector:
    """Loads Austin's trained artifacts and applies the z-depth correction.

    Example
    -------
    >>> corr = ZDepthCorrector()                 # loads from outputs/
    >>> fixed = corr.correct_joint_array(mp12)   # (T,12,3) MP world -> GP lab
    """

    # SG window/poly are not persisted as artifacts (tuned at train time and
    # only worth ~0.2 mm); these defaults sit inside Austin's swept range. The
    # MLP residual is the dominant correction (~72 mm of the 73 mm gain).
    def __init__(self, out_dir: Path | str = _DEFAULT_OUT,
                 sg_window: int = 11, sg_poly: int = 2, temporal_k: int = 3):
        out_dir = Path(out_dir)
        with open(out_dir / "procrustes_params.json") as f:
            params = json.load(f)
        self.R = np.asarray(params["R"], dtype=np.float64)
        self.s = float(params["s"])
        self.mlp = joblib.load(out_dir / "mlp_model.joblib")
        self.scaler = joblib.load(out_dir / "mlp_scaler.joblib")
        self.sg_window = sg_window
        self.sg_poly = sg_poly
        self.temporal_k = temporal_k

    def correct_joint_array(self, mp12: np.ndarray, smooth: bool = True) -> np.ndarray:
        """Correct a root-centered (T,12,3) MediaPipe world-landmark array.

        Joints must be in GP_IDX order (see MP_IDX for the source MediaPipe
        indices). Returns (T,12,3) in GolfPose lab space with corrected z.
        """
        mp12 = np.asarray(mp12, dtype=np.float64)
        if mp12.ndim != 3 or mp12.shape[1:] != (N_JOINTS, 3):
            raise ValueError(f"expected (T,{N_JOINTS},3), got {mp12.shape}")
        aligned = apply_alignment(mp12, self.R, self.s)
        if smooth:
            aligned = smooth_sg(aligned, self.sg_window, self.sg_poly)
        return apply_mlp([aligned], self.mlp, self.scaler, self.temporal_k)[0]

    def correct_mediapipe_world(self, world33: np.ndarray, smooth: bool = True) -> np.ndarray:
        """Correct full (T,33,3) MediaPipe world landmarks.

        Selects the 12 mapped joints, root-centers per frame on the mid-hip
        (MediaPipe landmarks 23/24), then applies the correction. Returns
        (T,12,3) in GolfPose lab space (JOINT_NAMES ordering).
        """
        world33 = np.asarray(world33, dtype=np.float64)
        if world33.ndim != 3 or world33.shape[1] < 33 or world33.shape[2] != 3:
            raise ValueError(f"expected (T,33,3) MediaPipe world landmarks, got {world33.shape}")
        mp12 = world33[:, MP_IDX, :].copy()                 # (T,12,3)
        lh = mp12[:, MP_IDX.index(23), :]                   # left hip
        rh = mp12[:, MP_IDX.index(24), :]                   # right hip
        mp12 -= (0.5 * (lh + rh))[:, np.newaxis, :]         # root-center on mid-hip
        return self.correct_joint_array(mp12, smooth=smooth)


def _selftest() -> int:
    """Load the real artifacts and run a forward pass; assert it changes z."""
    corr = ZDepthCorrector()
    print(f"[ok] loaded R det={np.linalg.det(corr.R):.4f} s={corr.s:.4f}")
    print(f"[ok] MLP: {corr.mlp.hidden_layer_sizes} hidden, "
          f"{corr.mlp.n_features_in_} features -> {corr.mlp.n_outputs_} dZ outputs")

    rng = np.random.default_rng(0)
    mp12 = rng.normal(scale=0.3, size=(40, N_JOINTS, 3))    # fake MP world, 40 frames
    out = corr.correct_joint_array(mp12)
    assert out.shape == mp12.shape, out.shape
    assert np.isfinite(out).all(), "non-finite output"

    aligned = apply_alignment(mp12, corr.R, corr.s)
    dz = np.abs(out[..., 2] - aligned[..., 2]).mean()
    assert dz > 1e-6, "MLP made no change to z"
    print(f"[ok] forward pass: out shape {out.shape}, mean |dZ from MLP| = {dz*1000:.1f} mm")
    print("[ok] z-depth corrector is wired and functional.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
