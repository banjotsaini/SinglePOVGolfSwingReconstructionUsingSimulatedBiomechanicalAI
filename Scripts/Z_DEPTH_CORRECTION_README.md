# Z-Depth Correction Stage

Austin Lee's GolfPose z-depth optimization, packaged as a reusable pipeline stage.

## What it is

A learned correction for the **depth (z) axis** of single-camera 3D pose. Measured
on the GolfPose Vicon test set (S5/S6):

| Method | Z-MPJPE | Δ vs baseline |
|---|---|---|
| MediaPipe raw (+ Procrustes) | 106.0 mm | — |
| + Savitzky-Golay | 105.8 mm | −0.2 |
| + Limb-length constraint | 106.0 mm | 0.0 |
| **+ Learned residual MLP** | **34.1 mm** | **−71.9** |
| **+ Combined (SG+Limb+MLP)** | **32.7 mm** | **−73.2** |

**~69% / 3.2× improvement**, driven almost entirely by the learned MLP residual.
Per-joint pattern matches the reliability EDA: hips best (~12 mm), wrists/elbows
worst (~57 mm). Full breakdown: [`outputs/comparison/FINDINGS.md`](../outputs/comparison/FINDINGS.md),
[`outputs/comparison/mpjpe_summary.csv`](../outputs/comparison/mpjpe_summary.csv).

## How to use

```python
from z_depth_correction import ZDepthCorrector

corr = ZDepthCorrector()                       # loads trained artifacts from outputs/

# Full MediaPipe world landmarks (T, 33, 3):
fixed = corr.correct_mediapipe_world(world33)  # -> (T, 12, 3) in GolfPose lab space

# Already-mapped 12-joint array (GP_IDX order, root-centered):
fixed = corr.correct_joint_array(mp12)
```

Self-test (loads the real artifacts, runs a forward pass):

```bash
python Scripts/z_depth_correction.py
```

## Pipeline

```
MediaPipe world landmarks (root-centered on mid-hip)
  -> Procrustes alignment into GolfPose lab space   (R, s    — outputs/procrustes_params.json)
  -> Savitzky-Golay smoothing on z                  (optional, ~0.2 mm)
  -> learned residual MLP, ΔZ per joint             (outputs/mlp_model.joblib + scaler — the gain)
```

## Where this is valid — and where it is NOT (read before wiring)

The Procrustes fit and MLP are trained on **MediaPipe world-landmark depth vs
Vicon**, on the 12 GolfPose body joints. They are valid for the
**MediaPipe-world-landmark path only**.

Our production coaching pipeline is `MediaPipe Lite (2D) → MotionBERT-Full (3D) →
scorecard`. MotionBERT's 3D lives in a different normalized, COCO-17 space, so this
corrector is **deliberately not wired into the live MotionBERT → scorecard path** —
applying it there would be out-of-distribution and could corrupt the coaching
numbers.

To bring this gain to the coaching scorecard, re-derive the Procrustes+MLP on
**MotionBERT** 3D output (Austin's documented "swap `pred_list` in `results_dict`,
harness unchanged" next step). The eval harness in [`golfpose_eval.py`](../golfpose_eval.py)
is model-agnostic for exactly this.

## Files

| File | Role |
|---|---|
| `golfpose_eval.py` (repo root) | Full eval harness; **trains** the artifacts (needs the 5.4 GB `golfswing/` dataset, gitignored) |
| `Scripts/z_depth_correction.py` | Reusable inference stage; loads artifacts, no cv2/mediapipe needed |
| `outputs/procrustes_params.json` | R (3×3), s (scalar) alignment |
| `outputs/mlp_model.joblib` + `mlp_scaler.joblib` | Trained ΔZ residual head |
| `outputs/comparison/` | MPJPE summary, per-joint plot, z-trajectory plot, GolfDB demo videos |

The three numpy/scipy correction functions in `z_depth_correction.py` are copied
verbatim from `golfpose_eval.py` to keep the inference stage light. `golfpose_eval.py`
remains the source of truth that trains the artifacts; if its math changes, re-sync.
