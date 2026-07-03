# Does Austin's z-depth + jitter smoothing make the coaching measurements more accurate?

A ground-truth benchmark, built because GolfDB has no 3D truth but **GolfPose Vicon does**.

## Method

- **Truth:** GolfPose Vicon 3D (`golfpose_data/data_3d_golf_gt.npz`), test subjects G5/G6 (8 swings, held out from the lifter's training).
- **Estimate:** lift the Vicon 2D ground-truth keypoints to 3D with a real lifter, best-fit align to the Vicon 3D (one Procrustes rotation+scale per swing), then compute the coaching geometric quantities (shoulder/hip turn, spine tilt, arm bend, knee flex) from both — every frame.
- **Metric:** mean-abs measurement error vs truth, in degrees.
- **Realistic noise:** clean 2D GT has no detection jitter, so we inject calibrated Gaussian 2D jitter (6/12 px) to mimic MediaPipe and see what smoothing recovers.

Harness: `Scripts/coaching_accuracy_benchmark.py`, `Scripts/run_accuracy_sweep.py`, `Scripts/benchmark_interventions.py`.

## Result 1 — the biggest accuracy lever is the LIFTER, not smoothing

On clean 2D input (no jitter):

| Lifter | Coaching-measurement error |
|---|---|
| MotionBERT-Full (**the current app's lifter**) | **7.59°** |
| GolfPose MixSTE (golf-fine-tuned) | **2.83°** |

Swapping to the golf-domain lifter cuts measurement error **63%** — far more than any post-hoc smoothing. MotionBERT's error is *systematic lift bias* on golf poses, which no temporal filter can remove (smoothing it recovers only ~9%).

## Result 2 — jitter smoothing helps, but only under noise and only on a good lifter

Overall error (deg), MixSTE lifter:

| Intervention | 0px | 6px | 12px |
|---|---|---|---|
| none (baseline) | 2.83 | 7.69 | 12.52 |
| Savitzky-Golay | 2.83 | 5.50 | 7.62 |
| **One-Euro** | 3.04 | **5.17** | **6.75** |
| bone-length lock | 2.84 | 7.81 | 13.03 |

- **Clean input:** smoothing does nothing (One-Euro slightly *hurts* via lag). There is no jitter to remove.
- **Under realistic noise:** One-Euro cuts error **33% @ 6px, 46% @ 12px**. Tuned (min_cutoff=0.3, beta=0.4): **45% @ 10px**.
- On MotionBERT the same smoothing recovers only ~9% — because its error is bias, not jitter. **Smoothing pays off in proportion to how good the lifter already is.**

## Result 3 — two repo claims do NOT hold for coaching-measurement accuracy

- **Bone-length lock:** negligible / slightly harmful here (7.81° vs 7.69° @ 6px). After per-swing scale alignment, bone length isn't the error mode; forcing the median length can fight the truth. (It may still help raw jitter-% — a different metric.)
- **Austin's z-depth MLP:** not applicable to these metrics. It corrects only the **height** axis and is trained on **MediaPipe world landmarks**; the marquee coaching metrics (shoulder/hip turn, X-factor) are horizontal-plane rotations on a different pipeline, so the z-MLP leaves them unchanged (see `outputs/coaching_compare/`).

## Bottom line — measurable performance increase

The user's hypothesis ("z-depth + jitter smoothing → more accurate coaching measurements") is **half right, and the loop found the bigger lever**:

1. **Lifter swap (MotionBERT → golf-fine-tuned MixSTE): −63%** measurement error on clean input (7.59° → 2.83°). This is the dominant, structural fix.
2. **One-Euro smoothing of the lifted 3D (min_cutoff=0.3, beta=0.4): −33–46%** of the error that detection noise injects — and it stacks best on the good lifter.
3. Austin's *specific* z-depth MLP and bone-lock do **not** move these coaching measurements; the value is in the lifter + model-agnostic temporal smoothing.

**Recommended production config:** golf-fine-tuned MixSTE lifter + One-Euro (0.3 / 0.4) on the lifted 3D. End-to-end, this takes the app's coaching measurements from ~7.6° error (clean) / ~13° (noisy) on the current MotionBERT path toward ~2.8° (clean) / ~6° (noisy).

## Files
| File | Role |
|---|---|
| `Scripts/coaching_accuracy_benchmark.py` | GT benchmark (Vicon truth vs lifted estimate) |
| `Scripts/benchmark_interventions.py` | smoothing/correction candidates |
| `Scripts/run_accuracy_sweep.py` | noise × intervention sweep + figure |
| `Scripts/tune_smoothing_accuracy.py` | One-Euro parameter tuning |
| `outputs/coaching_accuracy/sweep_{mixste,motionbert}.{json,png}` | results |
| `outputs/coaching_accuracy/oneeuro_tuning.json` | tuned params |
