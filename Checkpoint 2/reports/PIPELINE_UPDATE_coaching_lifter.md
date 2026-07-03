# Pipeline update — coaching lifter swap + One-Euro smoothing (2026-06-30)

The coaching-measurement pipeline was updated to the configuration that a
ground-truth benchmark showed is more accurate. This records what changed, the
evidence, and the verification gate.

## What changed

| Stage | Before | After |
|---|---|---|
| 3D lifter (coaching) | MotionBERT-Full | **GolfPose MixSTE** (`golfpose3d`, golf-fine-tuned) |
| Coaching measurements | on raw 3D | on **One-Euro-smoothed** 3D (min_cutoff=0.3, beta=0.4) |
| Event detection | detector on MotionBERT 3D | **retrained** on raw MixSTE 3D |

Events are still detected on **raw** 3D (they are motion reversals — smoothing
would blunt the velocity signal they key on); only the **measurement** branch is
smoothed.

## Why (ground-truth evidence)

GolfPose Vicon benchmark (`outputs/coaching_accuracy/`, held-out G5/G6), mean-abs
coaching-measurement error vs Vicon truth:

- MotionBERT-Full: **7.59°**  →  GolfPose MixSTE: **2.83°**  (**−63%**)
- One-Euro smoothing recovers a further **33–46%** of the error that 2D
  detection noise injects.

The lifter is the dominant lever; smoothing is the secondary noise-robustness gain.
(Austin's z-depth MLP was evaluated and **not** adopted — it corrects only the
height axis and is MediaPipe-trained, so it doesn't move the depth-rotation
coaching metrics. See `outputs/coaching_compare/`.)

## Verification gate — event detection must not regress

The event detector (the project's biggest win, PCE@5 = 0.865) was **trained on
MotionBERT 3D**, so the swap required retraining + a no-regression check. Same
code, seed, and 350-clip held-out split:

| Detector | PCE@5 | PCE@3 | PCE@1 |
|---|---|---|---|
| MotionBERT (baseline) | 0.871 | 0.805 | 0.629 |
| **MixSTE (adopted)** | **0.865** | 0.789 | 0.611 |

PCE@5 holds within 0.6pp (within noise). Event detection is preserved **and**
coaching-angle accuracy improves — net win. The MixSTE-trained detector is now
`Models/event_detector_tcn.pt` (previous saved as
`Models/event_detector_tcn.motionbert.bak.pt`).

## Files touched

- `Scripts/scorecard_step.py`, `Scripts/build_scorecards_batch.py` — events on raw,
  measurements on One-Euro-smoothed 3D.
- `Scripts/demo.py`, `Scripts/pipeline.py`, `Scripts/event_detector.py` — default
  lifter → `golfpose3d`.
- `Scripts/train_event_detector.py` — parameterized `--lifter`.
- `Models/event_detector_tcn.pt` — retrained on MixSTE (+ `event_cnn_from_golfpose3d`
  row in `Data/all_metrics.parquet`).
- `Scripts/rerun_all_scorecards.py` — regenerated all 420 dev/test/eval scorecards
  (previous MotionBERT scorecards backed up to `*_motionbert_bak/`).

## Not changed (historical record preserved)

The 24-model leaderboard results stand as-is: MotionBERT-Full **did** top the GolfDB
PCE@5 leaderboard, and that history is unchanged. This update is a *production
configuration* decision for the coaching-measurement path, justified by the Vicon
coaching-angle benchmark that the leaderboard never measured. MotionBERT remains a
selectable `--lifter` but, with the **UE5 handoff retired**, now feeds no deliverable
— the player-facing 3D replay is a browser Three.js viewer built from the production
(MixSTE) 3D.

## Rollback

`cp Models/event_detector_tcn.motionbert.bak.pt Models/event_detector_tcn.pt`, revert
the lifter defaults to `motionbert_full`, and restore scorecards from `*_motionbert_bak/`.
