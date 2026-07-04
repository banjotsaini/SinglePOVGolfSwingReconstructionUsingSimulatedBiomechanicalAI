# Model Handoff — MotionBERT-Full 3D Pose (for z-depth work)

**From:** Banjot
**What this is:** the 3D pose output from our best-performing pipeline
(**MediaPipe Lite → MotionBERT-Full**), run on all 1,400 GolfDB clips,
consolidated into one file and joined to GolfDB metadata so you can pair
it with the swing DB directly.

This is the pipeline that topped our 24-model leaderboard. The 3D lift
(MotionBERT-Full) is what produces the **z-depth** you want to optimize.

---

## Files

| File | What |
|---|---|
| `motionbert3d_all_clips.parquet` | 5.1M rows — 3D landmarks for all 1,400 clips, GolfDB-joined |
| `run_model_on_clip.py` | Run the full pipeline on any new mp4 → 3D landmarks |
| `README.md` | This file |

---

## The data: `motionbert3d_all_clips.parquet`

One row per (clip, frame, joint). Columns:

| Column | Meaning |
|---|---|
| `clip_id` | GolfDB clip id — **joins directly to golfDB.pkl `id`** |
| `player`, `club`, `view`, `slow` | GolfDB metadata (pre-joined for convenience) |
| `frame` | frame index within the clip |
| `kp_idx`, `kp_name` | joint index + name (COCO-17 ordering, body joints only) |
| `x`, `y`, `z` | **3D position** in MotionBERT's normalized space |

### Quick start

```python
import pandas as pd
df = pd.read_parquet("motionbert3d_all_clips.parquet")

# Just drivers (952 clips) — matches your EDA focus
drivers = df[df.club == "driver"]

# One clip's 3D trajectory
clip0 = df[df.clip_id == 0]

# Pivot to (frames, joints, xyz) array for one clip
import numpy as np
piv = clip0.pivot_table(index="frame", columns="kp_idx", values=["x","y","z"])
```

### IMPORTANT notes on the z-axis (read before optimizing)

1. **z is camera-relative and normalized, NOT metric.** Values sit roughly
   in [-0.6, 0.9]. The skeleton is root-relative (hip-centered). If your
   z-depth optimization assumes meters or absolute depth, you'll need to
   scale/anchor first.
2. **Body joints only.** Face keypoints (eyes/ears/nose) were dropped —
   they came from an H36M→COCO projection and had no real depth. You have
   the 12 body joints that matter: shoulders, elbows, wrists, hips, knees,
   ankles (+ derived hip_center/spine/thorax/neck if you regenerate).
3. **Skeleton is COCO-17 ordering** here (the eval schema). The raw
   MotionBERT output is H36M-17; if you want the native H36M skeleton with
   hip_center/spine/thorax/neck, regenerate with the runner below (it can
   emit either).
4. **This is single-camera monocular depth** — it's a *learned estimate*,
   not triangulated. That's exactly the thing worth refining with GolfDB's
   multi-view clips, which is what I think you're after.

---

## Running the model on a NEW clip: `run_model_on_clip.py`

If you want to run the pipeline yourself (e.g. on a clip not in GolfDB, or
to get the native H36M skeleton):

```bash
python run_model_on_clip.py path/to/swing.mp4 --out my_clip_3d.parquet
```

Requires the project venv (torch+CUDA, mediapipe, the MotionBERT weights
auto-download from HuggingFace on first run). Pipeline:
`MediaPipe Lite (2D) → COCO→H36M → MotionBERT-Full (3D lift) → parquet`.

The adapters live in `Scripts/adapters/` in the main repo — this runner
imports them. Easiest path: run it from inside the repo with the venv
activated.

---

## How this pairs with GolfDB for z-depth optimization

The thing you flagged: GolfDB has some clips from the same YouTube source
at different camera angles. The idea (as I understand it):

- Take two views of the same swing
- Our MotionBERT z is a monocular *guess* per view
- Use the multi-view geometry to **refine / supervise** the z estimate

The `clip_id` join lets you pull our per-view 3D directly. Austin — if you
group `motionbert3d_all_clips.parquet` by `player` + `view`, you can find
the same player shot face-on vs down-the-line and compare/optimize the z
between them.

(Heads up: GolfDB clips sharing a `youtube_id` are NOT synchronized stereo
pairs — they're independently extracted broadcast segments. We hit this
wall in the multi-view EDA. So exact triangulation isn't free; it's more
of a soft consistency prior. Happy to talk through it.)

---

## What model this actually is (for your notes)

- **2D backbone:** MediaPipe Pose Lite (COCO-17, ~107 FPS CPU)
- **3D lifter:** MotionBERT-Full (DSTformer, 33M params, 243-frame window,
  pretrained AMASS+H36M, fine-tuned H36M-SH)
- **Leaderboard result:** PCE@5 = 0.170 (best of the 24 specialized
  pipelines; the LLM event-detector beat it but isn't a deployable model)
- Full eval + adapters are in the repo under `Scripts/`.
