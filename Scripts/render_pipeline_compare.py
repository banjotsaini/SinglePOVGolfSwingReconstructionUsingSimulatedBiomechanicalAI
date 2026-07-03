"""End-to-end pipeline comparison video: baseline vs the measured improvement.

2x2 layout (shared inputs on top, the 3D divergence on the bottom):
   ┌───────────────────────┬───────────────────────┐
   │ RAW VIDEO             │ 2D POSE (MediaPipe)   │
   ├───────────────────────┼───────────────────────┤
   │ 3D — BASELINE         │ 3D — IMPROVED         │
   │ MotionBERT, raw       │ MixSTE + One-Euro     │
   └───────────────────────┴───────────────────────┘

Baseline = the current app path (MotionBERT-Full, no smoothing).
Improved  = the config the GT benchmark showed is more accurate
            (golf-fine-tuned GolfPose MixSTE + One-Euro smoothing).

Both 3D panels self-normalize (hip-centered orthographic); the visible story is
jitter (smoothing) — the accuracy story (lifter) is in the scorecards.

Output: outputs/pipeline_compare/<clip>_pipeline_compare.mp4
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eval_utils import video_info
from export_ue5 import load_3d_parquet_as_h36m
from smoothing import smooth_sequence
from render_progression_4panel import (
    load_2d_landmarks, project_xyz_ortho, draw_2d_overlay,
    draw_3d_skeleton_ortho, label_panel, ffmpeg_path,
)

PROJECT_ROOT = Path(__file__).parent.parent
VIDEO_DIR = PROJECT_ROOT / "Data" / "videos_160"
CACHE_DIR = PROJECT_ROOT / "Data" / "eval_runs"
OUT_DIR = PROJECT_ROOT / "outputs" / "pipeline_compare"

# tuned One-Euro from the GT benchmark (Scripts/tune_smoothing_accuracy.py)
ONEEURO = dict(method="oneeuro", min_cutoff=0.3, beta=0.4, bone_lock=False)


def _grid(panel):
    g = np.full((panel, panel, 3), 30, np.uint8)
    for x in range(0, panel, 40):
        cv2.line(g, (x, 0), (x, panel), (50, 50, 50), 1)
    for y in range(0, panel, 40):
        cv2.line(g, (0, y), (panel, y), (50, 50, 50), 1)
    return g


def render(clip_id: int, backbone="mediapipe_lite", panel=360):
    video_path = VIDEO_DIR / f"{clip_id}.mp4"
    p_2d = CACHE_DIR / backbone / f"{clip_id}.parquet"
    p_base = CACHE_DIR / f"motionbert_full_from_{backbone}" / f"{clip_id}.parquet"
    p_impr = CACHE_DIR / f"golfpose3d_from_{backbone}" / f"{clip_id}.parquet"
    for p in (video_path, p_2d, p_base, p_impr):
        if not p.exists():
            raise FileNotFoundError(p)

    info = video_info(video_path)
    fps = info["fps"] or 30.0
    xy_2d, conf_2d = load_2d_landmarks(backbone, clip_id)

    xyz_base = load_3d_parquet_as_h36m(p_base)                 # MotionBERT raw
    xyz_mix = load_3d_parquet_as_h36m(p_impr)                  # MixSTE raw
    xyz_impr = smooth_sequence(xyz_mix, fps=fps, **ONEEURO)    # + One-Euro

    T = min(xyz_base.shape[0], xyz_impr.shape[0], xy_2d.shape[0])
    proj_base = project_xyz_ortho(xyz_base[:T], panel, panel)
    proj_impr = project_xyz_ortho(xyz_impr[:T], panel, panel)

    xy_panel = xy_2d.copy()
    xy_panel[..., 0] *= panel / info["width"]
    xy_panel[..., 1] *= panel / info["height"]

    # jitter readout (mean 2nd-difference = acceleration)
    acc = lambda a: np.linalg.norm(np.diff(a, n=2, axis=0), axis=-1).mean()
    base_acc, impr_acc = acc(xyz_base[:T]), acc(xyz_impr[:T])
    pct = (1 - impr_acc / max(base_acc, 1e-9)) * 100

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUT_DIR / f"{clip_id}_pipeline_compare_tmp.mp4"
    final = OUT_DIR / f"{clip_id}_pipeline_compare.mp4"
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (panel * 2, panel * 2))

    cap = cv2.VideoCapture(str(video_path))
    fi = 0
    while fi < T:
        ok, frame = cap.read()
        if not ok:
            break
        raw = cv2.resize(frame, (panel, panel))
        label_panel(raw, "  STAGE 0: RAW VIDEO", f"  clip {clip_id}")

        two_d = cv2.resize(frame, (panel, panel))
        draw_2d_overlay(two_d, xy_panel[fi], conf_2d[fi])
        label_panel(two_d, "  STAGE 1: 2D POSE", f"  MediaPipe ({backbone})")

        base = _grid(panel)
        draw_3d_skeleton_ortho(base, proj_base[fi], (80, 80, 240), (80, 80, 240))
        label_panel(base, "  3D - BASELINE", f"  MotionBERT raw  accel {base_acc:.4f}",
                    title_color=(120, 120, 255))

        impr = _grid(panel)
        draw_3d_skeleton_ortho(impr, proj_impr[fi], (80, 240, 80), (80, 240, 80))
        label_panel(impr, "  3D - IMPROVED", f"  MixSTE + One-Euro  -{pct:.0f}% jitter",
                    title_color=(120, 255, 120))

        top = np.concatenate([raw, two_d], axis=1)
        bot = np.concatenate([base, impr], axis=1)
        writer.write(np.concatenate([top, bot], axis=0))
        fi += 1

    cap.release()
    writer.release()

    ffp = ffmpeg_path()
    if ffp:
        try:
            subprocess.run([ffp, "-y", "-i", str(tmp), "-vcodec", "libx264",
                            "-pix_fmt", "yuv420p", "-loglevel", "error", str(final)],
                           check=True, timeout=300)
            tmp.unlink()
        except Exception as e:
            print(f"  ffmpeg fallback: {e}")
            tmp.rename(final)
    else:
        tmp.rename(final)
    print(f"[+] {final.relative_to(PROJECT_ROOT)}  ({final.stat().st_size // 1024} KB, {fi} frames)")
    return final, {"base_accel": float(base_acc), "impr_accel": float(impr_acc),
                   "jitter_drop_pct": float(pct), "n_frames": fi}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", type=int, default=1292)
    ap.add_argument("--backbone", default="mediapipe_lite")
    a = ap.parse_args()
    render(a.clip, a.backbone)
