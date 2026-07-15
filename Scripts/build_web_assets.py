"""Build real deploy/web/assets/<clip>/ bundles from pipeline outputs.

Replaces Austin's hand-made placeholder assets (Phase 2 of the deployment plan):
for a GolfDB clip with cached pose + a scorecard, emit the five files the front
end consumes — raw.mp4, overlay.mp4 (H.264/yuv420p, the codec Chrome needs),
metrics.json, explanation.json, replay_3d.json — in exactly the shapes app.js /
replay3d.js already read, so no UI changes are needed.

Usage:
  python build_web_assets.py 417 886          # needs Data/demo/clip<id>/ scorecard
Inputs per clip:
  Data/videos_160/<id>.mp4                                 raw video
  Data/eval_runs/mediapipe_lite/<id>.parquet               cached 2D pose
  Data/eval_runs/golfpose3d_from_mediapipe_lite/<id>.parquet  cached 3D pose
  Data/demo/clip<id>/clip<id>_scorecard.json               scorecard (demo.py fast path)
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# the JSON builders live in web_artifacts.py, SHARED with the real upload path
# (pipeline.py + the processing Lambda) so demo and upload assets can't drift
from web_artifacts import (load_kb, build_replay_json, build_metrics_json,
                           build_explanation_json)

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "deploy" / "web" / "assets"
KB = load_kb()


def ffmpeg() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def h264(src: Path, dst: Path) -> None:
    """Re-encode to H.264/yuv420p — the only combo Chrome reliably plays."""
    subprocess.run([ffmpeg(), "-y", "-i", str(src), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an",
                    str(dst)], check=True, capture_output=True)


def build_overlay(clip: int, out: Path, fps: float = 30.0) -> None:
    sys.path.insert(0, str(ROOT / "Scripts"))
    from pipeline import write_2d_overlay
    from smoothing import one_euro_filter
    from pose_diagnostics import load_2d, repair_lr_swaps
    video = ROOT / "Data" / "videos_160" / f"{clip}.mp4"
    pq2d = ROOT / "Data" / "eval_runs" / "mediapipe_lite" / f"{clip}.parquet"
    # The cache holds RAW landmarks; un-swap L/R identities then smooth the
    # drawn skeleton so the overlay neither crosses limbs nor jitters (same
    # repair + tuned One-Euro as the replay/coaching paths).
    df = pd.read_parquet(pq2d).sort_values(["frame", "kp_idx"]).reset_index(drop=True)
    T, K = df["frame"].nunique(), df["kp_idx"].nunique()
    xy_r, conf_r = load_2d(pq2d)
    xy_r, _, _ = repair_lr_swaps(xy_r, conf_r)
    df[["x", "y"]] = one_euro_filter(xy_r[:T, :K], fps=fps,
                                     min_cutoff=0.3, beta=0.4).reshape(T * K, 2)
    with tempfile.TemporaryDirectory() as td:
        sm_pq = Path(td) / "smoothed_2d.parquet"
        df.to_parquet(sm_pq, index=False)
        tmp = Path(td) / "overlay_raw.mp4"
        write_2d_overlay(video, sm_pq, tmp)         # writes mp4v — not browser-safe
        h264(tmp, out)


def build_replay(clip: int, out: Path, fps: float) -> None:
    # prefer the lift from L/R-repaired landmarks (pipeline.py writes it to the
    # sibling _lrfix cache) — fall back to the raw-2D lift
    pq = ROOT / "Data" / "eval_runs" / "golfpose3d_from_mediapipe_lite_lrfix" / f"{clip}.parquet"
    if not pq.exists():
        pq = ROOT / "Data" / "eval_runs" / "golfpose3d_from_mediapipe_lite" / f"{clip}.parquet"
    build_replay_json(pq, out, fps)


def build_metrics(clip: int, sc: dict, out: Path) -> None:
    build_metrics_json(sc, out, str(clip), KB)


def build_explanation(clip: int, sc: dict, out: Path) -> None:
    build_explanation_json(sc, out, str(clip), KB)


def build(clip: int) -> None:
    scp = ROOT / "Data" / "demo" / f"clip{clip}" / f"clip{clip}_scorecard.json"
    sc = json.loads(scp.read_text(encoding="utf-8"))
    dst = WEB / str(clip)
    dst.mkdir(parents=True, exist_ok=True)
    video = ROOT / "Data" / "videos_160" / f"{clip}.mp4"
    cap = cv2.VideoCapture(str(video)); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0; cap.release()

    h264(video, dst / "raw.mp4")
    print(f"[{clip}] raw.mp4")
    build_overlay(clip, dst / "overlay.mp4", fps)
    print(f"[{clip}] overlay.mp4 (H.264)")
    build_replay(clip, dst / "replay_3d.json", fps)
    print(f"[{clip}] replay_3d.json")
    build_metrics(clip, sc, dst / "metrics.json")
    print(f"[{clip}] metrics.json")
    build_explanation(clip, sc, dst / "explanation.json")
    print(f"[{clip}] explanation.json")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        build(int(arg))
    print("done")
