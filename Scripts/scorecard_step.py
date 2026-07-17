"""Scorecard step — runs the trained event detector + coaching scorecard on a
cached 3D parquet. Kept as a standalone step (torch only, no mediapipe) so it
can run as its own subprocess, mirroring the benchmark's process isolation.

Usage:
    python scorecard_step.py --parquet <3d.parquet> --out-dir <dir> --stem <name>
      [--meta-json <json with player/club/view>]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))


def load_coco_xyz(parquet_3d: Path) -> np.ndarray:
    df = pd.read_parquet(parquet_3d)
    T = int(df["frame"].max()) + 1
    xyz = np.zeros((T, 17, 3), dtype=np.float32)
    for r in df.itertuples(index=False):
        if 0 <= r.kp_idx < 17:
            xyz[r.frame, r.kp_idx] = (r.x, r.y, getattr(r, "z", 0.0))
    return xyz


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--stem", required=True)
    p.add_argument("--meta-json", default=None)
    p.add_argument("--fps", type=float, default=None,
                   help="real-time capture fps of the source video; enables the "
                        "time-based hand-speed indicator (omit for slow-motion)")
    p.add_argument("--no-smooth", action="store_true",
                   help="disable One-Euro smoothing on the coaching-measurement branch")
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    xyz = load_coco_xyz(Path(args.parquet))

    # 1) detect the 8 events with the trained CNN — on RAW 3D. Events are motion
    #    reversals, so the detector needs the sharp velocity signal (smoothing it
    #    would blunt exactly the frames it keys on).
    from event_detector import EventDetector
    det = EventDetector()
    events = det.predict(xyz)                       # {name: frame}
    from coaching_indicators import EVENTS
    events_local = np.array([events[e] for e in EVENTS])
    print(f"[scorecard] detected events: {events}")

    # 2) coaching MEASUREMENTS come off One-Euro-smoothed 3D. The GT (Vicon)
    #    benchmark showed this cuts angular measurement error 33-46% under
    #    detection noise (Scripts/tune_smoothing_accuracy.py). Tuned params.
    xyz_measure = xyz
    if not args.no_smooth:
        from smoothing import smooth_sequence
        xyz_measure = smooth_sequence(xyz, method="oneeuro", min_cutoff=0.3,
                                       beta=0.4, bone_lock=False)

    # 3) build + render the coaching scorecard (indicators at events, on smoothed 3D)
    meta = {}
    if args.meta_json and Path(args.meta_json).exists():
        meta = json.loads(Path(args.meta_json).read_text())
    meta.setdefault("player", args.stem)
    fps = args.fps if args.fps else meta.get("fps")

    from coaching_scorecard import build_scorecard
    from render_scorecard import render
    sc = build_scorecard(xyz_measure, events_local, meta, fps=fps)

    json_path = out_dir / f"{args.stem}_scorecard.json"
    png_path  = out_dir / f"{args.stem}_scorecard.png"
    json_path.write_text(json.dumps(sc, indent=2))
    render(sc, str(png_path))

    print(f"[scorecard] summary: {sc['summary']}")
    print(f"[scorecard] wrote {json_path.name} + {png_path.name}")


if __name__ == "__main__":
    main()
