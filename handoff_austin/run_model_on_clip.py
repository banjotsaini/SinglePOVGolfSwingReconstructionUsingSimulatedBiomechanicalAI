"""Run the best-performing pipeline (MediaPipe Lite -> MotionBERT-Full) on a
single video and write 3D landmarks to a parquet.

For Austin's z-depth work. Run from inside the main repo with the project
venv active so the adapters + weights resolve.

Usage:
    python run_model_on_clip.py path/to/swing.mp4 --out clip_3d.parquet
    python run_model_on_clip.py swing.mp4 --skeleton h36m   # native H36M-17

Output parquet columns: frame, kp_idx, kp_name, x, y, z, conf
  - default skeleton: COCO-17 (matches the consolidated handoff file)
  - --skeleton h36m: native H36M-17 (hip_center/spine/thorax/neck included)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Resolve the repo's Scripts/ dir so adapters import cleanly
REPO_SCRIPTS = Path(__file__).resolve().parent.parent / "Scripts"
sys.path.insert(0, str(REPO_SCRIPTS))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video", help="Path to an mp4 swing clip")
    p.add_argument("--out", default="clip_3d.parquet", help="Output parquet path")
    p.add_argument("--backbone", default="mediapipe_lite",
                   help="2D backbone (default mediapipe_lite)")
    p.add_argument("--skeleton", choices=["coco", "h36m"], default="coco",
                   help="Output skeleton: coco (eval schema) or h36m (native lifter output)")
    args = p.parse_args()

    from adapters import ADAPTER_REGISTRY

    video = Path(args.video)
    if not video.exists():
        sys.exit(f"video not found: {video}")

    # 1) run the 2D backbone so its cache exists (the 3D lifter reads from it)
    cache_dir = Path("_run_cache")
    cache_dir.mkdir(exist_ok=True)
    print(f"[1/2] 2D pose: {args.backbone}")
    backbone = ADAPTER_REGISTRY[args.backbone]()
    backbone.predict_and_cache(video, cache_dir, overwrite=True)

    # 2) run MotionBERT-Full on top of that 2D cache
    lifter_name = f"motionbert_full_from_{args.backbone}"
    if lifter_name not in ADAPTER_REGISTRY:
        sys.exit(f"no registered lifter '{lifter_name}' — check adapters/__init__.py")
    print(f"[2/2] 3D lift: {lifter_name}")
    lifter = ADAPTER_REGISTRY[lifter_name]()
    # point the lifter at our run cache (it reads upstream 2D from there)
    lifter.upstream_2d_cache_dir = cache_dir
    res = lifter.predict(video)

    df = res.landmarks
    if args.skeleton == "h36m":
        # convert the COCO-schema output back to native H36M-17
        import pandas as pd
        from export_ue5 import load_3d_parquet_as_h36m
        tmp = cache_dir / "_tmp_3d.parquet"
        df.to_parquet(tmp, index=False)
        xyz = load_3d_parquet_as_h36m(tmp)  # (T, 17, 3) H36M order
        from eval_utils import H36M17_NAMES
        rows = []
        for fi in range(xyz.shape[0]):
            for ji in range(17):
                rows.append({"frame": fi, "kp_idx": ji, "kp_name": H36M17_NAMES[ji],
                              "x": float(xyz[fi, ji, 0]), "y": float(xyz[fi, ji, 1]),
                              "z": float(xyz[fi, ji, 2]), "conf": 1.0})
        df = pd.DataFrame(rows)

    df.to_parquet(args.out, index=False)
    print(f"\nwrote {args.out}: {len(df):,} rows, {df['frame'].nunique()} frames, "
          f"skeleton={args.skeleton}")


if __name__ == "__main__":
    main()
