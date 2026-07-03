"""Generate baseline vs improved coaching scorecards for the comparison panel.

baseline  = MotionBERT-Full 3D (current app), no smoothing
improved  = GolfPose MixSTE 3D + One-Euro smoothing (the measured-better config)

Both run through the SAME scorecard + event detector, so any difference is purely
the 3D quality. Output: outputs/pipeline_compare/scorecards/.
"""
from __future__ import annotations
import sys, json, subprocess, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
CACHE = ROOT / "Data" / "eval_runs"
OUT = ROOT / "outputs" / "pipeline_compare" / "scorecards"
OUT.mkdir(parents=True, exist_ok=True)
PY = sys.executable
sys.path.insert(0, str(Path(__file__).parent))
from smoothing import smooth_sequence

ONEEURO = dict(method="oneeuro", min_cutoff=0.3, beta=0.4, bone_lock=False)


def write_smoothed_parquet(src_parquet: Path, dst: Path):
    df = pd.read_parquet(src_parquet)
    T = int(df["frame"].max()) + 1
    xyz = np.zeros((T, 17, 3), np.float32)
    for r in df.itertuples(index=False):
        if 0 <= r.kp_idx < 17:
            xyz[r.frame, r.kp_idx] = (r.x, r.y, getattr(r, "z", 0.0))
    sm = smooth_sequence(xyz, fps=30.0, **ONEEURO)  # One-Euro is per-coordinate; skeleton-agnostic
    df2 = df.copy()
    for r in df2.itertuples():
        if 0 <= r.kp_idx < 17:
            df2.at[r.Index, "x"] = sm[r.frame, r.kp_idx, 0]
            df2.at[r.Index, "y"] = sm[r.frame, r.kp_idx, 1]
            df2.at[r.Index, "z"] = sm[r.frame, r.kp_idx, 2]
    df2.to_parquet(dst)


def meta_for(clip_id: int) -> Path:
    pkl = ROOT / "golfdb" / "golfDB.pkl"
    meta = {"player": str(clip_id)}
    if pkl.exists():
        db = pd.read_pickle(pkl).set_index("id")
        if clip_id in db.index:
            row = db.loc[clip_id]
            meta = {"player": str(row.get("player", clip_id)),
                    "club": str(row.get("club", "")), "view": str(row.get("view", ""))}
    p = OUT / f"{clip_id}_meta.json"
    p.write_text(json.dumps(meta))
    return p


def run_scorecard(parquet: Path, stem: str, meta: Path):
    subprocess.run([PY, str(Path(__file__).parent / "scorecard_step.py"),
                    "--parquet", str(parquet), "--out-dir", str(OUT),
                    "--stem", stem, "--meta-json", str(meta)],
                   check=True, cwd=str(Path(__file__).parent))


def main(clip_id=1292, backbone="mediapipe_lite"):
    meta = meta_for(clip_id)
    base_pq = CACHE / f"motionbert_full_from_{backbone}" / f"{clip_id}.parquet"
    mix_pq = CACHE / f"golfpose3d_from_{backbone}" / f"{clip_id}.parquet"
    smoothed = OUT / f"{clip_id}_improved_3d.parquet"
    write_smoothed_parquet(mix_pq, smoothed)

    print("== BASELINE (MotionBERT, raw) ==")
    run_scorecard(base_pq, f"{clip_id}_baseline", meta)
    print("== IMPROVED (MixSTE + One-Euro) ==")
    run_scorecard(smoothed, f"{clip_id}_improved", meta)
    print(f"[+] scorecards in {OUT}")


if __name__ == "__main__":
    cid = int(sys.argv[1]) if len(sys.argv) > 1 else 1292
    main(cid)
