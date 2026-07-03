"""Batch-build coaching scorecards for N clips (torch-only, no LLM).

Stage A of the LLM-coaching eval: for a stratified sample of GolfDB clips,
run the trained detector + biomechanical indicators + reliability gating and
save the scorecard JSON. The LLM eval (stage B/C) reads these.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from event_detector import EventDetector
from coaching_indicators import EVENTS
from coaching_scorecard import build_scorecard
from smoothing import smooth_sequence

PROJECT_ROOT = Path(__file__).parent.parent
CACHE = PROJECT_ROOT / "Data" / "eval_runs" / "golfpose3d_from_mediapipe_lite"
GOLFDB = pd.read_pickle(PROJECT_ROOT / "golfdb" / "golfDB.pkl").set_index("id")
OUT = PROJECT_ROOT / "Data" / "coaching" / "eval_scorecards"


def load_xyz(cid):
    p = CACHE / f"{cid}.parquet"
    if not p.exists(): return None
    df = pd.read_parquet(p)
    T = int(df.frame.max()) + 1
    xyz = np.zeros((T, 17, 3), dtype=np.float32)
    for r in df.itertuples(index=False):
        if 0 <= r.kp_idx < 17: xyz[r.frame, r.kp_idx] = (r.x, r.y, r.z)
    return xyz


def stratified_clip_ids(n: int, seed: int = 7) -> list[int]:
    df = GOLFDB.copy()
    df = df[[(CACHE / f"{c}.parquet").exists() for c in df.index]]
    df["strat"] = df["view"].astype(str) + "_" + df["slow"].astype(str)
    # proportional sample per stratum
    rng = np.random.default_rng(seed)
    picks = []
    for _, grp in df.groupby("strat"):
        k = max(1, round(n * len(grp) / len(df)))
        picks += list(rng.choice(grp.index, size=min(k, len(grp)), replace=False))
    rng.shuffle(picks)
    return [int(c) for c in picks[:n]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    clip_ids = stratified_clip_ids(args.n)
    det = EventDetector()
    n_ok = 0
    for cid in clip_ids:
        xyz = load_xyz(cid)
        if xyz is None or xyz.shape[0] < 30:
            continue
        events = det.predict(xyz)                       # events off RAW 3D
        ev_local = np.array([events[e] for e in EVENTS])
        xyz_measure = smooth_sequence(xyz, method="oneeuro", min_cutoff=0.3,
                                      beta=0.4, bone_lock=False)  # measurements off smoothed 3D
        meta = {"clip_id": cid, "player": str(GOLFDB.loc[cid, "player"]),
                "club": str(GOLFDB.loc[cid, "club"]), "view": str(GOLFDB.loc[cid, "view"])}
        sc = build_scorecard(xyz_measure, ev_local, meta)
        (OUT / f"{cid}.json").write_text(json.dumps(sc, indent=1))
        n_ok += 1
    # save the clip-id list for the eval stage
    (OUT / "_clip_ids.json").write_text(json.dumps(clip_ids))
    print(f"[scorecards] built {n_ok} scorecards in {OUT}")


if __name__ == "__main__":
    main()
