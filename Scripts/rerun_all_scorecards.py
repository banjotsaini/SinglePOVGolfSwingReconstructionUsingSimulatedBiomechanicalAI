"""Regenerate every coaching scorecard with the improved pipeline.

Improved pipeline (this is now the production path):
  - 3D lifter  = GolfPose MixSTE (golfpose3d_from_mediapipe_lite)
  - events     = trained detector on RAW MixSTE 3D
  - measures   = coaching indicators on One-Euro-smoothed MixSTE 3D

Rewrites the JSONs in place for every clip present in the dev/test/eval
scorecard sets (and the demo clips), so downstream LLM-coaching evals read the
improved numbers. Keeps a one-time backup of the previous JSONs.
"""
from __future__ import annotations
import sys, json, shutil, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))
from event_detector import EventDetector
from coaching_indicators import EVENTS
from coaching_scorecard import build_scorecard
from smoothing import smooth_sequence

CACHE = ROOT / "Data" / "eval_runs" / "golfpose3d_from_mediapipe_lite"
GOLFDB = pd.read_pickle(ROOT / "golfdb" / "golfDB.pkl").set_index("id")
SETS = ["dev_scorecards", "test_scorecards", "eval_scorecards"]


def load_xyz(cid):
    p = CACHE / f"{cid}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    T = int(df["frame"].max()) + 1
    xyz = np.zeros((T, 17, 3), np.float32)
    for r in df.itertuples(index=False):
        if 0 <= r.kp_idx < 17:
            xyz[r.frame, r.kp_idx] = (r.x, r.y, getattr(r, "z", 0.0))
    return xyz


def meta_for(cid):
    if cid in GOLFDB.index:
        row = GOLFDB.loc[cid]
        return {"clip_id": int(cid), "player": str(row.get("player", cid)),
                "club": str(row.get("club", "")), "view": str(row.get("view", ""))}
    return {"clip_id": int(cid), "player": str(cid)}


def regen_one(cid, det):
    xyz = load_xyz(cid)
    if xyz is None or xyz.shape[0] < 30:
        return None
    events = det.predict(xyz)                                  # raw 3D
    ev_local = np.array([events[e] for e in EVENTS])
    xyz_m = smooth_sequence(xyz, method="oneeuro", min_cutoff=0.3, beta=0.4, bone_lock=False)
    return build_scorecard(xyz_m, ev_local, meta_for(cid))


def clip_id_from(name: str):
    stem = Path(name).stem
    for suf in ("_scorecard", ""):
        s = stem.replace(suf, "")
        if s.isdigit():
            return int(s)
    return int(stem) if stem.isdigit() else None


def main():
    det = EventDetector()
    total = 0
    for setname in SETS:
        d = ROOT / "Data" / "coaching" / setname
        if not d.exists():
            continue
        bak = d.parent / f"{setname}_motionbert_bak"
        if not bak.exists():
            shutil.copytree(d, bak)  # one-time backup of previous (MotionBERT) scorecards
        jsons = sorted(d.glob("*.json"))
        n = 0
        for jf in jsons:
            cid = clip_id_from(jf.name)
            if cid is None:
                continue
            sc = regen_one(cid, det)
            if sc is None:
                continue
            jf.write_text(json.dumps(sc, indent=1))
            n += 1
        print(f"[{setname}] regenerated {n}/{len(jsons)} scorecards (backup: {bak.name})")
        total += n
    print(f"[done] {total} scorecards regenerated with improved pipeline")


if __name__ == "__main__":
    main()
