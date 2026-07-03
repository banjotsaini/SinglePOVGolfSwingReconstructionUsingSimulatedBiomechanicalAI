"""Torch-only step: run the trained detector on a set of clips and save
predicted vs ground-truth event frames to JSON (for the accuracy visual).
Kept separate so torch never co-runs with matplotlib/mediapipe."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from event_detector import EventDetector
from coaching_indicators import EVENTS

PROJECT_ROOT = Path(__file__).parent.parent
CACHE = PROJECT_ROOT / "Data" / "eval_runs" / "motionbert_full_from_mediapipe_lite"
GOLFDB = pd.read_pickle(PROJECT_ROOT / "golfdb" / "golfDB.pkl").set_index("id")
OUT = PROJECT_ROOT / "Data" / "coaching" / "event_predictions.json"

CLIPS = [1292, 234, 916, 0, 7, 459]   # variety of views/players

def load_xyz(cid):
    df = pd.read_parquet(CACHE / f"{cid}.parquet")
    T = int(df.frame.max()) + 1
    xyz = np.zeros((T, 17, 3), dtype=np.float32)
    for r in df.itertuples(index=False):
        if 0 <= r.kp_idx < 17: xyz[r.frame, r.kp_idx] = (r.x, r.y, r.z)
    return xyz

det = EventDetector()
rows = []
for cid in CLIPS:
    if cid not in GOLFDB.index or not (CACHE / f"{cid}.parquet").exists():
        continue
    xyz = load_xyz(cid)
    pred = det.predict(xyz)
    ev = np.asarray(GOLFDB.loc[cid, "events"]); gt = (ev[1:9] - ev[0]).tolist()
    rows.append({
        "clip_id": int(cid), "player": str(GOLFDB.loc[cid, "player"]),
        "view": str(GOLFDB.loc[cid, "view"]), "n_frames": int(xyz.shape[0]),
        "predicted": [pred[e] for e in EVENTS],
        "ground_truth": [int(g) for g in gt],
    })
OUT.write_text(json.dumps(rows, indent=1))
print(f"wrote {OUT}: {len(rows)} clips")
