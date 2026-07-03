"""Compute coaching indicators across all GolfDB clips -> pro reference bands.

TASK #5: the "good swing" reference. Every GolfDB clip is a professional
swing, so the distribution of each indicator across the corpus is our
data-backed normal range. A user's value is later interpreted relative to
this band (percentile), never against an invented absolute threshold.

Outputs:
  Data/coaching/indicators_all.parquet   per-clip indicator values + metadata
  Data/coaching/reference_bands.json      per-indicator p10/p25/p50/p75/p90
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from coaching_indicators import compute_indicators, INDICATOR_NAMES, EVENTS

PROJECT_ROOT = Path(__file__).parent.parent
CACHE_3D = PROJECT_ROOT / "Data" / "eval_runs" / "motionbert_full_from_mediapipe_lite"
GOLFDB_PKL = PROJECT_ROOT / "golfdb" / "golfDB.pkl"
OUT_DIR = PROJECT_ROOT / "Data" / "coaching"


def load_clip_xyz(clip_id: int):
    p = CACHE_3D / f"{clip_id}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    T = int(df["frame"].max()) + 1
    xyz = np.zeros((T, 17, 3), dtype=np.float32)
    for r in df.itertuples(index=False):
        if 0 <= r.kp_idx < 17:
            xyz[r.frame, r.kp_idx] = (r.x, r.y, r.z)
    return xyz


def events_clip_local(ev):
    ev = np.asarray(ev)
    return ev[1:9] - ev[0]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_pickle(GOLFDB_PKL).set_index("id")

    rows = []
    for cid in df.index:
        xyz = load_clip_xyz(int(cid))
        if xyz is None or xyz.shape[0] < 30:
            continue
        ev = events_clip_local(df.loc[cid, "events"])
        if (ev < 0).any() or (ev >= xyz.shape[0]).any():
            continue
        try:
            ind = compute_indicators(xyz, ev)
        except Exception as e:
            continue
        ind["clip_id"] = int(cid)
        ind["view"] = df.loc[cid, "view"]
        ind["club"] = df.loc[cid, "club"]
        ind["slow"] = int(df.loc[cid, "slow"])
        rows.append(ind)

    out = pd.DataFrame(rows)
    out.to_parquet(OUT_DIR / "indicators_all.parquet", index=False)
    print(f"[ref] computed indicators for {len(out)} clips")

    # Reference bands: robust percentiles per indicator (whole corpus)
    bands = {}
    for name in INDICATOR_NAMES:
        vals = out[name].replace([np.inf, -np.inf], np.nan).dropna()
        bands[name] = {
            "p10": float(vals.quantile(0.10)),
            "p25": float(vals.quantile(0.25)),
            "p50": float(vals.quantile(0.50)),
            "p75": float(vals.quantile(0.75)),
            "p90": float(vals.quantile(0.90)),
            "mean": float(vals.mean()),
            "std": float(vals.std()),
        }
    with open(OUT_DIR / "reference_bands.json", "w") as f:
        json.dump(bands, f, indent=1)
    print(f"[ref] wrote reference_bands.json")
    print()
    print("=== Pro reference (median [p25-p75]) — sanity check ===")
    for name in INDICATOR_NAMES:
        b = bands[name]
        unit = "deg" if name.endswith("_deg") else ("%" if name.endswith("_pct") else "")
        print(f"  {name:30} {b['p50']:7.1f} {unit:3}  [{b['p25']:6.1f} - {b['p75']:6.1f}]")


if __name__ == "__main__":
    main()
