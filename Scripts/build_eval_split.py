"""Build a fair, PLAYER-DISJOINT dev/test split of coaching scorecards.

For the research->prototype->evaluate loop we need:
  - a DEV set to iterate prompts/KB/features against, and
  - a held-out TEST set of 200 to gate features on (>=0.8 to ship).

Fairness: no player appears in both sets (same guard as validate_player_disjoint.py),
so a feature can't "win" on test by having seen the same golfer in dev.

Scorecards are built deterministically (trained detector + indicators +
reliability gating) — no LLM here. Re-runnable; skips clips already built.
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

PROJECT_ROOT = Path(__file__).parent.parent
CACHE = PROJECT_ROOT / "Data" / "eval_runs" / "motionbert_full_from_mediapipe_lite"
GOLFDB = pd.read_pickle(PROJECT_ROOT / "golfdb" / "golfDB.pkl").set_index("id")
COACH = PROJECT_ROOT / "Data" / "coaching"
DEV_DIR = COACH / "dev_scorecards"
TEST_DIR = COACH / "test_scorecards"
MANIFEST = COACH / "eval_split.json"


def load_xyz(cid):
    p = CACHE / f"{cid}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    T = int(df.frame.max()) + 1
    xyz = np.zeros((T, 17, 3), dtype=np.float32)
    for r in df.itertuples(index=False):
        if 0 <= r.kp_idx < 17:
            xyz[r.frame, r.kp_idx] = (r.x, r.y, r.z)
    return xyz


def player_disjoint_clipsets(n_test: int, n_dev: int, seed: int = 13):
    """Allocate whole players (with all their clips) to test, then dev, so the
    two sets never share a player. Over-provision so we still hit the targets
    after clips that fail to build."""
    df = GOLFDB.copy()
    df = df[[(CACHE / f"{c}.parquet").exists() for c in df.index]]
    by_player = {p: list(g.index) for p, g in df.groupby("player")}
    players = sorted(by_player)  # deterministic
    rng = np.random.default_rng(seed)
    rng.shuffle(players)

    test, dev = [], []
    pi = 0
    while pi < len(players) and len(test) < n_test:
        test += [int(c) for c in by_player[players[pi]]]; pi += 1
    while pi < len(players) and len(dev) < n_dev:
        dev += [int(c) for c in by_player[players[pi]]]; pi += 1
    return test, dev


def build_into(clip_ids, out_dir: Path, det: EventDetector, target: int) -> list[int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    built = []
    for cid in clip_ids:
        if len(built) >= target:
            break
        dest = out_dir / f"{cid}.json"
        if dest.exists():
            built.append(cid); continue
        xyz = load_xyz(cid)
        if xyz is None or xyz.shape[0] < 30:
            continue
        try:
            events = det.predict(xyz)
            ev_local = np.array([events[e] for e in EVENTS])
            meta = {"clip_id": cid, "player": str(GOLFDB.loc[cid, "player"]),
                    "club": str(GOLFDB.loc[cid, "club"]), "view": str(GOLFDB.loc[cid, "view"])}
            sc = build_scorecard(xyz, ev_local, meta)
            dest.write_text(json.dumps(sc, indent=1))
            built.append(cid)
        except Exception as e:
            print(f"  clip {cid} failed: {repr(e)[:100]}")
        if len(built) % 25 == 0:
            print(f"  {out_dir.name}: {len(built)}/{target}", flush=True)
    return built


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-test", type=int, default=200)
    p.add_argument("--n-dev", type=int, default=120)
    args = p.parse_args()

    # over-provision candidate pools (~1.3x) to survive build failures
    test_pool, dev_pool = player_disjoint_clipsets(int(args.n_test * 1.3), int(args.n_dev * 1.3))
    det = EventDetector()
    print(f"[split] building TEST (target {args.n_test}) ...")
    test = build_into(test_pool, TEST_DIR, det, args.n_test)
    print(f"[split] building DEV (target {args.n_dev}) ...")
    dev = build_into(dev_pool, DEV_DIR, det, args.n_dev)

    # sanity: player-disjoint
    tp = {GOLFDB.loc[c, "player"] for c in test}
    dp = {GOLFDB.loc[c, "player"] for c in dev}
    overlap = tp & dp
    manifest = {
        "seed": 13,
        "n_test": len(test), "n_dev": len(dev),
        "player_disjoint": len(overlap) == 0,
        "player_overlap": sorted(overlap),
        "test_clip_ids": test, "dev_clip_ids": dev,
        "test_players": sorted(tp), "dev_players": sorted(dp),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=1))
    print(f"\n[split] TEST={len(test)} DEV={len(dev)} player_disjoint={len(overlap)==0}")
    print(f"[split] wrote {MANIFEST}")


if __name__ == "__main__":
    main()
