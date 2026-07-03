"""Build same-player swing PAIRS for the progress-tracking eval.

For each split, group scorecards by player and sample up to `--cap` pairs per
player (deterministic), targeting `--n` pairs total. Each pair (A=lower clip id,
B=higher) is treated as earlier->later for delta-narration testing. Player-disjoint
property is inherited from the underlying split.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
COACH = PROJECT_ROOT / "Data" / "coaching"
SPLIT_DIRS = {"dev": COACH / "dev_scorecards", "test": COACH / "test_scorecards"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["dev", "test"], required=True)
    p.add_argument("--n", type=int, required=True, help="target number of pairs")
    p.add_argument("--cap", type=int, default=6, help="max pairs per player")
    p.add_argument("--seed", type=int, default=23)
    args = p.parse_args()

    byp = defaultdict(list)
    for q in SPLIT_DIRS[args.split].glob("*.json"):
        if not q.stem.isdigit():
            continue
        sc = json.loads(q.read_text(encoding="utf-8"))
        byp[sc["meta"]["player"]].append(int(sc["meta"]["clip_id"]))

    rng = np.random.default_rng(args.seed)
    players = sorted(k for k, v in byp.items() if len(v) >= 2)
    rng.shuffle(players)
    pairs = []
    for pl in players:
        clips = sorted(byp[pl])
        allp = [(a, b) for a, b in combinations(clips, 2)]
        rng.shuffle(allp)
        for a, b in allp[: args.cap]:
            pairs.append({"a": int(a), "b": int(b), "player": pl})
    rng.shuffle(pairs)
    pairs = pairs[: args.n]

    out = COACH / f"progress_pairs_{args.split}.json"
    out.write_text(json.dumps(pairs, indent=1), encoding="utf-8")
    print(f"[pairs] {args.split}: {len(pairs)} pairs from {len(set(p['player'] for p in pairs))} players -> {out}")


if __name__ == "__main__":
    main()
