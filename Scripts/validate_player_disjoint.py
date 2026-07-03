"""Player-disjoint validation of the event detector (TASK #7).

The headline 0.865 uses GolfDB's official split, which has player overlap.
This re-trains/evaluates with a strict PLAYER-DISJOINT split — no golfer
appears in both train and test — to answer "does it work on a brand-new
golfer?" Reuses the same model + data pipeline as train_event_detector.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from train_event_detector import (
    EventTCN, load_dataset, evaluate, write_leaderboard_row,
    N_CLASSES, TOLERANCES,
)

PROJECT_ROOT = Path(__file__).parent.parent
GOLFDB_PKL = PROJECT_ROOT / "golfdb" / "golfDB.pkl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    torch.manual_seed(0); np.random.seed(0)
    print(f"[player-disjoint] device={DEVICE}")
    samples = load_dataset()

    # attach player, build a player-disjoint split (~20% of players to test)
    gdb = pd.read_pickle(GOLFDB_PKL).set_index("id")
    for s in samples:
        s["player"] = str(gdb.loc[s["clip_id"], "player"])
    players = sorted({s["player"] for s in samples})
    rng = np.random.default_rng(42)
    rng.shuffle(players)
    n_test_players = max(1, int(round(len(players) * 0.20)))
    test_players = set(players[:n_test_players])

    train = [s for s in samples if s["player"] not in test_players]
    test  = [s for s in samples if s["player"] in test_players]
    # sanity: zero player overlap
    overlap = {s["player"] for s in train} & {s["player"] for s in test}
    print(f"[player-disjoint] {len(players)} players -> {len(test_players)} held out")
    print(f"[player-disjoint] train={len(train)} clips / test={len(test)} clips")
    print(f"[player-disjoint] player overlap between sets: {len(overlap)} (must be 0)")

    # class weights
    cc = np.zeros(N_CLASSES)
    for s in train:
        for c in range(N_CLASSES):
            cc[c] += (s["target"] == c).sum()
    w = torch.tensor(np.clip(cc.sum() / (N_CLASSES * np.maximum(cc, 1)), 1.0, 200.0),
                     dtype=torch.float32, device=DEVICE)

    model = EventTCN().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(25):
        model.train(); np.random.shuffle(train); tot = 0.0
        for s in train:
            x = torch.tensor(s["feats"].T[None], device=DEVICE)
            y = torch.tensor(s["target"][None], device=DEVICE)
            loss = F.cross_entropy(model(x), y, weight=w)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        if (epoch + 1) % 5 == 0:
            pce = evaluate(model, test)
            print(f"  epoch {epoch+1:2}  loss={tot/len(train):.3f}  PCE@5={pce[5]:.3f}")

    pce = evaluate(model, test)
    print("\n" + "=" * 56)
    print(f"  PLAYER-DISJOINT TEST (n={len(test)} clips, {len(test_players)} unseen golfers)")
    print(f"    PCE@5 = {pce[5]:.3f}")
    print(f"    PCE@3 = {pce[3]:.3f}")
    print(f"    PCE@1 = {pce[1]:.3f}")
    print("=" * 56)
    print(f"  vs standard GolfDB split (player overlap): PCE@5 = 0.865")

    # persist a result row for the visual
    import json
    out = PROJECT_ROOT / "Data" / "coaching" / "player_disjoint_result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "split": "player-disjoint",
        "n_test_clips": len(test), "n_test_players": len(test_players),
        "pce_at_5": pce[5], "pce_at_3": pce[3], "pce_at_1": pce[1],
        "standard_split_pce5": 0.865,
    }, indent=1))
    print(f"\n[player-disjoint] wrote {out}")


if __name__ == "__main__":
    main()
