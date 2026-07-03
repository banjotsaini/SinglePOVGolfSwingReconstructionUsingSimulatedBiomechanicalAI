"""Build a gold-labeled Q&A set over a split's scorecards.

Each item = {id, clip, question, gold, target}. Gold categories:
  answer            -> answerable from a med/high-confidence metric (target=that key)
  refuse_lowconf    -> targets a low-confidence metric (arm bend) -> must refuse
  refuse_unmeasured -> topic we don't measure (grip, ball flight, ...) -> must refuse
  refuse_scope      -> asks what to fix/practice -> must refuse (MVP scope)

Sampled to --n with fixed proportions (answer .5, unmeasured .25, lowconf .15, scope .10)
so the held-out gate stresses correct refusal as much as correct answering.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
COACH = PROJECT_ROOT / "Data" / "coaching"
SPLIT_DIRS = {"dev": COACH / "dev_scorecards", "test": COACH / "test_scorecards"}
import sys
sys.path.insert(0, str(Path(__file__).parent))
import coaching_qa as QA


def build_pools(split):
    pools = {"answer": [], "refuse_lowconf": [], "refuse_unmeasured": [], "refuse_scope": []}
    for q in sorted(SPLIT_DIRS[split].glob("*.json")):
        if not q.stem.isdigit():
            continue
        sc = json.loads(q.read_text(encoding="utf-8"))
        cid = int(q.stem)
        inds = sc["indicators"]
        for key, qs in QA.ANSWERABLE.items():
            if key in inds and inds[key].get("confidence_tier") != "low":
                for qi, text in enumerate(qs):
                    pools["answer"].append({"id": f"{cid}_ans_{key}_{qi}", "clip": cid,
                                            "question": text, "gold": "answer", "target": key})
        for key, qs in QA.LOWCONF.items():
            if key in inds:
                for qi, text in enumerate(qs):
                    pools["refuse_lowconf"].append({"id": f"{cid}_lc_{key}_{qi}", "clip": cid,
                                                    "question": text, "gold": "refuse_lowconf", "target": key})
        for qi, text in enumerate(QA.UNMEASURED):
            pools["refuse_unmeasured"].append({"id": f"{cid}_um_{qi}", "clip": cid,
                                               "question": text, "gold": "refuse_unmeasured", "target": ""})
        for qi, text in enumerate(QA.SCOPE):
            pools["refuse_scope"].append({"id": f"{cid}_sc_{qi}", "clip": cid,
                                          "question": text, "gold": "refuse_scope", "target": ""})
    return pools


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["dev", "test"], required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--seed", type=int, default=31)
    args = p.parse_args()

    pools = build_pools(args.split)
    rng = np.random.default_rng(args.seed)
    quota = {"answer": round(0.50 * args.n), "refuse_unmeasured": round(0.25 * args.n),
             "refuse_lowconf": round(0.15 * args.n)}
    quota["refuse_scope"] = args.n - sum(quota.values())
    items = []
    for cat, k in quota.items():
        pool = pools[cat]
        idx = rng.permutation(len(pool))[:min(k, len(pool))]
        items += [pool[i] for i in idx]
    rng.shuffle(items)
    items = items[: args.n]

    out = COACH / f"qa_set_{args.split}.json"
    out.write_text(json.dumps(items, indent=1), encoding="utf-8")
    from collections import Counter
    print(f"[qa] {args.split}: {len(items)} items -> {out} | mix={dict(Counter(i['gold'] for i in items))}")


if __name__ == "__main__":
    main()
