"""Eval harness for the PROGRESS-TRACKING narration (round 3 of the loop).

Scores delta narrations over same-player swing pairs with a composite in [0,1]:
    0.45 * grounded_rate        (no wrong-direction / invented-trend / leakage in delta claims)
  + 0.25 * coverage_mean        (mentions each meaningfully-changed metric)
  + 0.15 * nonprescriptive_rate (no fix-it advice)
  + 0.15 * readable_rate        (Flesch >= 60)

Same fairness model as the single-clip harness: iterate on dev pairs, gate on the
held-out 200 test pairs. Features are JSON configs (rules_append etc.); generations
cached by config hash so prompt edits auto-invalidate.

Usage:
  python coaching_progress_eval.py --feature progress_v1 --split dev --workers 4
  python coaching_progress_eval.py --feature progress_v1 --split test    # 200-pair gate
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
COACH = PROJECT_ROOT / "Data" / "coaching"
FEATURES_DIR = COACH / "features"
GEN_DIR = COACH / "eval_gen"
REPORTS_DIR = COACH / "eval_reports"
SPLIT_DIRS = {"dev": COACH / "dev_scorecards", "test": COACH / "test_scorecards"}

sys.path.insert(0, str(Path(__file__).parent))
import coaching_progress as P
import coaching_llm_summary_v2 as v2

WEIGHTS = {"grounded": 0.45, "coverage": 0.25, "nonprescriptive": 0.15, "readable": 0.15}
PRESCRIPTIVE = [
    r"\byou should\b", r"\btry to\b", r"\btry \b", r"\bwork on\b", r"\bfocus on\b",
    r"\bkeep your\b", r"\bmake sure\b", r"\bto fix\b", r"\bto improve\b", r"\bpractice\b",
    r"\bdrill\b", r"\byou need to\b", r"\baim to\b", r"\bensure\b", r"\bremember to\b",
]
DEFAULT_FEATURE = {
    "id": "progress_v1",
    "description": "Progress narration baseline: Codex + KB + structured delta grounding.",
    "backend": "codex", "model": None, "rules_append": None, "rules_override": None,
}


def _syllables(w):
    w = w.lower(); v = "aeiouy"; n = 0; prev = False
    for ch in w:
        cur = ch in v
        if cur and not prev:
            n += 1
        prev = cur
    if w.endswith("e") and n > 1:
        n -= 1
    return max(1, n)


def flesch(text):
    sents = max(1, len(re.findall(r"[.!?]+", text)))
    words = re.findall(r"[a-zA-Z]+", text)
    if not words:
        return 0.0
    syl = sum(_syllables(w) for w in words)
    return 206.835 - 1.015 * (len(words) / sents) - 84.6 * (syl / len(words))


def load_feature(fid):
    if fid == "progress_v1":
        return dict(DEFAULT_FEATURE)
    path = FEATURES_DIR / f"{fid}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    cfg = json.loads(path.read_text(encoding="utf-8"))
    cfg.setdefault("id", fid)
    for k, val in DEFAULT_FEATURE.items():
        cfg.setdefault(k, val)
    return cfg


def rules_for(cfg):
    rules = cfg.get("rules_override") or P.RULES
    if cfg.get("rules_append"):
        rules = rules + "\n" + cfg["rules_append"]
    return rules


def config_hash(cfg, kb):
    sig = json.dumps({"rules": rules_for(cfg), "backend": cfg.get("backend"),
                      "model": cfg.get("model"), "kb_version": kb.get("_version")}, sort_keys=True)
    return hashlib.sha1(sig.encode("utf-8")).hexdigest()[:10]


def gen_cached(cfg, split, pair, scA, scB, kb):
    cdir = GEN_DIR / cfg["id"] / f"progress_{split}" / config_hash(cfg, kb)
    cdir.mkdir(parents=True, exist_ok=True)
    cache = cdir / f"{pair['a']}_{pair['b']}.json"
    if cache.exists():
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except Exception:
            pass
    try:
        out = P.generate(scA, scB, kb, rules=rules_for(cfg),
                         backend=cfg.get("backend", "codex"), model=cfg.get("model"))
    except Exception as e:
        print(f"  pair {pair['a']}_{pair['b']}: gen failed: {repr(e)[:120]}")
        return None
    cache.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def score_pair(scA, scB, out):
    g = P.verify_progress(scA, scB, out.get("claims", []))
    text = out.get("explanation", "")
    presc = [p for p in PRESCRIPTIVE if re.search(p, text.lower())]
    return {"grounded": g["grounded"], "violations": g["violations"], "coverage": g["coverage"],
            "n_meaningful": g["n_meaningful"], "prescriptive_count": len(presc),
            "flesch": round(flesch(text), 1)}


def composite(rows):
    n = len(rows)
    grounded = sum(r["grounded"] for r in rows) / n
    covs = [r["coverage"] for r in rows if r["coverage"] is not None]
    coverage = (sum(covs) / len(covs)) if covs else 1.0
    nonpresc = sum(r["prescriptive_count"] == 0 for r in rows) / n
    readable = sum(r["flesch"] >= 60 for r in rows) / n
    comp = (WEIGHTS["grounded"] * grounded + WEIGHTS["coverage"] * coverage
            + WEIGHTS["nonprescriptive"] * nonpresc + WEIGHTS["readable"] * readable)
    return {"composite": round(comp, 4), "grounded_rate": round(grounded, 4),
            "coverage_mean": round(coverage, 4), "nonprescriptive_rate": round(nonpresc, 4),
            "readable_rate": round(readable, 4), "n_pairs_with_change": len(covs)}


def run(fid, split, n, workers):
    cfg = load_feature(fid)
    kb = v2.load_kb()
    pairs = json.loads((COACH / f"progress_pairs_{split}.json").read_text(encoding="utf-8"))
    if n:
        pairs = pairs[:n]
    if not pairs:
        raise SystemExit(f"no pairs — run build_progress_pairs.py --split {split} first")
    print(f"[progress-eval] feature={cfg['id']} split={split} pairs={len(pairs)} workers={workers}")
    scs = {}

    def _sc(cid, split):
        if cid not in scs:
            scs[cid] = json.loads((SPLIT_DIRS[split] / f"{cid}.json").read_text(encoding="utf-8"))
        return scs[cid]

    def _one(pair):
        scA, scB = _sc(pair["a"], split), _sc(pair["b"], split)
        out = gen_cached(cfg, split, pair, scA, scB, kb)
        if out is None:
            return None
        row = score_pair(scA, scB, out)
        row["pair"] = f"{pair['a']}_{pair['b']}"
        return row

    rows, done = [], 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_one, pr): pr for pr in pairs}
        for fut in as_completed(futs):
            done += 1
            r = fut.result()
            if r is not None:
                rows.append(r)
            if done % 20 == 0:
                print(f"  {done}/{len(pairs)}", flush=True)

    agg = composite(rows)
    report = {"feature_id": cfg["id"], "task": "progress", "split": split, "weights": WEIGHTS,
              "n_evaluated": len(rows), "n_requested": len(pairs), "scores": agg,
              "passes_0.8": agg["composite"] >= 0.8, "config": cfg, "per_pair": rows}
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / f"{cfg['id']}__progress_{split}.json"
    out_path.write_text(json.dumps(report, indent=1), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"  {cfg['id']}  [progress/{split}]  n={len(rows)}")
    print("=" * 60)
    for k in ("grounded_rate", "coverage_mean", "nonprescriptive_rate", "readable_rate"):
        print(f"  {k:22s}: {agg[k]}")
    print(f"  {'COMPOSITE':22s}: {agg['composite']}   {'PASS (>=0.8)' if report['passes_0.8'] else 'below 0.8'}")
    print(f"\n[progress-eval] wrote {out_path}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--feature", default="progress_v1")
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    run(a.feature, a.split, a.n, a.workers)
