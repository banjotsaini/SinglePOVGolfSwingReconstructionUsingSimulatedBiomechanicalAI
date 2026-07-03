"""Eval harness for swing Q&A (round 4) — scores REASONING, not just narration.

Composite ∈ [0,1]:
    0.40 * decision_accuracy   (correctly answer answerable Qs, refuse the rest)
  + 0.30 * answer_faithfulness (answered the right metric with the right in/out-of-range call)
  + 0.15 * clean_rate          (no low-confidence leak, no prescriptive advice)
  + 0.15 * readable_rate       (Flesch >= 60)

decision_accuracy is the hard new signal: the model must refuse unmeasured / low-confidence /
fix-it questions. Iterate on dev, gate on the held-out 200.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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
import coaching_qa as QA
import coaching_llm_summary_v2 as v2

WEIGHTS = {"decision": 0.40, "faithful": 0.30, "clean": 0.15, "readable": 0.15}
DEFAULT_FEATURE = {"id": "qa_v1", "description": "Swing Q&A baseline: Codex + KB + refusal reasoning.",
                   "backend": "codex", "model": None, "rules_append": None, "rules_override": None}


def load_feature(fid):
    if fid == "qa_v1":
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
    rules = cfg.get("rules_override") or QA.RULES
    if cfg.get("rules_append"):
        rules = rules + "\n" + cfg["rules_append"]
    return rules


def config_hash(cfg, kb):
    sig = json.dumps({"rules": rules_for(cfg), "backend": cfg.get("backend"),
                      "model": cfg.get("model"), "kb_version": kb.get("_version")}, sort_keys=True)
    return hashlib.sha1(sig.encode("utf-8")).hexdigest()[:10]


def gen_cached(cfg, split, item, sc, kb):
    cdir = GEN_DIR / cfg["id"] / f"qa_{split}" / config_hash(cfg, kb)
    cdir.mkdir(parents=True, exist_ok=True)
    cache = cdir / f"{item['id']}.json"
    if cache.exists():
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except Exception:
            pass
    try:
        out = QA.generate(sc, item["question"], kb, rules=rules_for(cfg),
                          backend=cfg.get("backend", "codex"), model=cfg.get("model"))
    except Exception as e:
        print(f"  item {item['id']}: gen failed: {repr(e)[:120]}")
        return None
    cache.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def composite(rows):
    n = len(rows)
    decision = sum(r["decision_correct"] for r in rows) / n
    ans_rows = [r for r in rows if r["gold"] == "answer"]
    faithful = (sum(r["decision_correct"] and r["faithful"] for r in ans_rows) / len(ans_rows)) if ans_rows else 1.0
    clean = sum((not r["leak"]) and (not r["prescriptive"]) for r in rows) / n
    readable = sum(r["flesch"] >= 60 for r in rows) / n
    comp = (WEIGHTS["decision"] * decision + WEIGHTS["faithful"] * faithful
            + WEIGHTS["clean"] * clean + WEIGHTS["readable"] * readable)
    # per-category decision accuracy for diagnostics
    from collections import defaultdict
    cat = defaultdict(lambda: [0, 0])
    for r in rows:
        cat[r["gold"]][0] += int(r["decision_correct"]); cat[r["gold"]][1] += 1
    by_cat = {k: round(v[0] / v[1], 3) for k, v in cat.items()}
    return {"composite": round(comp, 4), "decision_accuracy": round(decision, 4),
            "answer_faithfulness": round(faithful, 4), "clean_rate": round(clean, 4),
            "readable_rate": round(readable, 4), "decision_by_category": by_cat}


def run(fid, split, n, workers):
    cfg = load_feature(fid)
    kb = v2.load_kb()
    items = json.loads((COACH / f"qa_set_{split}.json").read_text(encoding="utf-8"))
    if n:
        items = items[:n]
    if not items:
        raise SystemExit(f"no qa items — run build_qa_set.py --split {split} first")
    print(f"[qa-eval] feature={cfg['id']} split={split} items={len(items)} workers={workers}")
    sc_cache = {}

    def _sc(cid):
        if cid not in sc_cache:
            sc_cache[cid] = json.loads((SPLIT_DIRS[split] / f"{cid}.json").read_text(encoding="utf-8"))
        return sc_cache[cid]

    def _one(item):
        sc = _sc(item["clip"])
        out = gen_cached(cfg, split, item, sc, kb)
        if out is None:
            return None
        return QA.grade(item, sc, out)

    rows, done = [], 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_one, it): it for it in items}
        for fut in as_completed(futs):
            done += 1
            r = fut.result()
            if r is not None:
                rows.append(r)
            if done % 20 == 0:
                print(f"  {done}/{len(items)}", flush=True)

    agg = composite(rows)
    report = {"feature_id": cfg["id"], "task": "qa", "split": split, "weights": WEIGHTS,
              "n_evaluated": len(rows), "n_requested": len(items), "scores": agg,
              "passes_0.8": agg["composite"] >= 0.8, "config": cfg, "per_item": rows}
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / f"{cfg['id']}__qa_{split}.json"
    out_path.write_text(json.dumps(report, indent=1), encoding="utf-8")

    print("\n" + "=" * 60 + f"\n  {cfg['id']}  [qa/{split}]  n={len(rows)}\n" + "=" * 60)
    for k in ("decision_accuracy", "answer_faithfulness", "clean_rate", "readable_rate"):
        print(f"  {k:22s}: {agg[k]}")
    print(f"  decision_by_category : {agg['decision_by_category']}")
    print(f"  {'COMPOSITE':22s}: {agg['composite']}   {'PASS (>=0.8)' if report['passes_0.8'] else 'below 0.8'}")
    print(f"\n[qa-eval] wrote {out_path}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--feature", default="qa_v1")
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    run(a.feature, a.split, a.n, a.workers)
