"""Production single-clip explainer — the shipped interpretation layer.

Takes a scorecard JSON and writes a beginner-friendly, grounded explanation into
it, using the GATED feature config (f_strict_grounding — composite 1.000 on the
held-out 200). Single source of truth: it reads the same feature file the eval
loop validated, so production behavior == the gated behavior.

Grounding-failure-tolerant: if the LLM call fails (e.g. Codex offline) it warns
and exits 0, leaving the scorecard intact, so it never breaks the demo pipeline.

Usage:
    python coaching_explain.py --scorecard <path>_scorecard.json [--feature f_strict_grounding]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import coaching_eval_harness as H
import coaching_llm_summary_v2 as v2

PROD_FEATURE = "f_strict_grounding"   # gated winner, held-out test composite 1.000


def explain(scorecard_json: Path, feature: str = PROD_FEATURE) -> dict | None:
    sc = json.loads(Path(scorecard_json).read_text(encoding="utf-8"))
    cfg = H.load_feature(feature)
    out = H.generate(cfg, sc, v2.load_kb(), None)
    grounding = v2.verify_grounding(sc, out.get("claims", []))
    sc["llm_explanation"] = out["explanation"]
    sc["llm_claims"] = out.get("claims", [])
    sc["llm_grounding"] = grounding
    sc["llm_feature"] = feature
    Path(scorecard_json).write_text(json.dumps(sc, indent=2), encoding="utf-8")
    return sc


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scorecard", required=True)
    p.add_argument("--feature", default=PROD_FEATURE)
    args = p.parse_args()
    try:
        sc = explain(Path(args.scorecard), args.feature)
    except Exception as e:
        print(f"[explain] LLM step skipped ({repr(e)[:140]}); scorecard left intact")
        return  # exit 0 — never break the pipeline
    g = sc["llm_grounding"]
    print("\n" + "=" * 56 + "\n  LLM EXPLANATION (gated " + args.feature + ")\n" + "=" * 56)
    print(sc["llm_explanation"])
    print(f"\n[explain] grounded={g['grounded']} violations={len(g['violations'])} "
          f"coverage={g['coverage']}")


if __name__ == "__main__":
    main()
