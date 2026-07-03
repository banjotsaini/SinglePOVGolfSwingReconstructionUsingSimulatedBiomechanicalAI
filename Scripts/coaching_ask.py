"""Production swing Q&A CLI — the shipped reasoning layer.

Answers a golfer's free-text question from their scorecard, or refuses when the
question is about something not measured / a low-confidence metric / a fix to make.
Uses the GATED feature config (qa_readable — composite 0.9910, decision accuracy
1.000 on the held-out 200). Tolerant of LLM failure (warns, exits 0).

Usage:
    python coaching_ask.py --scorecard <stem>_scorecard.json --question "How was my tempo?"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import coaching_qa as QA
import coaching_qa_eval as QE
import coaching_llm_summary_v2 as v2

PROD_FEATURE = "qa_readable"


def ask(scorecard_json: Path, question: str, feature: str = PROD_FEATURE) -> dict:
    sc = json.loads(Path(scorecard_json).read_text(encoding="utf-8"))
    cfg = QE.load_feature(feature)
    return QA.generate(sc, question, v2.load_kb(), rules=QE.rules_for(cfg),
                       backend=cfg.get("backend", "codex"), model=cfg.get("model"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scorecard", required=True)
    p.add_argument("--question", required=True)
    p.add_argument("--feature", default=PROD_FEATURE)
    args = p.parse_args()
    try:
        out = ask(Path(args.scorecard), args.question, args.feature)
    except Exception as e:
        print(f"[ask] LLM step skipped ({repr(e)[:140]})")
        return
    print("\nQ:", args.question)
    print("A:", out.get("answer", ""))
    if not out.get("answerable"):
        print(f"   (refused — reason: {out.get('refusal_reason')})")
    elif out.get("primary_indicator"):
        print(f"   (from {out['primary_indicator']} — {out.get('metric_assessment')})")


if __name__ == "__main__":
    main()
