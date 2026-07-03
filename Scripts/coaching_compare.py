"""Production progress-comparison CLI — the shipped delta-narration layer.

Compares two scorecards from the SAME golfer (earlier A -> later B) and writes a
grounded, beginner-friendly note on what changed, using the GATED feature config
(pg_readable — composite 0.9998 on the held-out 200 pairs). Single source of
truth: reads the same feature file the eval loop validated.

Tolerant of LLM failure (warns, exits 0).

Usage:
    python coaching_compare.py --a <earlier>_scorecard.json --b <later>_scorecard.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import coaching_progress as P
import coaching_progress_eval as PE
import coaching_llm_summary_v2 as v2

PROD_FEATURE = "pg_readable"   # gated winner, held-out test composite 0.9998


def compare(a_json: Path, b_json: Path, feature: str = PROD_FEATURE, out_json: Path | None = None) -> dict:
    scA = json.loads(Path(a_json).read_text(encoding="utf-8"))
    scB = json.loads(Path(b_json).read_text(encoding="utf-8"))
    cfg = PE.load_feature(feature)
    out = P.generate(scA, scB, v2.load_kb(), rules=PE.rules_for(cfg),
                     backend=cfg.get("backend", "codex"), model=cfg.get("model"))
    grounding = P.verify_progress(scA, scB, out.get("claims", []))
    result = {
        "session_a": scA.get("meta", {}), "session_b": scB.get("meta", {}),
        "progress_note": out["explanation"], "claims": out.get("claims", []),
        "grounding": grounding, "feature": feature,
    }
    dest = Path(out_json) if out_json else Path(b_json).with_name(Path(b_json).stem + "_progress.json")
    dest.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a", required=True, help="earlier session scorecard JSON")
    p.add_argument("--b", required=True, help="later session scorecard JSON")
    p.add_argument("--feature", default=PROD_FEATURE)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    try:
        r = compare(Path(args.a), Path(args.b), args.feature, args.out)
    except Exception as e:
        print(f"[compare] LLM step skipped ({repr(e)[:140]})")
        return
    g = r["grounding"]
    print("\n" + "=" * 56 + "\n  PROGRESS NOTE (gated " + args.feature + ")\n" + "=" * 56)
    print(r["progress_note"])
    print(f"\n[compare] grounded={g['grounded']} violations={len(g['violations'])} "
          f"coverage={g['coverage']}")


if __name__ == "__main__":
    main()
