"""Eval harness for the LLM coaching/explanation layer.

Checks (the things that matter for a user-facing narration layer):
  DETERMINISTIC (all clips, no extra LLM calls):
    - leakage:        does it mention a LOW-confidence metric? (must be 0)
    - prescriptiveness: does it give fix-it advice? (MVP scope: must be 0)
    - coverage:       does it mention each flagged (out-of-band) metric?
    - readability:    Flesch reading-ease (beginner-friendly target ~>=60)
  LLM-JUDGE (subset, Codex):
    - grounding:      every claim supported by the measured metrics? (faithfulness)
  REPEAT (subset):
    - consistency:    same clip, 3 runs -> how stable are the claims?

Generation is resumable (cached per clip) and tolerant of individual failures.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SCORECARDS = PROJECT_ROOT / "Data" / "coaching" / "eval_scorecards"
GEN_DIR = PROJECT_ROOT / "Data" / "coaching" / "eval_llm_text"
REPORT = PROJECT_ROOT / "Data" / "coaching" / "llm_eval_report.json"
CODEX = shutil.which("codex.cmd") or shutil.which("codex") or "codex"

sys.path.insert(0, str(Path(__file__).parent))
from coaching_llm_summary import build_prompt, call_codex  # reuse the production prompt

# metric concept -> phrases that signal the LLM referenced it
CONCEPTS = {
    "shoulder_turn_top_deg": ["shoulder turn", "shoulders turn", "shoulder rotation", "turn your shoulders"],
    "hip_turn_top_deg": ["hip turn", "hips turn", "hip rotation"],
    "hip_turn_impact_deg": ["hips were", "hips open", "hip rotation", "hips turned", "open at impact"],
    "x_factor_top_deg": ["separation", "x-factor", "coil"],
    "spine_tilt_address_deg": ["spine", "posture", "tilt", "upper-body tilt"],
    "spine_tilt_impact_deg": ["spine", "posture", "tilt", "upper-body tilt"],
    "posture_loss_deg": ["posture", "spine angle", "stood up", "early extension"],
    "head_sway_max_pct": ["head", "side to side", "sway"],
    "head_lift_max_pct": ["head", "head height", "up and down"],
    "left_arm_bend_top_deg": ["lead arm", "lead-arm", "left arm", "arm extension", "arm straight", "elbow", "width"],
    "right_arm_bend_top_deg": ["trail arm", "right arm", "elbow"],
    "lead_knee_flex_address_deg": ["knee"],
    "lead_knee_flex_impact_deg": ["knee"],
    "hip_lateral_shift_pct": ["weight shift", "weight moved", "lateral", "slide", "drive toward", "weight transfer"],
    "tempo_ratio": ["tempo", "timing", "rhythm"],
}

# prescriptive / fix-it language (MVP says explain, don't coach)
PRESCRIPTIVE = [
    r"\byou should\b", r"\btry to\b", r"\btry \b", r"\bwork on\b", r"\bfocus on\b",
    r"\bkeep your\b", r"\bmake sure\b", r"\bto fix\b", r"\bto improve\b", r"\bpractice\b",
    r"\bdrill\b", r"\byou need to\b", r"\baim to\b", r"\bensure\b", r"\bremember to\b",
]


def _count_syllables(word: str) -> int:
    word = word.lower()
    v = "aeiouy"; n = 0; prev = False
    for ch in word:
        is_v = ch in v
        if is_v and not prev: n += 1
        prev = is_v
    if word.endswith("e") and n > 1: n -= 1
    return max(1, n)


def flesch_reading_ease(text: str) -> float:
    sents = max(1, len(re.findall(r"[.!?]+", text)))
    words = re.findall(r"[a-zA-Z]+", text)
    if not words: return 0.0
    syl = sum(_count_syllables(w) for w in words)
    return 206.835 - 1.015 * (len(words) / sents) - 84.6 * (syl / len(words))


def low_conf_metrics(sc: dict) -> list[str]:
    return [k for k, v in sc["indicators"].items() if v.get("confidence_tier") == "low"]


def flagged_metrics(sc: dict) -> list[str]:
    return [f["indicator"] for f in sc["feedback"] if f.get("severity") == "review"]


def mentions(text: str, metric: str) -> bool:
    t = text.lower()
    return any(p in t for p in CONCEPTS.get(metric, []))


def deterministic_checks(sc: dict, text: str) -> dict:
    t = text.lower()
    leaks = [m for m in low_conf_metrics(sc) if mentions(text, m)]
    presc = [p for p in PRESCRIPTIVE if re.search(p, t)]
    flagged = flagged_metrics(sc)
    covered = [m for m in flagged if mentions(text, m)]
    return {
        "leakage_metrics": leaks,
        "leakage_count": len(leaks),
        "prescriptive_hits": presc,
        "prescriptive_count": len(presc),
        "flagged_metrics": flagged,
        "flagged_covered": covered,
        "coverage": (len(covered) / len(flagged)) if flagged else None,
        "flesch_reading_ease": round(flesch_reading_ease(text), 1),
        "word_count": len(re.findall(r"[a-zA-Z]+", text)),
    }


JUDGE_PROMPT = """TASK: You are auditing whether a swing explanation is FAITHFUL to measured data.
Below are the MEASURED METRICS (each with value, tour median, in_range flag, confidence)
and an EXPLANATION written about them. List ONLY statements that genuinely CONTRADICT the
metrics or INVENT specifics. Apply these rules exactly:

  - A claim that a metric is "in range"/"tour-like" is SUPPORTED iff that exact metric's
    in_range=True; flag it ONLY if it says in-range while in_range=False (or vice versa).
  - An accurate summary such as "nothing stood out" / "no confident metric was outside the
    tour range" IS supported when no high-confidence metric has in_range=False. Do not flag
    true summaries.
  - General encouragement, plain-language glosses of golf terms, and neutral descriptions of
    a real out-of-range metric are ALLOWED — not unsupported.
  - DO flag: stating an out-of-range metric is in range, wrong direction, an invented number,
    or a phase/timing detail not present in the metric names.

OUTPUT: a JSON object only: {"unsupported": ["..."], "grounded": true/false}
(grounded=true means zero genuinely contradicting/invented statements).
"""


def judge_grounding(sc: dict, text: str, timeout: int = 180) -> dict | None:
    metrics_lines = []
    for k, v in sc["indicators"].items():
        inb = v["pro_band"][0] <= v["value"] <= v["pro_band"][1]
        metrics_lines.append(f"  {k}: you={v['value']} tour_median={v['pro_median']} "
                             f"in_range={inb} confidence={v.get('confidence_tier')}")
    prompt = (JUDGE_PROMPT + "\nMEASURED METRICS:\n" + "\n".join(metrics_lines)
              + "\n\nEXPLANATION:\n" + text + "\n\nReturn the JSON now:")
    out = call_codex(prompt, timeout=timeout)
    if not out: return None
    m = re.search(r"\{[\s\S]*\}", out)
    if not m: return None
    try:
        return json.loads(m.group())
    except Exception:
        return None


def generate(sc: dict, clip_id: int) -> str | None:
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    cache = GEN_DIR / f"{clip_id}.txt"
    if cache.exists() and cache.read_text(encoding="utf-8").strip():
        return cache.read_text(encoding="utf-8")
    txt = call_codex(build_prompt(sc))
    if txt:
        cache.write_text(txt, encoding="utf-8")
    return txt


def mentioned_concepts(text: str) -> set[str]:
    """Set of metric concepts the text references (for consistency comparison)."""
    return {m for m in CONCEPTS if mentions(text, m)}


def consistency_pass(clip_ids: list[int], k_clips: int, n_runs: int = 3, workers: int = 8) -> dict:
    """Regenerate the same clips n_runs times and measure stability of which
    metrics get narrated (mean pairwise Jaccard of the mentioned-concept set)."""
    from concurrent.futures import ThreadPoolExecutor

    def _one(cid):
        sc = json.loads((SCORECARDS / f"{cid}.json").read_text())
        run_sets = []
        for r in range(n_runs):
            cache = GEN_DIR / f"{cid}.txt" if r == 0 else GEN_DIR / f"{cid}__c{r}.txt"
            if cache.exists() and cache.read_text(encoding="utf-8").strip():
                txt = cache.read_text(encoding="utf-8")
            else:
                txt = call_codex(build_prompt(sc))
                if txt:
                    cache.write_text(txt, encoding="utf-8")
            if txt:
                run_sets.append(mentioned_concepts(txt))
        if len(run_sets) < 2:
            return None
        sims = []
        for a in range(len(run_sets)):
            for b in range(a + 1, len(run_sets)):
                u = run_sets[a] | run_sets[b]
                sims.append((len(run_sets[a] & run_sets[b]) / len(u)) if u else 1.0)
        return {"clip_id": cid, "runs": len(run_sets),
                "mean_jaccard": round(sum(sims) / len(sims), 3)}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        rows = [r for r in ex.map(_one, clip_ids[:k_clips]) if r is not None]
    import numpy as np
    return {"n_clips": len(rows),
            "mean_jaccard": round(float(np.mean([r["mean_jaccard"] for r in rows])), 3) if rows else None,
            "per_clip": rows}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--judge-n", type=int, default=25, help="clips to run LLM-judge grounding on")
    p.add_argument("--consistency-n", type=int, default=0,
                   help="clips to regenerate 3x for a stability check (0 = skip)")
    p.add_argument("--workers", type=int, default=8,
                   help="concurrent Codex calls (generation + judge)")
    args = p.parse_args()

    from concurrent.futures import ThreadPoolExecutor, as_completed

    clip_ids = json.loads((SCORECARDS / "_clip_ids.json").read_text())
    clip_ids = [c for c in clip_ids if (SCORECARDS / f"{c}.json").exists()]
    print(f"[eval] {len(clip_ids)} clips, {args.workers} workers")

    # --- Stage B+det: generate (parallel) + deterministic checks ---
    def _gen_and_check(cid):
        sc = json.loads((SCORECARDS / f"{cid}.json").read_text())
        text = generate(sc, cid)
        if not text:
            return None
        det = deterministic_checks(sc, text)
        det["clip_id"] = cid
        return det

    per_clip = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_gen_and_check, cid): cid for cid in clip_ids}
        for fut in as_completed(futs):
            done += 1
            d = fut.result()
            if d is None:
                print(f"  clip {futs[fut]}: generation failed, skipping")
            else:
                per_clip.append(d)
            if done % 10 == 0:
                print(f"  generated+checked {done}/{len(clip_ids)}", flush=True)
    per_clip.sort(key=lambda d: clip_ids.index(d["clip_id"]))

    # aggregate deterministic
    import numpy as np
    leak = sum(d["leakage_count"] for d in per_clip)
    presc = sum(d["prescriptive_count"] for d in per_clip)
    cov = [d["coverage"] for d in per_clip if d["coverage"] is not None]
    flesch = [d["flesch_reading_ease"] for d in per_clip]

    # --- Stage C: LLM-judge grounding on a subset (parallel) ---
    def _judge(cid):
        sc = json.loads((SCORECARDS / f"{cid}.json").read_text())
        text = (GEN_DIR / f"{cid}.txt").read_text(encoding="utf-8")
        verdict = judge_grounding(sc, text)
        if verdict is None:
            return None
        return {"clip_id": cid, "grounded": bool(verdict.get("grounded")),
                "n_unsupported": len(verdict.get("unsupported", [])),
                "unsupported": verdict.get("unsupported", [])}

    judge_cids = [d["clip_id"] for d in per_clip[: args.judge_n]]
    judged = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(_judge, judge_cids):
            if r is not None:
                judged.append(r)

    grounded_rate = (sum(j["grounded"] for j in judged) / len(judged)) if judged else None

    consistency = consistency_pass(clip_ids, args.consistency_n, workers=args.workers) if args.consistency_n else None

    report = {
        "n_clips_evaluated": len(per_clip),
        "deterministic": {
            "leakage_total": leak,
            "leakage_rate": round(leak / max(1, len(per_clip)), 3),
            "prescriptive_total": presc,
            "prescriptive_rate": round(presc / max(1, len(per_clip)), 3),
            "coverage_mean": round(float(np.mean(cov)), 3) if cov else None,
            "flesch_mean": round(float(np.mean(flesch)), 1),
            "flesch_min": round(float(np.min(flesch)), 1) if flesch else None,
        },
        "grounding_judge": {
            "n_judged": len(judged),
            "grounded_rate": round(grounded_rate, 3) if grounded_rate is not None else None,
        },
        "consistency": consistency,
        "per_clip": per_clip,
        "judged": judged,
    }
    REPORT.write_text(json.dumps(report, indent=1))

    print("\n" + "=" * 56)
    print(f"  LLM COACHING-LAYER EVAL  (n={len(per_clip)} clips)")
    print("=" * 56)
    d = report["deterministic"]
    print(f"  Low-confidence leakage : {d['leakage_total']} mentions  (target 0)")
    print(f"  Prescriptive advice    : {d['prescriptive_total']} hits  (target 0)")
    print(f"  Flagged-metric coverage: {d['coverage_mean']}  (1.0 = always mentions the flag)")
    print(f"  Readability (Flesch)   : mean {d['flesch_mean']}, min {d['flesch_min']}  (>=60 = plain English)")
    g = report["grounding_judge"]
    print(f"  Grounding (LLM-judge)  : {g['grounded_rate']} grounded on n={g['n_judged']} audited")
    if consistency:
        print(f"  Consistency (3x regen) : mean Jaccard {consistency['mean_jaccard']} on n={consistency['n_clips']}")
    print(f"\n[eval] wrote {REPORT}")


if __name__ == "__main__":
    main()
