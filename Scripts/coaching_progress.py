"""Progress-tracking narration: compare two swings by the same golfer and
narrate what CHANGED, grounded in the measured metric deltas.

New capability for round 3 of the eval loop. Same grounding philosophy as the
single-clip layer: deterministic code computes the per-metric change relative to
the tour-pro range; the LLM only narrates it. A structured `claims` list lets us
mechanically verify the narration never invents or mis-states a trend.

"Session A = earlier, Session B = later" is a framing convenience — for two
different swings by the same player it tests DELTA-NARRATION FAITHFULNESS, not
real longitudinal improvement.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))
import coaching_llm_summary_v2 as v2

# --------------------------------------------------------------------------- #
# Deterministic delta truth (no LLM)
# --------------------------------------------------------------------------- #

def _band_dist(v: float, band: list) -> float:
    lo, hi = band
    if v < lo:
        return lo - v
    if v > hi:
        return v - hi
    return 0.0


def delta_truth(scA: dict, scB: dict) -> dict:
    """Per-indicator change between two scorecards, classified relative to the
    tour-pro band: improved (moved toward range), regressed (away), or unchanged.
    Low-confidence-in-either indicators are marked non-narratable."""
    out = {}
    a_inds, b_inds = scA["indicators"], scB["indicators"]
    for k, a in a_inds.items():
        b = b_inds.get(k)
        if b is None:
            continue
        if a.get("confidence_tier") == "low" or b.get("confidence_tier") == "low":
            out[k] = {"narratable": False, "reason": "low_confidence"}
            continue
        band = a["pro_band"]
        bw = max(1e-6, band[1] - band[0])
        dA, dB = _band_dist(a["value"], band), _band_dist(b["value"], band)
        moved = abs(b["value"] - a["value"]) >= 0.15 * bw   # meaningful value move
        ddist = dB - dA                                      # <0 => toward range
        if not moved:
            direction = "unchanged"
        elif ddist < -0.05 * bw:
            direction = "improved"
        elif ddist > 0.05 * bw:
            direction = "regressed"
        else:
            direction = "unchanged"   # moved but stayed equally in/along the band
        out[k] = {"narratable": True, "direction": direction, "meaningful": moved,
                  "valA": a["value"], "valB": b["value"], "band": band,
                  "pctA": a["percentile"], "pctB": b["percentile"]}
    return out


# --------------------------------------------------------------------------- #
# Prompt assembly
# --------------------------------------------------------------------------- #

RULES = """You write a short, beginner-friendly note comparing two swings by the SAME
golfer — session A (earlier) and session B (later) — using MEASURED data (already
computed; do not recompute). One-shot writing task: the note only, no preamble.

Strict rules:
  - 2-3 short paragraphs, warm and plain. Gloss any golf term in a few words.
  - Lead with what IMPROVED (metrics whose change_vs_tour=improved).
  - Then describe any REGRESSED metrics as neutral observations of what changed —
    never as faults, never as instructions/tips to fix.
  - Describe a metric as improved/regressed ONLY if its change_vs_tour says so.
    Never claim a change for a metric whose change_vs_tour=unchanged.
  - NEVER mention any metric marked low-confidence.
  - Invent no numbers or trends not in the data. End with one encouraging sentence.
  - Use ONLY the wording/glosses from the KNOWLEDGE BASE; introduce no new golf terms.

Also return a `claims` list decomposing your note: each factual change statement
tagged with its indicator_key and the change (improved/regressed/unchanged), so it
can be checked against the data. Use claim_type:
  - "delta"        : a statement that a metric improved/regressed/stayed the same
  - "encouragement": general encouragement (indicator_key="", change="none")
  - "gloss"        : a plain-language term explanation (cite the indicator_key, change="none")
Every delta claim MUST give the exact indicator_key and the correct change."""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "explanation": {"type": "string", "description": "The comparison note, 2-3 short paragraphs."},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "indicator_key": {"type": "string"},
                    "change": {"type": "string", "enum": ["improved", "regressed", "unchanged", "none"]},
                    "claim_type": {"type": "string", "enum": ["delta", "encouragement", "gloss"]},
                },
                "required": ["text", "indicator_key", "change", "claim_type"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["explanation", "claims"],
    "additionalProperties": False,
}


def build_comparison_text(scA: dict, scB: dict, truth: dict) -> str:
    lines = ["TWO SESSIONS, SAME GOLFER (A = earlier, B = later).",
             "\nPER-METRIC CHANGE (already computed — narrate, do not recompute):"]
    for k, t in truth.items():
        if not t["narratable"]:
            lines.append(f"  - {k}: low-confidence — DO NOT mention.")
        else:
            lines.append(
                f"  - {k}: A={t['valA']} (p{t['pctA']}) -> B={t['valB']} (p{t['pctB']});  "
                f"change_vs_tour={t['direction']};  meaningful_move={t['meaningful']}")
    lines.append("\nNow write the comparison note and the claims list.")
    return "\n".join(lines)


def build_prompt(scA: dict, scB: dict, kb: dict, rules: str | None = None) -> str:
    truth = delta_truth(scA, scB)
    return (rules or RULES) + "\n\n" + v2.build_kb_block(kb) + "\n\n" + build_comparison_text(scA, scB, truth)


# --------------------------------------------------------------------------- #
# Deterministic verification of the narration vs the true deltas
# --------------------------------------------------------------------------- #

def verify_progress(scA: dict, scB: dict, claims: list[dict]) -> dict:
    truth = delta_truth(scA, scB)
    violations = []
    for cl in claims:
        if cl.get("claim_type") != "delta":
            continue
        k = cl.get("indicator_key", "")
        if k not in truth:
            violations.append({"type": "unknown_metric", "key": k, "claim": cl["text"]})
            continue
        t = truth[k]
        if not t["narratable"]:
            violations.append({"type": "low_confidence_leakage", "key": k, "claim": cl["text"]})
            continue
        claimed = cl.get("change")
        if claimed != t["direction"]:
            # covers invented trends (truth=unchanged, claim=improved/regressed) and wrong direction
            vtype = "invented_trend" if t["direction"] == "unchanged" else "wrong_direction"
            violations.append({"type": vtype, "key": k, "claimed": claimed,
                               "truth": t["direction"], "claim": cl["text"]})

    # coverage: meaningful (improved/regressed) changes that got a delta claim
    meaningful = [k for k, t in truth.items() if t["narratable"] and t["direction"] in ("improved", "regressed")]
    mentioned = {cl.get("indicator_key") for cl in claims if cl.get("claim_type") == "delta"}
    covered = [k for k in meaningful if k in mentioned]
    coverage = (len(covered) / len(meaningful)) if meaningful else None
    return {
        "grounded": len(violations) == 0,
        "violations": violations,
        "coverage": coverage,
        "n_meaningful": len(meaningful),
        "n_delta_claims": sum(1 for c in claims if c.get("claim_type") == "delta"),
    }


# --------------------------------------------------------------------------- #
# Generation (codex / anthropic) — mirrors v2
# --------------------------------------------------------------------------- #

def generate(scA: dict, scB: dict, kb: dict, rules: str | None = None,
             backend: str = "codex", model: str | None = None) -> dict:
    prompt = build_prompt(scA, scB, kb, rules)
    if backend == "anthropic":
        import anthropic
        msg = anthropic.Anthropic().messages.create(
            model=model or v2.DEFAULT_ANTHROPIC_MODEL, max_tokens=4000,
            system=[{"type": "text", "text": rules or RULES},
                    {"type": "text", "text": v2.build_kb_block(kb), "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": build_comparison_text(scA, scB, delta_truth(scA, scB))}],
            output_config={"format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
            thinking={"type": "adaptive"})
        return v2._coerce_json(next(b.text for b in msg.content if b.type == "text"))
    return v2.call_codex(prompt, model=model, schema=OUTPUT_SCHEMA)
