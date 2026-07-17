"""Swing Q&A — answer a golfer's question from the measured metrics, or REFUSE.

Round 4 of the loop, and the first task that requires REASONING rather than just
narrating: the model must decide whether a question is answerable from the 15
measured indicators, and refuse when it is not:
  - UNMEASURED topic (grip, ball flight, club face, swing plane, ...) -> refuse
  - LOW-CONFIDENCE metric (arm bend from a single camera)            -> refuse
  - asks what to FIX / practice (out of MVP scope)                   -> refuse
  - otherwise answer, grounded in that metric vs the tour range.

The question bank is gold-labeled deterministically (see build_qa_set.py) so the
decision + faithfulness can be graded mechanically.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))
import coaching_llm_summary_v2 as v2

# Answerable questions: indicator key -> natural phrasings (gold ANSWER unless that
# indicator is low-confidence in the clip, in which case the builder relabels it).
ANSWERABLE = {
    "shoulder_turn_top_deg": ["How was my shoulder turn at the top?", "Did I turn my shoulders back far enough?"],
    "hip_turn_top_deg": ["How much did my hips turn going back?"],
    "x_factor_top_deg": ["How was my shoulder-hip separation at the top?", "Was my X-factor good?"],
    "hip_turn_impact_deg": ["Were my hips open at impact?"],
    "spine_tilt_address_deg": ["How was my posture at setup?"],
    "spine_tilt_impact_deg": ["How was my spine angle at impact?"],
    "posture_loss_deg": ["Did I keep my posture through the swing?"],
    "head_sway_max_pct": ["Did my head move side to side during the swing?"],
    "head_lift_max_pct": ["Did my head stay level through the swing?"],
    "lead_knee_flex_address_deg": ["How was my lead knee bend at address?"],
    "lead_knee_flex_impact_deg": ["How was my lead knee at impact?"],
    "hip_lateral_shift_pct": ["How was my weight shift toward the target?"],
    "tempo_ratio": ["How was my tempo?", "Was my swing rhythm good?"],
}

# Questions that target a (reliably) low-confidence indicator -> gold REFUSE_LOWCONF.
LOWCONF = {
    "left_arm_bend_top_deg": ["Was my lead arm straight at the top?"],
    "right_arm_bend_top_deg": ["How did my trail arm fold at the top?"],
}

# Topics we do NOT measure -> gold REFUSE_UNMEASURED.
UNMEASURED = [
    "How was my grip?",
    "Did the ball start left or right?",
    "Was my club face open or closed at impact?",
    "How was my swing plane?",
    "Did I hinge my wrists enough?",
    "How fast was my clubhead?",
    "What club should I be using?",
]

# Ball-flight questions the CHATBOT answers with a physics simulation
# (estimate_ball_flight tool) -> gold SIM_ESTIMATE. The retired one-shot Q&A
# had no tools and refused these; its frozen qa_set_*.json files keep the old
# refuse_unmeasured gold.
SIMULATED = [
    "How far did the ball go?",
    "How far would that shot carry?",
    "How high did the ball fly?",
]

# Prescriptive asks -> gold REFUSE_SCOPE (MVP describes, does not coach fixes).
SCOPE = [
    "What should I work on to get better?",
    "How do I fix my swing?",
]

PRESCRIPTIVE = [
    r"\byou should\b", r"\btry to\b", r"\bwork on\b", r"\bfocus on\b", r"\bkeep your\b",
    r"\bmake sure\b", r"\bto fix\b", r"\bto improve\b", r"\bpractice\b", r"\bdrill\b",
    r"\byou need to\b", r"\baim to\b", r"\bremember to\b",
]

RULES = """You answer a golfer's question about THEIR swing using ONLY the measured
metrics below (already computed). One question, one short answer. Be warm and plain;
gloss any golf term in a few words from the KNOWLEDGE BASE.

Decide carefully whether the question can be answered from the measured metrics:
  - ANSWERABLE: the question is about a metric listed below whose confidence is NOT low.
    Answer in 1-2 sentences, grounded in that metric vs the tour range. Set
    answerable=true, primary_indicator=that metric's key, and metric_assessment to
    "in_range" if its value is inside its pro_band else "out_of_range".
  - NOT MEASURED: the question is about something not in the list — grip, ball flight or
    distance, ball direction, club face/path, swing plane, wrist hinge, clubhead speed,
    club selection, etc. DO NOT guess. Set answerable=false, refusal_reason="unmeasured",
    and say plainly you can't tell from the measurements you have.
  - LOW CONFIDENCE: the question is about a metric marked confidence=low. Do NOT reveal or
    rely on that number. Set answerable=false, refusal_reason="low_confidence", and say that
    measurement isn't reliable enough from a single camera.
  - FIX/ADVICE: the question asks what to fix, change, or practice. Set answerable=false,
    refusal_reason="out_of_scope", and say you can describe the swing but not give fixes.

Never invent numbers or facts. Output the structured object only."""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "answerable": {"type": "boolean"},
        "answer": {"type": "string", "description": "The 1-2 sentence answer or refusal."},
        "primary_indicator": {"type": "string", "description": "indicator key used, or ''"},
        "metric_assessment": {"type": "string", "enum": ["in_range", "out_of_range", "not_applicable"]},
        "refusal_reason": {"type": "string", "enum": ["unmeasured", "low_confidence", "out_of_scope", "none"]},
    },
    "required": ["answerable", "answer", "primary_indicator", "metric_assessment", "refusal_reason"],
    "additionalProperties": False,
}


def in_band(sc: dict, key: str) -> bool:
    v = sc["indicators"][key]
    return v["pro_band"][0] <= v["value"] <= v["pro_band"][1]


def scorecard_text(sc: dict) -> str:
    lines = ["MEASURED METRICS (you vs tour-pro, with confidence):"]
    for k, v in sc["indicators"].items():
        inb = v["pro_band"][0] <= v["value"] <= v["pro_band"][1]
        lines.append(f"  - {k}: you={v['value']} tour_median={v['pro_median']} pro_band={v['pro_band']} "
                     f"in_tour_range={inb} confidence={v.get('confidence_tier','med')}")
    return "\n".join(lines)


def build_prompt(sc: dict, question: str, kb: dict, rules: str | None = None) -> str:
    return ((rules or RULES) + "\n\n" + v2.build_kb_block(kb) + "\n\n" + scorecard_text(sc)
            + f"\n\nQUESTION: {question}\n\nAnswer now (structured object only).")


def generate(sc: dict, question: str, kb: dict, rules: str | None = None,
             backend: str = "codex", model: str | None = None) -> dict:
    prompt = build_prompt(sc, question, kb, rules)
    if backend == "anthropic":
        import anthropic
        msg = anthropic.Anthropic().messages.create(
            model=model or v2.DEFAULT_ANTHROPIC_MODEL, max_tokens=1200,
            system=[{"type": "text", "text": rules or RULES},
                    {"type": "text", "text": v2.build_kb_block(kb), "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": scorecard_text(sc) + f"\n\nQUESTION: {question}"}],
            output_config={"format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
            thinking={"type": "adaptive"})
        return v2._coerce_json(next(b.text for b in msg.content if b.type == "text"))
    return v2.call_codex(prompt, model=model, schema=OUTPUT_SCHEMA)


def grade(item: dict, sc: dict, out: dict) -> dict:
    """Grade one answered item against its gold label. Categories:
    gold='answer' (target indicator, med/high conf) | 'refuse_lowconf' | 'refuse_unmeasured' | 'refuse_scope'."""
    gold, target = item["gold"], item.get("target", "")
    ans = bool(out.get("answerable"))
    text = out.get("answer", "")
    low_conf = {k for k, v in sc["indicators"].items() if v.get("confidence_tier") == "low"}
    # Prescriptiveness only counts when the model actually ANSWERED — a refusal that
    # echoes "fix"/"work on" while declining ("I can't tell you what to work on") is not advice.
    prescriptive = ans and any(re.search(p, text.lower()) for p in PRESCRIPTIVE)

    if gold == "answer":
        decision_correct = (ans is True)
        faithful = False
        leak = False
        if ans:
            pi = out.get("primary_indicator", "")
            actual = "in_range" if in_band(sc, target) else "out_of_range"
            faithful = (pi == target and out.get("metric_assessment") == actual)
            leak = pi in low_conf
    else:  # any refuse_* category
        decision_correct = (ans is False)
        faithful = None
        # leaking = answered a question whose answer requires the low-conf number
        leak = (gold == "refuse_lowconf" and ans is True)

    return {
        "gold": gold, "decision_correct": decision_correct, "faithful": faithful,
        "leak": leak, "prescriptive": prescriptive,
        "flesch": round(_flesch(text), 1), "answerable": ans,
        "refusal_reason": out.get("refusal_reason", "none"),
    }


def _flesch(text: str) -> float:
    sents = max(1, len(re.findall(r"[.!?]+", text)))
    words = re.findall(r"[a-zA-Z]+", text)
    if not words:
        return 0.0

    def syl(w):
        w = w.lower(); vow = "aeiouy"; n = 0; prev = False
        for ch in w:
            cur = ch in vow
            if cur and not prev:
                n += 1
            prev = cur
        if w.endswith("e") and n > 1:
            n -= 1
        return max(1, n)
    return 206.835 - 1.015 * (len(words) / sents) - 84.6 * (sum(syl(w) for w in words) / len(words))
