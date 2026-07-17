"""Interpretive feedback engine + swing scorecard.

TASK #4 (feedback rules) + #6 (scorecard). Each indicator is scored as a
percentile against the GolfDB pro reference band, then a transparent rule
turns out-of-band values into plain-language "coaching-lite" feedback.

Design commitments (per professor guidance — interpretive, not prescriptive):
  - Feedback describes WHAT differs from the pro distribution and at WHICH
    event. It never tells the golfer how to "fix" their swing.
  - Every message cites the metric + percentile so it is auditable.
  - "Within tour range" is a valid, common result — we don't manufacture
    problems.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from coaching_indicators import compute_indicators, INDICATOR_NAMES, EVENTS

PROJECT_ROOT = Path(__file__).parent.parent
REF_PATH = PROJECT_ROOT / "Data" / "coaching" / "reference_bands.json"
CONF_PATH = PROJECT_ROOT / "Data" / "coaching" / "indicator_confidence.json"


def load_confidence() -> dict:
    """Per-indicator depth-reliability confidence (from Austin Lee's EDA).
    Falls back to all-high if the file is missing."""
    if CONF_PATH.exists():
        return json.loads(CONF_PATH.read_text()).get("indicators", {})
    return {}

# Rule table: which direction is noteworthy + how to phrase it.
#   flag_low  : message if value below p10
#   flag_high : message if value above p90
#   event     : which swing event this relates to (for the user)
RULES = {
    "shoulder_turn_top_deg": {
        "event": "top", "label": "Shoulder turn",
        "flag_low": "Restricted shoulder turn at the top — less than most tour swings (loses coil/power).",
        "flag_high": None,
    },
    "x_factor_top_deg": {
        "event": "top", "label": "X-factor (shoulder-hip separation)",
        "flag_low": "Low shoulder-hip separation at the top — less stored coil than typical tour swings.",
        "flag_high": "Very high shoulder-hip separation — more coil than most; watch for lower-back strain.",
    },
    "hip_turn_impact_deg": {
        "event": "impact", "label": "Hip rotation at impact",
        "flag_low": "Hips less open at impact than typical — may indicate stalled rotation through the ball.",
        "flag_high": None,
    },
    "posture_loss_deg": {
        "event": "impact", "label": "Posture (spine angle)",
        "flag_low": None,
        "flag_high": "Noticeable posture change from address to impact — spine angle shifts more than most tour swings (early extension / standing up).",
    },
    "head_sway_max_pct": {
        "event": "backswing", "label": "Head sway",
        "flag_low": None,
        "flag_high": "More lateral head movement than typical tour swings — head drifts off the ball during the swing.",
    },
    "head_lift_max_pct": {
        "event": "downswing", "label": "Head height",
        "flag_low": None,
        "flag_high": "More vertical head movement than typical — head rises/drops more than most tour swings.",
    },
    "left_arm_bend_top_deg": {
        "event": "top", "label": "Lead-arm extension",
        "flag_low": "Lead arm more bent at the top than typical tour swings (less width/structure).",
        "flag_high": None,
    },
    "posture_drop_addr": {  # placeholder, not used
        "event": "address", "label": "", "flag_low": None, "flag_high": None,
    },
    "hip_lateral_shift_pct": {
        "event": "impact", "label": "Weight shift",
        "flag_low": "Less lateral hip drive toward target than typical — possible reverse/limited weight shift.",
        "flag_high": "More lateral hip slide than typical tour swings (sway rather than rotate).",
    },
    "tempo_ratio": {
        "event": "overall", "label": "Tempo (backswing : downswing)",
        "flag_low": "Quicker backswing-to-downswing ratio than the ~3:1 tour norm — tempo is on the fast side.",
        "flag_high": "Slower / longer backswing relative to downswing than the ~3:1 tour norm.",
    },
}


def load_reference():
    with open(REF_PATH) as f:
        return json.load(f)


def percentile_of(value: float, band: dict) -> float:
    """Approximate percentile of value within the reference band using its
    p10..p90 anchors with linear interpolation (robust, no full CDF needed)."""
    anchors = [(0.10, band["p10"]), (0.25, band["p25"]), (0.50, band["p50"]),
               (0.75, band["p75"]), (0.90, band["p90"])]
    if value <= anchors[0][1]:
        return 10.0 if value == anchors[0][1] else max(1.0, 10.0 * value / (anchors[0][1] + 1e-9)) if anchors[0][1] > 0 else 5.0
    if value >= anchors[-1][1]:
        return 90.0 + min(9.0, (value - anchors[-1][1]) / (abs(anchors[-1][1]) + 1e-9) * 10)
    for (p0, v0), (p1, v1) in zip(anchors, anchors[1:]):
        if v0 <= value <= v1:
            frac = (value - v0) / (v1 - v0 + 1e-9)
            return (p0 + frac * (p1 - p0)) * 100
    return 50.0


def generate_feedback(indicators: dict, reference: dict, confidence: dict | None = None) -> list[dict]:
    """Return feedback items for indicators outside the pro band.

    Reliability gating (Austin Lee's depth-reliability EDA): we do NOT raise a
    coaching note off a LOW-confidence indicator (e.g. arm bend, whose wrist
    depth is unreliable from single-POV). Low-confidence out-of-band values are
    surfaced as an informational note, not a coaching flag."""
    confidence = confidence or {}
    feedback = []
    for name, rule in RULES.items():
        if name not in indicators or name not in reference or not rule.get("label"):
            continue
        val = indicators[name]
        band = reference[name]
        pct = percentile_of(val, band)
        tier = confidence.get(name, {}).get("tier", "med")
        msg = None
        if pct <= 10 and rule.get("flag_low"):
            msg = rule["flag_low"]
        elif pct >= 90 and rule.get("flag_high"):
            msg = rule["flag_high"]
        if not msg:
            continue
        if tier == "low":
            # measurement too unreliable to coach on — flag as low-confidence only
            severity = "low_confidence"
            msg = f"(low-confidence measurement) {msg}"
        else:
            severity = "review"
        feedback.append({
            "indicator": name, "label": rule["label"], "event": rule["event"],
            "value": round(val, 1), "percentile": round(pct),
            "pro_median": round(band["p50"], 1), "confidence_tier": tier,
            "message": msg, "severity": severity,
        })
    return feedback


def build_scorecard(xyz: np.ndarray, events_local, clip_meta: dict | None = None,
                    fps: float | None = None) -> dict:
    reference = load_reference()
    confidence = load_confidence()
    indicators = compute_indicators(xyz, np.asarray(events_local), fps=fps)
    feedback = generate_feedback(indicators, reference, confidence)

    # status line — count only real coaching flags (not low-confidence notes)
    real_flags = [f for f in feedback if f["severity"] == "review"]
    n_flags = len(real_flags)
    if n_flags == 0:
        summary = "All confidently-measured indicators fall within the tour-pro range. Clean, repeatable mechanics."
    else:
        summary = f"{n_flags} indicator(s) fall outside the typical tour range — see notes below."

    scored = {}
    for name in INDICATOR_NAMES:
        if name in reference and name in indicators:
            scored[name] = {
                "value": round(indicators[name], 2),
                "percentile": round(percentile_of(indicators[name], reference[name])),
                "pro_median": round(reference[name]["p50"], 1),
                "pro_band": [round(reference[name]["p25"], 1), round(reference[name]["p75"], 1)],
                "confidence_tier": confidence.get(name, {}).get("tier", "med"),
            }

    return {
        "meta": clip_meta or {},
        "events": {e: int(events_local[i]) for i, e in enumerate(EVENTS)},
        "indicators": scored,
        "feedback": feedback,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# CLI / demo: build a scorecard for a GolfDB clip (treating it as "the user")
# ---------------------------------------------------------------------------

def _load_golfdb_clip(clip_id: int):
    import pandas as pd
    cache = PROJECT_ROOT / "Data" / "eval_runs" / "motionbert_full_from_mediapipe_lite" / f"{clip_id}.parquet"
    df = pd.read_parquet(cache)
    T = int(df["frame"].max()) + 1
    xyz = np.zeros((T, 17, 3), dtype=np.float32)
    for r in df.itertuples(index=False):
        if 0 <= r.kp_idx < 17:
            xyz[r.frame, r.kp_idx] = (r.x, r.y, r.z)
    gdb = pd.read_pickle(PROJECT_ROOT / "golfdb" / "golfDB.pkl").set_index("id")
    ev = np.asarray(gdb.loc[clip_id, "events"]); ev = ev[1:9] - ev[0]
    meta = {"clip_id": clip_id, "player": gdb.loc[clip_id, "player"],
            "club": gdb.loc[clip_id, "club"], "view": gdb.loc[clip_id, "view"]}
    return xyz, ev, meta


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--clip", type=int, default=0)
    args = p.parse_args()
    xyz, ev, meta = _load_golfdb_clip(args.clip)
    sc = build_scorecard(xyz, ev, meta)
    print(json.dumps(sc, indent=2))
