"""Presentation-grade Week-7 hero visuals (matplotlib, no torch).

  1. viz_event_detector_results.png  — the breakthrough: 0.170 -> 0.865, plus
     the player-disjoint validation and the SwingNet reference line.
  2. viz_event_timeline_accuracy.png — predicted vs ground-truth events on real
     clips (proof the detector lands the events).
  3. viz_hero_scorecard.png          — scorecard metrics + the LLM explanation,
     one cohesive presentation card.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from coaching_indicators import EVENTS

PROJECT_ROOT = Path(__file__).parent.parent
COACH = PROJECT_ROOT / "Data" / "coaching"
OUT = COACH
OUT.mkdir(parents=True, exist_ok=True)

# ---- design system ----
INK = "#1d1d1f"; SUB = "#6b7280"; FAINT = "#d8dce2"
GREEN = "#1f9d57"; GREEN_D = "#136b3b"; GOLD = "#f0a847"; PURPLE = "#9b59b6"
INDIGO = "#5b6cdb"; RED = "#d9485f"; BANDC = "#cdeadб" if False else "#cfe9d8"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.edgecolor": INK,
    "axes.linewidth": 0.8, "figure.dpi": 130, "savefig.dpi": 150,
})


def _titles(fig, title, sub):
    fig.text(0.04, 0.95, title, fontsize=23, fontweight="bold", color=INK, va="top")
    fig.text(0.04, 0.885, sub, fontsize=13.5, color=SUB, va="top")


# ===========================================================================
# 1. Event detector results
# ===========================================================================

def event_detector_results():
    pd_res = json.loads((COACH / "player_disjoint_result.json").read_text())
    rows = [
        ("Heuristic detector\n(wrist-Y argmin/argmax)", 0.170, "#9aa3b2"),
        ("GPT-5 / Codex\n(vision-only, n=32)", 0.258, PURPLE),
        ("Trained 1D-CNN\n(standard GolfDB split)", 0.865, GREEN),
        (f"Trained 1D-CNN\n(player-disjoint · {pd_res['n_test_players']} unseen golfers)",
         pd_res["pce_at_5"], GREEN_D),
    ]
    fig = plt.figure(figsize=(13, 7.2)); fig.patch.set_facecolor("white")
    _titles(fig, "The Event Detector — The Breakthrough",
            "We diagnosed that the swing-event detector, not the pose model, was the ceiling — then trained the fix.")
    ax = fig.add_axes([0.34, 0.22, 0.60, 0.54])

    y = np.arange(len(rows))[::-1]
    for yi, (lbl, val, c) in zip(y, rows):
        ax.barh(yi, val, height=0.6, color=c, edgecolor=INK, linewidth=0.8, zorder=3)
        ax.text(val + 0.012, yi, f"{val:.3f}", va="center", ha="left",
                fontsize=16, fontweight="bold", color=INK)
        ax.text(-0.02, yi, lbl, va="center", ha="right", fontsize=12, color=INK)
    # SwingNet reference line
    ax.axvline(0.76, color=RED, ls="--", lw=1.6, zorder=2)
    ax.text(0.76, len(rows) - 0.30, "GolfDB SwingNet\npaper ≈ 0.76", color=RED,
            fontsize=10.5, ha="center", va="bottom", fontweight="bold")

    ax.set_yticks([]); ax.set_xlim(0, 1.15)
    ax.set_xlabel("PCE@5  —  % of 8 swing events found within ±5 frames  (higher = better)", fontsize=12)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=FAINT, lw=0.7, zorder=0); ax.set_axisbelow(True)

    # headline callout
    fig.text(0.04, 0.105,
             "Generalizes to unseen golfers: 0.864 (player-disjoint) ≈ 0.865 (standard) — no player-overlap inflation.",
             fontsize=12.5, color=GREEN_D, fontweight="bold")
    fig.text(0.04, 0.045,
             "Why we beat the published baseline: the CNN consumes clean MotionBERT-Full 3D landmarks, not raw 160px video.",
             fontsize=11, color=SUB)
    out = OUT / "viz_event_detector_results.png"
    fig.savefig(out, facecolor="white", bbox_inches="tight"); plt.close(fig)
    return out


# ===========================================================================
# 2. Event timeline accuracy (predicted vs ground truth)
# ===========================================================================

def event_timeline_accuracy():
    data = json.loads((COACH / "event_predictions.json").read_text())
    data = data[:5]
    fig = plt.figure(figsize=(13, 7.6)); fig.patch.set_facecolor("white")
    _titles(fig, "Detected Events vs Ground Truth",
            "Trained detector (●) against human GolfDB labels (▲) on clips spanning players, views, tempo.")
    ax = fig.add_axes([0.16, 0.10, 0.80, 0.70])

    n = len(data)
    for row_i, d in enumerate(data):
        yy = n - 1 - row_i
        T = d["n_frames"]
        gt = np.array(d["ground_truth"]) / T
        pr = np.array(d["predicted"]) / T
        ax.plot([0, 1], [yy, yy], color=FAINT, lw=1.2, zorder=1)
        # connectors colored by error
        for g, p in zip(gt, pr):
            err = abs(g - p) * T
            c = GREEN if err <= 5 else (GOLD if err <= 10 else RED)
            ax.plot([g, p], [yy + 0.10, yy - 0.10], color=c, lw=1.6, zorder=2)
        ax.scatter(gt, [yy + 0.10] * len(gt), marker="^", s=60, color=INK, zorder=3)
        ax.scatter(pr, [yy - 0.10] * len(pr), marker="o", s=55, color=GREEN, zorder=3,
                   edgecolors="white", linewidths=0.6)
        hits = int(sum(abs(g - p) * T <= 5 for g, p in zip(gt, pr)))
        ax.text(-0.02, yy, f"{d['player'][:16]}\n{d['view']}", va="center", ha="right",
                fontsize=10, color=INK)
        ax.text(1.02, yy, f"{hits}/8 @±5", va="center", ha="left", fontsize=10.5,
                color=GREEN_D, fontweight="bold")

    ax.set_xlim(-0.0, 1.0); ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks([]); ax.set_xlabel("normalized swing timeline (address → finish)", fontsize=12)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    # legend
    h = [plt.Line2D([0], [0], marker="^", color="w", markerfacecolor=INK, markersize=10, label="GolfDB label (truth)"),
         plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=GREEN, markersize=10, label="detector prediction"),
         mpatches.Patch(color=GREEN, label="within ±5 frames"),
         mpatches.Patch(color=GOLD, label="±6–10"), mpatches.Patch(color=RED, label=">10")]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=5,
              frameon=False, fontsize=10)
    out = OUT / "viz_event_timeline_accuracy.png"
    fig.savefig(out, facecolor="white", bbox_inches="tight"); plt.close(fig)
    return out


# ===========================================================================
# 3. Hero scorecard (metrics + LLM explanation)
# ===========================================================================

def hero_scorecard():
    sc = json.loads((PROJECT_ROOT / "Data" / "demo" / "clip1292" / "clip1292_scorecard.json").read_text())
    inds = sc["indicators"]
    llm = sc.get("llm_summary", "")
    meta_player = "CHARLEY HULL"   # clip 1292

    fig = plt.figure(figsize=(14, 8.8)); fig.patch.set_facecolor("white")
    fig.text(0.035, 0.965, "Swing Scorecard + Plain-English Explanation", fontsize=22,
             fontweight="bold", color=INK, va="top")
    fig.text(0.035, 0.918, f"{meta_player} · face-on driver  —  the measured metrics layer feeds the LLM interpretation layer",
             fontsize=13, color=SUB, va="top")
    # column header for the metrics (own line, clear of the subtitle)
    fig.text(0.035, 0.862, "MEASURED METRICS   (● you   ▬ tour p25–p75)",
             fontsize=12.5, fontweight="bold", color=INK, va="top")

    # LEFT: indicator gauges
    rows = [
        ("Tempo (back:down)", "tempo_ratio", ":1"),
        ("Shoulder turn @ top", "shoulder_turn_top_deg", "°"),
        ("X-factor (coil)", "x_factor_top_deg", "°"),
        ("Hip rotation @ impact", "hip_turn_impact_deg", "°"),
        ("Spine tilt @ address", "spine_tilt_address_deg", "°"),
        ("Posture change", "posture_loss_deg", "°"),
        ("Head sway", "head_sway_max_pct", "%"),
        ("Weight shift (lateral)", "hip_lateral_shift_pct", "%"),
        ("Lead-arm extension", "left_arm_bend_top_deg", "°"),
    ]
    ax = fig.add_axes([0.035, 0.075, 0.52, 0.74]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, len(rows))
    fb_keys = {f["indicator"]: f for f in sc["feedback"] if f["severity"] == "review"}
    for i, (lbl, key, unit) in enumerate(rows):
        yy = len(rows) - 1 - i + 0.5
        d = inds[key]; low = d.get("confidence_tier") == "low"
        flagged = key in fb_keys
        tag = "  ⚠ low-confidence" if low else ""
        ax.text(0.0, yy + 0.20, lbl + tag, fontsize=11.5,
                color="#9aa3b2" if low else INK, va="center")
        ax.text(0.0, yy - 0.18, f"you {d['value']:g}{unit} · tour {d['pro_median']:g}{unit}",
                fontsize=8.8, color=SUB, va="center")
        bx0, bx1 = 0.46, 0.99
        ax.plot([bx0, bx1], [yy, yy], color="#e9ecf1", lw=6, solid_capstyle="round")
        ax.plot([bx0 + (bx1 - bx0) * .25, bx0 + (bx1 - bx0) * .75], [yy, yy],
                color=BANDC, lw=6, solid_capstyle="round")
        mp = bx0 + (bx1 - bx0) * min(max(d["percentile"], 2), 98) / 100
        mc = "#c2c8d0" if low else (GOLD if flagged else GREEN)
        ax.scatter(mp, yy, s=95, color=mc, edgecolors="white", linewidths=1.3, zorder=4)

    # RIGHT: LLM explanation card (top-aligned with the metrics column header)
    axr = fig.add_axes([0.58, 0.075, 0.40, 0.79]); axr.axis("off")
    card = FancyBboxPatch((0.0, 0.0), 1.0, 1.0, boxstyle="round,pad=0.02,rounding_size=0.03",
                          transform=axr.transAxes, facecolor="#f6eefb",
                          edgecolor=PURPLE, linewidth=2.2)
    axr.add_patch(card)
    axr.text(0.06, 0.95, "LLM SWING EXPLANATION", fontsize=12.5, fontweight="bold",
             color=PURPLE, va="top")
    axr.text(0.06, 0.88, "beginner-friendly · describes only confident metrics · no fix-it advice",
             fontsize=9, color=SUB, va="top", style="italic")
    # wrap the LLM text
    import textwrap
    wrapped = "\n".join("\n".join(textwrap.wrap(p, 52)) for p in llm.split("\n") if p.strip())
    axr.text(0.06, 0.80, wrapped, fontsize=11.2, color=INK, va="top", linespacing=1.45)

    # bottom flagged note
    note = next((f for f in sc["feedback"] if f["severity"] == "review"), None)
    if note:
        fig.text(0.035, 0.035,
                 f"⚑ Coaching note (high-confidence):  {note['message']}  (you p{note['percentile']})",
                 fontsize=10.5, color=GOLD, fontweight="bold")
    out = OUT / "viz_hero_scorecard.png"
    fig.savefig(out, facecolor="white", bbox_inches="tight"); plt.close(fig)
    return out


def main():
    for fn in (event_detector_results, event_timeline_accuracy, hero_scorecard):
        out = fn(); print("wrote", out.name, f"({out.stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()
