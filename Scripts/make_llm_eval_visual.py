"""Slide visual for the LLM coaching-layer eval.

Renders Data/coaching/llm_eval_report.json into a presentation card:
a "risk -> check -> result" scorecard for the one component that can
hallucinate (the LLM explanation layer).

  python make_llm_eval_visual.py        # -> Data/coaching/viz_llm_eval.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

PROJECT_ROOT = Path(__file__).parent.parent
COACH = PROJECT_ROOT / "Data" / "coaching"
REPORT = COACH / "llm_eval_report.json"

# ---- design system (matches make_week7_hero_visuals.py) ----
INK = "#1d1d1f"; SUB = "#6b7280"; FAINT = "#d8dce2"
GREEN = "#1f9d57"; GREEN_D = "#136b3b"; GOLD = "#f0a847"
INDIGO = "#5b6cdb"; RED = "#d9485f"; CARD = "#f6f8fa"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.edgecolor": INK,
    "axes.linewidth": 0.8, "figure.dpi": 130, "savefig.dpi": 150,
})


def _card(ax, x, y, w, h, fc=CARD, ec=FAINT, lw=1.0):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.018",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2,
        transform=ax.transAxes, clip_on=False))


def _pill(ax, x, y, text, fc, tc="white"):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=10.5,
            fontweight="bold", color=tc, ha="center", va="center", zorder=5,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=fc, edgecolor="none"))


def build(report: dict, v1: dict | None = None) -> Path:
    d = report["deterministic"]
    g = report["grounding_judge"]
    cons = report.get("consistency")
    n = report["n_clips_evaluated"]
    gr_v1 = (v1 or {}).get("grounding_judge", {}).get("grounded_rate")

    # (label, sublabel, value-str, target-str, pass?, coverage-note)
    leak_ok = d["leakage_total"] == 0
    presc_ok = d["prescriptive_total"] == 0
    cov = d.get("coverage_mean")
    cov_ok = cov is not None and cov >= 0.9
    read_ok = d["flesch_mean"] >= 60
    gr = g.get("grounded_rate")
    gr_ok = gr is not None and gr >= 0.9

    cards = [
        ("Low-confidence leakage", "narrates a metric we don't trust",
         str(d["leakage_total"]), "target  0", leak_ok, f"all {n} clips"),
        ("Prescriptive advice", "gives fix-it tips (out of MVP scope)",
         str(d["prescriptive_total"]), "target  0", presc_ok, f"all {n} clips"),
        ("Flagged-metric coverage", "mentions every finding worth noting",
         f"{cov:.0%}" if cov is not None else "—", "target  100%", cov_ok, f"all {n} clips"),
        ("Readability (Flesch)", "plain English for a beginner",
         f"{d['flesch_mean']:.0f}", "target  ≥ 60", read_ok, f"all {n} · min {d.get('flesch_min')}"),
        ("Grounding / faithfulness", "every claim backed by the data",
         f"{gr:.0%}" if gr is not None else "—", "LLM-judge", gr_ok,
         (f"↑ from {gr_v1:.0%} after bug-fix" if gr_v1 is not None else f"audited n={g['n_judged']}")),
        ("Consistency", "same swing → same story",
         (f"{cons['mean_jaccard']:.0%}" if cons and cons.get("mean_jaccard") is not None else "—"),
         "3× regen", (cons is not None and (cons.get("mean_jaccard") or 0) >= 0.7),
         (f"n={cons['n_clips']}" if cons else "—")),
    ]

    fig = plt.figure(figsize=(13, 7.3)); fig.patch.set_facecolor("white")
    fig.text(0.045, 0.95, "Monitoring the LLM Coaching Layer", fontsize=24,
             fontweight="bold", color=INK, va="top")
    fig.text(0.045, 0.882,
             "The LLM is the only component that can make things up. We score every "
             f"explanation against the data it was written from  (n = {n} GolfDB clips).",
             fontsize=13, color=SUB, va="top")

    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # 2 rows x 3 cols grid of check cards
    cols, rows = 3, 2
    x0, y0 = 0.045, 0.135
    gw, gh = 0.91, 0.62
    pad = 0.022
    cw = (gw - (cols - 1) * pad) / cols
    ch = (gh - (rows - 1) * pad) / rows
    for i, (label, sub, val, tgt, ok, note) in enumerate(cards):
        r, c = divmod(i, cols)
        cx = x0 + c * (cw + pad)
        cy = y0 + (rows - 1 - r) * (ch + pad)
        accent = GREEN if ok else GOLD
        _card(ax, cx, cy, cw, ch, fc=CARD, ec=accent, lw=1.6)
        # accent bar on left
        ax.add_patch(FancyBboxPatch((cx, cy), 0.006, ch, boxstyle="square,pad=0",
                     facecolor=accent, edgecolor="none", zorder=3,
                     transform=ax.transAxes, clip_on=False))
        tx = cx + 0.022
        ax.text(tx, cy + ch - 0.045, label, transform=ax.transAxes,
                fontsize=13.5, fontweight="bold", color=INK, va="top")
        ax.text(tx, cy + ch - 0.088, sub, transform=ax.transAxes,
                fontsize=10.3, color=SUB, va="top", style="italic")
        # big result number
        ax.text(tx, cy + 0.052, val, transform=ax.transAxes,
                fontsize=33, fontweight="bold", color=accent, va="bottom")
        ax.text(cx + cw - 0.018, cy + 0.052, tgt, transform=ax.transAxes,
                fontsize=11, color=SUB, va="bottom", ha="right")
        # pass / flag pill + coverage note
        _pill(ax, tx + 0.052, cy + ch - 0.130, "PASS" if ok else "WATCH",
              GREEN if ok else GOLD)
        ax.text(cx + cw - 0.018, cy + 0.016, note, transform=ax.transAxes,
                fontsize=9, color=SUB, va="bottom", ha="right")

    # footer
    fig.text(0.045, 0.055,
             "Deterministic checks run on all 100 clips every time; grounding uses an LLM-judge on an audited subset (n reported, not hidden).",
             fontsize=10.5, color=SUB)
    fig.text(0.045, 0.022,
             "The eval earned its keep: it caught a real grounding bug (the LLM lumping sub-metrics into a family and calling it “in range”) — we fixed it and grounding doubled.",
             fontsize=10.5, color=INDIGO, fontweight="bold")

    out = COACH / "viz_llm_eval.png"
    fig.savefig(out, facecolor="white", bbox_inches="tight"); plt.close(fig)
    return out


if __name__ == "__main__":
    if not REPORT.exists():
        sys.exit(f"no report yet at {REPORT} — run coaching_llm_eval.py first")
    v1_path = COACH / "llm_eval_report_v1_before_fix.json"
    v1 = json.loads(v1_path.read_text()) if v1_path.exists() else None
    out = build(json.loads(REPORT.read_text()), v1)
    print(f"wrote {out}")
