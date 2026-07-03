"""Render a swing scorecard dict to a slide-ready PNG.

TASK #6 (visible deliverable). Layout:
  - header: player/club/view + one-line summary
  - swing-event timeline (the 8 detected events)
  - indicator gauges: pro band [p25-p75] bar + the golfer's marker, colored
    green (in band) / amber (outside)
  - feedback notes (the coaching-lite interpretation)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from coaching_indicators import EVENTS

GREEN = "#2e9e5b"
AMBER = "#e8973a"
GRAY  = "#5c6370"
DARK  = "#1e1e1e"
BAND  = "#cfe8d8"

# Indicators to show on the card + friendly labels + units
CARD_ROWS = [
    ("shoulder_turn_top_deg",      "Shoulder turn @ top",       "deg"),
    ("hip_turn_top_deg",           "Hip turn @ top",            "deg"),
    ("x_factor_top_deg",           "X-factor (coil)",           "deg"),
    ("hip_turn_impact_deg",        "Hip rotation @ impact",     "deg"),
    ("spine_tilt_address_deg",     "Spine tilt @ address",      "deg"),
    ("posture_loss_deg",           "Posture change (addr→imp)", "deg"),
    ("head_sway_max_pct",          "Head sway",                 "%"),
    ("head_lift_max_pct",          "Head lift",                 "%"),
    ("left_arm_bend_top_deg",      "Lead-arm extension @ top",  "deg"),
    ("lead_knee_flex_impact_deg",  "Lead knee flex @ impact",   "deg"),
    ("hip_lateral_shift_pct",      "Weight shift (lateral)",    "%"),
    ("tempo_ratio",                "Tempo (back:down)",         ":1"),
]


def render(scorecard: dict, out_path: str):
    meta = scorecard.get("meta", {})
    inds = scorecard["indicators"]
    fb_by_ind = {f["indicator"]: f for f in scorecard["feedback"]}

    fig = plt.figure(figsize=(13, 9), dpi=110, facecolor="white")
    gs = fig.add_gridspec(3, 2, height_ratios=[0.7, 0.5, 4.5], hspace=0.35, wspace=0.25,
                          left=0.04, right=0.97, top=0.95, bottom=0.05)

    # ---- header ----
    ax = fig.add_subplot(gs[0, :]); ax.axis("off")
    who = meta.get("player", "Golfer")
    sub = f"{meta.get('club','')} · {meta.get('view','')}".strip(" ·")
    ax.text(0, 0.7, f"Swing Scorecard — {who}", fontsize=24, fontweight="bold", color=DARK)
    ax.text(0, 0.15, sub, fontsize=13, color=GRAY)
    n_flags = len(scorecard["feedback"])
    badge = f"{n_flags} note(s)" if n_flags else "All within tour range"
    badge_color = AMBER if n_flags else GREEN
    ax.text(1.0, 0.7, badge, fontsize=15, fontweight="bold", color=badge_color,
            ha="right", transform=ax.transAxes)

    # ---- event timeline ----
    ax = fig.add_subplot(gs[1, :]); ax.axis("off")
    ev = scorecard["events"]
    maxf = max(ev.values()) or 1
    ax.plot([0, 1], [0.5, 0.5], color=GRAY, lw=1.5, transform=ax.transAxes)
    for name in EVENTS:
        x = ev[name] / maxf
        ax.scatter(x, 0.5, s=60, color=DARK, zorder=3, transform=ax.transAxes)
        ax.text(x, 0.05, name.replace("_", "\n"), fontsize=7.5, ha="center",
                va="bottom", color=GRAY, transform=ax.transAxes)
        ax.text(x, 0.78, f"{ev[name]}", fontsize=8, ha="center", color=DARK,
                transform=ax.transAxes)
    ax.text(0, 1.15, "Detected swing events (frame #)", fontsize=11, fontweight="bold",
            color=DARK, transform=ax.transAxes)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(0, 1.3)

    # ---- indicator gauges (two columns) ----
    half = (len(CARD_ROWS) + 1) // 2
    for col, rows in enumerate([CARD_ROWS[:half], CARD_ROWS[half:]]):
        ax = fig.add_subplot(gs[2, col]); ax.axis("off")
        ax.set_xlim(0, 1); ax.set_ylim(0, len(rows))
        for i, (key, label, unit) in enumerate(rows):
            y = len(rows) - 1 - i + 0.5
            if key not in inds:
                continue
            d = inds[key]
            flagged = key in fb_by_ind and fb_by_ind[key]["severity"] == "review"
            low_conf = d.get("confidence_tier") == "low"
            # label (+ low-confidence tag from Austin's reliability EDA)
            lbl = label + ("   ⚠ low-confidence" if low_conf else "")
            ax.text(0.0, y + 0.22, lbl, fontsize=11,
                    color="#999999" if low_conf else DARK, va="center")
            ax.text(0.0, y - 0.18, f"you: {d['value']:g}{unit}   ·   tour {d['pro_median']:g}{unit}",
                    fontsize=8.5, color=GRAY, va="center")
            # gauge geometry: map percentile 0..100 to a bar
            bx0, bx1, by = 0.50, 0.97, y
            ax.plot([bx0, bx1], [by, by], color="#dddddd", lw=6, solid_capstyle="round")
            # pro band p25-p75 = percentile 25..75
            pband0 = bx0 + (bx1 - bx0) * 0.25
            pband1 = bx0 + (bx1 - bx0) * 0.75
            ax.plot([pband0, pband1], [by, by], color=BAND, lw=6, solid_capstyle="round")
            # marker at the golfer's percentile (gray if low-confidence)
            mp = bx0 + (bx1 - bx0) * min(max(d["percentile"], 1), 99) / 100
            mcolor = "#bbbbbb" if low_conf else (AMBER if flagged else GREEN)
            ax.scatter(mp, by, s=90, color=mcolor,
                       edgecolors="white", linewidths=1.2, zorder=4)
        # column header
        ax.text(0.0, len(rows) + 0.1, "Indicator vs tour-pro band (●=you, █=p25–p75)",
                fontsize=10, fontweight="bold", color=DARK)

    # ---- feedback notes (overlay at bottom if any) ----
    if scorecard["feedback"]:
        lines = [f"• [{f['label']} @ {f['event']}]  {f['message']}  (you p{f['percentile']})"
                 for f in scorecard["feedback"]]
        note = "COACHING NOTES\n" + "\n".join(lines)
        fig.text(0.04, 0.005, note, fontsize=8.5, color=DARK, va="bottom",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff6e6", edgecolor=AMBER))

    fig.savefig(out_path, dpi=110, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    import argparse, json
    from coaching_scorecard import build_scorecard, _load_golfdb_clip
    p = argparse.ArgumentParser()
    p.add_argument("--clip", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    xyz, ev, meta = _load_golfdb_clip(args.clip)
    sc = build_scorecard(xyz, ev, meta)
    out = args.out or str(Path(__file__).parent.parent / "Data" / "coaching" / f"scorecard_clip{args.clip}.png")
    render(sc, out)
    print(f"wrote {out}")
    print(f"summary: {sc['summary']}")
