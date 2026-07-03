"""Slide diagrams matching the Week 7a deck's MVP framing.

  diagram_mvp_pipeline.excalidraw   — the MVP flow (slide 12 / 2) with REAL
      components + the measured 0.865 detector
  panel_metrics_to_llm.excalidraw   — motion metrics -> scorecard -> LLM
      explanation (slides 4-5 tie-together)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import excalidraw_builder as ex

OUT_DIR = Path(__file__).parent.parent / "Data" / "visualizations" / "excalidraw"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GOLD="#f0a847"; PALE="#fde2b3"; GREEN="#3a7c50"; LGREEN="#cce8d4"
INDIGO="#5b6cdb"; LIND="#dde0fa"; PURPLE="#9b59b6"; LPURP="#e8d5f0"
TEAL="#1ba39c"; GRAY="#5c6370"; DARK="#1e1e1e"


def mvp_pipeline():
    els = []
    els.append(ex.text(60, 30, "Technical Pipeline — Single-POV Video to Swing Explanation",
                        font_size=34, stroke_color=DARK, font_family=1, width=1500))
    els.append(ex.text(60, 78, "Every stage is the measured-best component from our 24-pipeline benchmark.",
                        font_size=16, stroke_color=GRAY, font_family=1, width=1400))

    stages = [
        ("Single-POV\nVideo", "phone or GolfDB", "#ffffff", GRAY),
        ("2D Pose", "MediaPipe Lite\n107 FPS · CPU", "#ffffff", INDIGO),
        ("3D Pose", "MotionBERT-Full\n6,900 FPS · GPU", "#ffffff", GOLD),
        ("Smoothing", "One-Euro + bone-lock\n-51% jitter", "#ffffff", GOLD),
        ("Swing Phases", "trained 1D-CNN\nPCE@5 = 0.865", LGREEN, GREEN),
        ("Motion Metrics", "15 indicators vs\ntour-pro band", LIND, INDIGO),
        ("LLM Explanation", "beginner-friendly\nswing summary", LPURP, PURPLE),
    ]
    n = len(stages)
    x0, y, bw, bh, gap = 60, 230, 230, 150, 18
    ids = []
    for i,(t,s,bg,c) in enumerate(stages):
        x = x0 + i*(bw+gap)
        box = ex.rect(x, y, bw, bh, text=f"{t}\n\n{s}", stroke_color=c,
                      background_color=bg, stroke_width=3 if i in (4,) else 2,
                      font_size=15, text_color=c)
        els.extend(box); ids.append(box[0]["id"])
        if i < n-1:
            els.append(ex.arrow(x+bw+2, y+bh/2, x+bw+gap-2, y+bh/2,
                                 stroke_color=DARK, start_id=ids[i]))

    # outputs row
    els.append(ex.text(60, 430, "OUTPUTS per swing:", font_size=18,
                        stroke_color=DARK, font_family=1, width=400))
    outs = [
        ("Coaching Scorecard", "PNG + JSON · the user-facing deliverable", GREEN, LGREEN),
        ("UE5 Handoff", "CSV / BVH / 3D viewer · Theo's render layer", GOLD, PALE),
        ("LLM Summary", "plain-English explanation · the MVP pitch", PURPLE, LPURP),
    ]
    ow = 520; ox = 60
    for i,(t,s,c,bg) in enumerate(outs):
        x = ox + i*(ow+30)
        els.extend(ex.rect(x, 480, ow, 90, text=f"{t}\n{s}", stroke_color=c,
                            background_color=bg, stroke_width=2.5, font_size=15, text_color=c))

    els.append(ex.text(60, 600,
        "One command:  python demo.py swing.mp4   →   all three outputs.  Fully working end-to-end today.",
        font_size=15, stroke_color=GREEN, font_family=1, width=1600))
    ex.save(els, str(OUT_DIR / "diagram_mvp_pipeline.excalidraw"))
    return OUT_DIR / "diagram_mvp_pipeline.excalidraw"


def metrics_to_llm():
    els = []
    els.append(ex.text(60, 30, "Motion Metrics → Scorecard → LLM Explanation",
                        font_size=32, stroke_color=DARK, font_family=1, width=1400))
    els.append(ex.text(60, 76, "The evidence layer (measured) feeds the interpretation layer (explained). Transparent end-to-end.",
                        font_size=15, stroke_color=GRAY, font_family=1, width=1500))

    # left: metric families
    box1 = ex.rect(60, 160, 520, 560, stroke_color=INDIGO, background_color=LIND, stroke_width=3)
    els.extend(box1)
    els.append(ex.text(85, 185, "1 · MOTION METRICS (measured)", font_size=18,
                        stroke_color=INDIGO, font_family=1, width=480))
    fams = [
        "Movement: head sway, lift, weight shift",
        "Angles: shoulder turn, hip turn, spine lean",
        "Separation: X-factor (shoulder-hip coil)",
        "Change over time: address → impact",
        "Timing: backswing : downswing tempo",
        "Confidence: per-metric reliability tier",
        "   (from Austin's depth-reliability EDA)",
    ]
    for i,f in enumerate(fams):
        els.append(ex.text(90, 240+i*58, "• "+f, font_size=15, stroke_color=DARK,
                            font_family=1, width=470))
    els.append(ex.text(90, 660, "15 indicators · each vs the 1,400-pro band", font_size=13,
                        stroke_color=GRAY, font_family=1, width=470))

    # middle: scorecard (reference the rendered PNG conceptually)
    box2 = ex.rect(620, 160, 480, 560, stroke_color=GREEN, background_color=LGREEN, stroke_width=3)
    els.extend(box2)
    els.append(ex.text(645, 185, "2 · SCORECARD (you vs tour)", font_size=18,
                        stroke_color=GREEN, font_family=1, width=440))
    rows = [
        ("Tempo", "3.4:1", "in range", True),
        ("Shoulder turn", "p30", "in range", True),
        ("X-factor (coil)", "p51", "in range", True),
        ("Hip rotation", "p92", "in range", True),
        ("Weight shift", "p5", "FLAGGED", False),
        ("Lead-arm bend", "low-conf", "suppressed", None),
    ]
    for i,(lbl,val,status,ok) in enumerate(rows):
        c = GREEN if ok else (GOLD if ok is False else GRAY)
        els.append(ex.text(650, 250+i*62, f"{lbl}", font_size=15, stroke_color=DARK, font_family=1, width=230))
        els.append(ex.text(900, 250+i*62, f"{val} · {status}", font_size=13, stroke_color=c, font_family=1, width=190))
    els.append(ex.text(650, 660, "sparse · transparent · reliability-gated", font_size=13,
                        stroke_color=GRAY, font_family=1, width=440))

    # right: LLM explanation
    box3 = ex.rect(1140, 160, 720, 560, stroke_color=PURPLE, background_color=LPURP, stroke_width=3)
    els.extend(box3)
    els.append(ex.text(1165, 185, "3 · LLM EXPLANATION (beginner-friendly)", font_size=18,
                        stroke_color=PURPLE, font_family=1, width=680))
    quote = ('"Several parts of your swing lined up well with the tour-pro\n'
             'range — your timing was very close to tour level, and at the\n'
             'top your shoulder turn and shoulder-hip separation were both\n'
             'in a normal tour pattern.\n\n'
             'The main difference showed up in how your lower body moved\n'
             'toward the target: your hips opened a lot by impact while\n'
             'sliding very little sideways, so your weight moved less\n'
             'laterally than is typical on tour. There are solid pieces\n'
             'here to build on."')
    els.append(ex.text(1165, 250, quote, font_size=14, stroke_color=DARK, font_family=1, width=680))
    els.append(ex.text(1165, 660, "explains only confident metrics · never prescribes a fix",
                        font_size=13, stroke_color=GRAY, font_family=1, width=680))

    # arrows
    els.append(ex.arrow(585, 440, 615, 440, stroke_color=DARK, start_id=box1[0]["id"], end_id=box2[0]["id"]))
    els.append(ex.arrow(1105, 440, 1135, 440, stroke_color=DARK, start_id=box2[0]["id"], end_id=box3[0]["id"]))

    ex.save(els, str(OUT_DIR / "panel_metrics_to_llm.excalidraw"))
    return OUT_DIR / "panel_metrics_to_llm.excalidraw"


def main():
    import random; random.seed(7)
    for fn in (mvp_pipeline, metrics_to_llm):
        out = fn(); print("wrote", out.name)


if __name__ == "__main__":
    main()
