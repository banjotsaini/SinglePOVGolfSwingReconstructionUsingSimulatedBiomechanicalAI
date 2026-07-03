"""Assemble the end-to-end baseline-vs-improved HTML panel for one clip.

Stitches: comparison video + both scorecards + indicator delta table + coaching
analysis text, framed by the Vicon-measured accuracy context. Self-contained HTML
in outputs/pipeline_compare/ (references the co-located mp4 + scorecard PNGs).
"""
from __future__ import annotations
import sys, json
from pathlib import Path

ROOT = Path(__file__).parent.parent
PC = ROOT / "outputs" / "pipeline_compare"
SC = PC / "scorecards"


def load_sc(tag, clip):
    return json.loads((SC / f"{clip}_{tag}_scorecard.json").read_text())


def fb_html(sc):
    fb = sc.get("feedback", [])
    if not fb:
        return '<p class="ok">No indicators outside the typical tour range.</p>'
    rows = []
    for f in fb:
        msg = f.get("message") or f.get("label", "")
        conf = f.get("confidence_tier", "")
        rows.append(f'<li><b>{f.get("label","")}</b> ({f.get("event","")}, conf={conf}): '
                    f'you {f.get("value","")} vs tour {f.get("pro_median","")} '
                    f'(p{f.get("percentile","")})<br><span class="msg">{msg}</span></li>')
    return "<ul>" + "".join(rows) + "</ul>"


def indicator_table(b, i):
    bi, ii = b.get("indicators", {}), i.get("indicators", {})
    names = [n for n in bi if n in ii]

    def val(d):
        return d.get("value") if isinstance(d, dict) else d

    def med(d):
        return d.get("pro_median", "") if isinstance(d, dict) else ""
    rows = []
    for n in names:
        bv, iv = val(bi[n]), val(ii[n])
        label = bi[n].get("label", n) if isinstance(bi[n], dict) else n
        try:
            delta = f"{float(iv) - float(bv):+.2f}"
        except Exception:
            delta = ""
        rows.append(f"<tr><td>{label}</td><td>{bv}</td><td>{iv}</td>"
                    f"<td>{delta}</td><td>{med(bi[n])}</td></tr>")
    return ("<table><thead><tr><th>Measurement</th><th>Baseline<br>(MotionBERT)</th>"
            "<th>Improved<br>(MixSTE+1€)</th><th>&Delta;</th><th>Tour median</th></tr></thead>"
            "<tbody>" + "".join(rows) + "</tbody></table>")


def main(clip=1292):
    b, i = load_sc("baseline", clip), load_sc("improved", clip)
    meta = b.get("meta", {})
    who = f'{meta.get("player","")} &middot; {meta.get("club","")} &middot; {meta.get("view","")}'
    video = f"{clip}_pipeline_compare.mp4"

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Pipeline comparison — clip {clip}</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; background:#111; color:#e8e8e8;
         max-width:1180px; margin:0 auto; padding:24px; line-height:1.5; }}
  h1 {{ margin:0 0 2px; font-size:24px; }} h2 {{ border-bottom:1px solid #333; padding-bottom:6px; margin-top:34px; }}
  .sub {{ color:#9aa0a6; margin:0 0 18px; }}
  .flow {{ background:#1b1b1b; border:1px solid #333; border-radius:8px; padding:12px 16px; font-size:14px; color:#bbb; }}
  .flow b {{ color:#7fd17f; }}
  .ctx {{ background:#16231a; border:1px solid #2a5; border-radius:8px; padding:12px 16px; margin:14px 0; font-size:14px; }}
  .ctx b {{ color:#8fe88f; }}
  .grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
  .card {{ background:#1b1b1b; border:1px solid #333; border-radius:8px; padding:14px; }}
  .base h3 {{ color:#8a8aff; }} .impr h3 {{ color:#7fd17f; }}
  h3 {{ margin:0 0 8px; font-size:15px; }}
  video, img {{ width:100%; border-radius:8px; background:#000; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th, td {{ padding:5px 8px; border-bottom:1px solid #2a2a2a; text-align:right; }}
  th:first-child, td:first-child {{ text-align:left; }}
  thead th {{ color:#9aa0a6; font-weight:600; }}
  ul {{ margin:6px 0; padding-left:18px; }} li {{ margin-bottom:8px; }}
  .msg {{ color:#cdb; }} .ok {{ color:#7fd17f; }}
  .note {{ color:#c9a227; font-size:13px; }}
  .legend span {{ display:inline-block; margin-right:16px; }}
  .dot-b {{ color:#8a8aff; }} .dot-i {{ color:#7fd17f; }}
</style></head><body>

<h1>End-to-end pipeline comparison — clip {clip}</h1>
<p class="sub">{who}</p>

<div class="flow">Pipeline: <b>Raw video</b> &rarr; <b>2D pose</b> (MediaPipe) &rarr;
<b>3D lift</b> &rarr; <b>smoothing</b> &rarr; <b>coaching scorecard</b> &rarr; <b>analysis</b>.
The two tracks diverge at the 3D-lift + smoothing stages.</div>

<div class="ctx">
<b>What "improved" means</b> — measured against GolfPose Vicon ground truth (held-out G5/G6):
the current app's lifter (MotionBERT-Full) has <b>7.59&deg;</b> mean coaching-measurement error on clean input;
the golf-fine-tuned <b>GolfPose MixSTE</b> has <b>2.83&deg;</b> (&minus;63%). Adding
<b>One-Euro</b> temporal smoothing recovers a further <b>33&ndash;46%</b> of the error that 2D
detection noise injects. Baseline = MotionBERT, no smoothing. Improved = MixSTE + One-Euro.
<span class="legend"><br><span class="dot-b">&#9679; baseline</span><span class="dot-i">&#9679; improved</span></span>
</div>

<h2>1&ndash;3 &middot; Raw &rarr; 2D &rarr; 3D (video)</h2>
<p class="sub">Top row: shared inputs (raw + 2D). Bottom row: the 3D divergence — jittery/compressed
MotionBERT (red) vs cleaner MixSTE+One-Euro (green).</p>
<video controls loop muted playsinline src="{video}"></video>

<h2>4 &middot; Coaching scorecard</h2>
<p class="note">Per-clip caveat: GolfDB clips have <b>no 3D ground truth</b>, so for this one swing we
can't declare either number "correct" — the accuracy claim is the Vicon <i>average</i> (above).
Monocular 3D compresses depth-rotation, so absolute turn angles read low for <i>both</i> lifters;
the scorecard reads them <i>relative</i> to a pro band built from the same pipeline. Expect the two
lifters to disagree most on the depth-dependent rotation metrics.</p>
<div class="grid2">
  <div class="card base"><h3>BASELINE — MotionBERT, raw</h3>
     <img src="scorecards/{clip}_baseline_scorecard.png" alt="baseline scorecard"></div>
  <div class="card impr"><h3>IMPROVED — MixSTE + One-Euro</h3>
     <img src="scorecards/{clip}_improved_scorecard.png" alt="improved scorecard"></div>
</div>

<h3 style="margin-top:18px">Measurement-by-measurement</h3>
{indicator_table(b, i)}

<h2>5 &middot; Coaching analysis</h2>
<p class="note">LLM rewriter is offline in this environment (no API key) — shown below is the
structured scorecard feedback that the LLM layer turns into prose. Same input, both tracks.</p>
<div class="grid2">
  <div class="card base"><h3>BASELINE analysis</h3><p>{b.get("summary","")}</p>{fb_html(b)}</div>
  <div class="card impr"><h3>IMPROVED analysis</h3><p>{i.get("summary","")}</p>{fb_html(i)}</div>
</div>

<p class="sub" style="margin-top:28px">Generated by <code>Scripts/build_compare_panel.py</code>.
Video: <code>Scripts/render_pipeline_compare.py</code> &middot; Scorecards:
<code>Scripts/make_compare_scorecards.py</code> &middot; Accuracy basis:
<code>outputs/coaching_accuracy/FINDINGS.md</code>.</p>
</body></html>"""

    out = PC / f"{clip}_compare.html"
    out.write_text(html, encoding="utf-8")
    print(f"[+] wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1292)
