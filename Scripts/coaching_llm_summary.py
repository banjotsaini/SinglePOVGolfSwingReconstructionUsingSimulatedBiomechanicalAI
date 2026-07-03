"""LLM Interpretation Layer — turn the structured scorecard into a
beginner-friendly, plain-English swing explanation (slide 5 of the deck).

This is the "LLM explains the swing" layer the MVP is built around. It does
NOT re-analyze video — it takes our already-computed, reliability-gated
metrics and feedback and asks the LLM to explain them in encouraging,
jargon-free language. Keeps the system transparent (the LLM only narrates
numbers we measured; it never invents observations).

Uses the Codex CLI (already authed) the same way as the benchmark.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
CODEX_BIN = shutil.which("codex.cmd") or shutil.which("codex") or "codex"

TASK = """TASK: Below is MEASURED motion data from one golfer's swing video (already
computed — do not re-analyze anything). Write a short beginner-friendly
explanation of what their body did during the swing, comparing to the tour-pro
range. This is a one-shot writing task — produce the explanation text only, no
questions back, no preamble.

Strict rules:
  - 2-3 short paragraphs, warm and plain. Gloss any golf term in a few words.
  - Lead with what they did WELL (metrics inside the tour range).
  - Then describe the 1-2 flagged observations as neutral descriptions of what
    their body did — never as faults, never as instructions/tips to fix.
  - NEVER mention any metric tagged confidence=low; we don't trust that number.
  - PRECISION: only call something "in range"/"tour-like" if THAT exact metric's
    in_tour_range=True. Never lump several metrics under one label (e.g. "head
    movement", "spine tilt", "knee position") and call the group in range — if
    ANY listed metric in that group has in_tour_range=False, do not describe it
    as in range. When unsure, say nothing about that metric.
  - Don't add phase/timing details (e.g. "during the backswing") unless the
    metric name itself says so (e.g. *_top vs *_impact vs *_address).
  - Invent nothing not in the data. End with one encouraging sentence.

OUTPUT: only the explanation paragraphs.
"""


def build_prompt(scorecard: dict) -> str:
    inds = scorecard["indicators"]
    fb = scorecard["feedback"]
    lines = [TASK, "", "MEASURED METRICS (you vs tour-pro median, with confidence):"]
    for k, v in inds.items():
        conf = v.get("confidence_tier", "med")
        in_band = v["pro_band"][0] <= v["value"] <= v["pro_band"][1]
        lines.append(f"  - {k}: you={v['value']}  tour_median={v['pro_median']}  "
                     f"percentile={v['percentile']}  in_tour_range={in_band}  confidence={conf}")
    lines.append("\nFLAGGED OBSERVATIONS:")
    real = [f for f in fb if f.get("severity") == "review"]
    if not real:
        lines.append("  (none — everything confidently measured is within the tour range)")
    for f in real:
        lines.append(f"  - {f['label']} at {f['event']}: {f['message']} (you p{f['percentile']})")
    lines.append("\nNow write the explanation (paragraphs only):")
    return "\n".join(lines)


_CALL_SEQ = 0
_CALL_LOCK = __import__("threading").Lock()


def call_codex(prompt: str, timeout: int = 180, out_file=None) -> str | None:
    # Unique output file per call so concurrent callers don't clobber each
    # other's '-o' file. Pass out_file explicitly to control it.
    if out_file is None:
        global _CALL_SEQ
        with _CALL_LOCK:
            _CALL_SEQ += 1
            seq = _CALL_SEQ
        out_file = PROJECT_ROOT / "Data" / "coaching" / f"_llm_out_{os.getpid()}_{seq}.txt"
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Pipe the (multi-line) prompt via stdin with '-'; passing it as a CLI
        # arg mangles newlines on Windows. Force UTF-8 I/O.
        cp = subprocess.run(
            [CODEX_BIN, "exec", "--sandbox", "read-only", "-o", str(out_file), "-"],
            input=prompt, timeout=timeout, capture_output=True,
            text=True, encoding="utf-8", errors="replace", shell=False,
        )
        if cp.returncode != 0:
            print(f"  codex exit {cp.returncode}: {(cp.stderr or '')[:200]}")
            return None
        if out_file.exists():
            txt = out_file.read_text(encoding="utf-8", errors="replace").strip()
            try:
                out_file.unlink()  # don't litter the dir with per-call temp files
            except OSError:
                pass
            return txt
        return (cp.stdout or "").strip()
    except Exception as e:
        print(f"  codex error: {repr(e)[:160]}")
        return None


def summarize(scorecard_json: Path) -> str | None:
    sc = json.loads(Path(scorecard_json).read_text())
    summary = call_codex(build_prompt(sc))
    if summary:
        sc["llm_summary"] = summary
        Path(scorecard_json).write_text(json.dumps(sc, indent=2))
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scorecard", required=True, help="Path to a *_scorecard.json")
    args = p.parse_args()
    s = summarize(Path(args.scorecard))
    print("\n" + "=" * 60)
    print("LLM SWING EXPLANATION")
    print("=" * 60)
    print(s or "(failed)")
