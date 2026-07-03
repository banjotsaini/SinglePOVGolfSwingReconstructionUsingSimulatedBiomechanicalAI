"""Build per-clip swing-frame montages for the MULTIMODAL coaching feature.

For each scorecard in a split, pull 4 key event frames (address, top, impact,
finish) from the matching GolfDB clip video and stitch them into one labeled
montage PNG. The multimodal feature feeds this image alongside the metrics so
the model can visually corroborate posture/positions — the grounding harness
then catches any visual claim not backed by the measured metrics.

No torch, no LLM. Frame indices come from the scorecard's "events" (same frame
space as the cached pose/parquet the events were detected in, which is the
video's own frame indexing).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).parent.parent
COACH = PROJECT_ROOT / "Data" / "coaching"
VIDEOS = PROJECT_ROOT / "Data" / "videos_160"
SPLIT_DIRS = {"dev": COACH / "dev_scorecards", "test": COACH / "test_scorecards"}
PANELS = ["address", "top", "impact", "finish"]   # the 4 frames we show
CELL = 224  # upscale each 160px frame for legibility


def montage_for(clip_id: int, events: dict) -> Image.Image | None:
    vid = VIDEOS / f"{clip_id}.mp4"
    if not vid.exists():
        return None
    cap = cv2.VideoCapture(str(vid))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    cells = []
    for name in PANELS:
        idx = int(events.get(name, 0))
        if total:
            idx = max(0, min(idx, total - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).resize((CELL, CELL))
        d = ImageDraw.Draw(img)
        label = f"{name} (f={idx})"
        d.rectangle([0, 0, len(label) * 7 + 6, 16], fill=(0, 0, 0))
        d.text((3, 3), label, fill=(255, 255, 255))
        cells.append(img)
    cap.release()
    out = Image.new("RGB", (CELL * len(cells), CELL), (20, 20, 20))
    for i, c in enumerate(cells):
        out.paste(c, (i * CELL, 0))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["dev", "test"], required=True)
    p.add_argument("--n", type=int, default=None)
    args = p.parse_args()
    import json

    out_dir = COACH / f"{args.split}_frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = sorted(int(q.stem) for q in SPLIT_DIRS[args.split].glob("*.json") if q.stem.isdigit())
    if args.n:
        ids = ids[: args.n]
    ok = miss = 0
    for cid in ids:
        dest = out_dir / f"{cid}.png"
        if dest.exists():
            ok += 1; continue
        sc = json.loads((SPLIT_DIRS[args.split] / f"{cid}.json").read_text(encoding="utf-8"))
        m = montage_for(cid, sc.get("events", {}))
        if m is None:
            miss += 1; continue
        m.save(dest)
        ok += 1
        if ok % 25 == 0:
            print(f"  {args.split}: {ok}/{len(ids)}", flush=True)
    print(f"[frames] {args.split}: wrote/had {ok}, missing video {miss} -> {out_dir}")


if __name__ == "__main__":
    main()
