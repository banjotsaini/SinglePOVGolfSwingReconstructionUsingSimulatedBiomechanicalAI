"""Build real deploy/web/assets/<clip>/ bundles from pipeline outputs.

Replaces Austin's hand-made placeholder assets (Phase 2 of the deployment plan):
for a GolfDB clip with cached pose + a scorecard, emit the five files the front
end consumes — raw.mp4, overlay.mp4 (H.264/yuv420p, the codec Chrome needs),
metrics.json, explanation.json, replay_3d.json — in exactly the shapes app.js /
replay3d.js already read, so no UI changes are needed.

Usage:
  python build_web_assets.py 417 886          # needs Data/demo/clip<id>/ scorecard
Inputs per clip:
  Data/videos_160/<id>.mp4                                 raw video
  Data/eval_runs/mediapipe_lite/<id>.parquet               cached 2D pose
  Data/eval_runs/golfpose3d_from_mediapipe_lite/<id>.parquet  cached 3D pose
  Data/demo/clip<id>/clip<id>_scorecard.json               scorecard (demo.py fast path)
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "deploy" / "web" / "assets"
KB = json.loads((ROOT / "Data" / "coaching" / "indicator_kb.json").read_text(encoding="utf-8"))

# h36m joint order the front end's replay3d.js expects (17 + estimated clubhead)
REPLAY_JOINTS = ["hip_center", "right_hip", "right_knee", "right_ankle", "left_hip",
                 "left_knee", "left_ankle", "spine", "thorax", "neck", "head",
                 "left_shoulder", "left_elbow", "left_wrist", "right_shoulder",
                 "right_elbow", "right_wrist"]
BONES = [{"a":0,"b":7,"color":"#E0314B"},{"a":7,"b":8,"color":"#E0314B"},
         {"a":8,"b":9,"color":"#E0314B"},{"a":9,"b":10,"color":"#E0314B"},
         {"a":8,"b":11,"color":"#17B6C4"},{"a":8,"b":14,"color":"#17B6C4"},
         {"a":11,"b":12,"color":"#19B6C4"},{"a":12,"b":13,"color":"#19B6C4"},
         {"a":14,"b":15,"color":"#2FB36B"},{"a":15,"b":16,"color":"#2FB36B"},
         {"a":0,"b":4,"color":"#6D5BD0"},{"a":0,"b":1,"color":"#6D5BD0"},
         {"a":4,"b":5,"color":"#37C26B"},{"a":5,"b":6,"color":"#37C26B"},
         {"a":1,"b":2,"color":"#F0A12E"},{"a":2,"b":3,"color":"#F0A12E"},
         {"a":13,"b":17,"color":"#9AA3AE"},{"a":16,"b":17,"color":"#9AA3AE"}]


def ffmpeg() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def h264(src: Path, dst: Path) -> None:
    """Re-encode to H.264/yuv420p — the only combo Chrome reliably plays."""
    subprocess.run([ffmpeg(), "-y", "-i", str(src), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an",
                    str(dst)], check=True, capture_output=True)


def build_overlay(clip: int, out: Path) -> None:
    sys.path.insert(0, str(ROOT / "Scripts"))
    from pipeline import write_2d_overlay
    video = ROOT / "Data" / "videos_160" / f"{clip}.mp4"
    pq2d = ROOT / "Data" / "eval_runs" / "mediapipe_lite" / f"{clip}.parquet"
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "overlay_raw.mp4"
        write_2d_overlay(video, pq2d, tmp)          # writes mp4v — not browser-safe
        h264(tmp, out)


def build_replay(clip: int, out: Path, fps: float) -> None:
    pq = ROOT / "Data" / "eval_runs" / "golfpose3d_from_mediapipe_lite" / f"{clip}.parquet"
    df = pd.read_parquet(pq)
    frames = []

    def mid(a, b):
        return tuple(round((a[i] + b[i]) / 2, 4) for i in range(3))

    for _, g in df.groupby("frame", sort=True):
        by = {r.kp_name: (round(float(r.x), 4), round(float(r.y), 4), round(float(r.z), 4))
              for r in g.itertuples()}
        # cache is COCO-named — derive the h36m torso chain the front end expects
        by["hip_center"] = mid(by["left_hip"], by["right_hip"])
        by["thorax"] = mid(by["left_shoulder"], by["right_shoulder"])
        by["spine"] = mid(by["hip_center"], by["thorax"])
        by["head"] = mid(by["left_ear"], by["right_ear"]) if "left_ear" in by else by["nose"]
        by["neck"] = mid(by["thorax"], by["head"])
        pose = [list(by[j]) for j in REPLAY_JOINTS]
        # clubhead: extrapolate the lead (left) forearm past the wrist ~2.2x
        ex, ey, ez = by["left_elbow"]; wx, wy, wz = by["left_wrist"]
        pose.append([round(wx + (wx-ex)*2.2, 4), round(wy + (wy-ey)*2.2, 4),
                     round(wz + (wz-ez)*2.2, 4)])
        frames.append(pose)
    out.write_text(json.dumps({
        "fps": fps,
        "axis_convention": "h36m-camera (x right, y DOWN, z depth)",
        "joint_names": REPLAY_JOINTS + ["clubhead_est"],
        "bones": BONES,
        "frames": frames,
    }, separators=(",", ":")), encoding="utf-8")


def build_metrics(clip: int, sc: dict, out: Path) -> None:
    cards = KB["indicators"]
    metrics = []
    for key, ind in sc["indicators"].items():
        card = cards.get(key, {})
        val, band = ind.get("value"), ind.get("pro_band")
        if val is None or not band:
            continue
        in_band = band[0] <= val <= band[1]
        unit = card.get("unit", "")
        disp = (lambda v: f"{v:.2f} : 1" if key == "tempo_ratio" else
                (f"{v:.1f}°" if unit == "degrees" else f"{v:.1f}"))
        span = band[1] - band[0]
        metrics.append({
            "key": key,
            "label": card.get("label", key),
            "you": round(val, 2), "you_display": disp(val),
            "tour": ind.get("pro_median"), "tour_display": disp(ind["pro_median"]),
            "axis": [round(min(band[0], val) - span * .35, 2),
                     round(max(band[1], val) + span * .35, 2)],
            "band": band,
            "status": "good" if in_band else "watch",
            "blurb": card.get("plain_name", ""),
            "confidence": ind.get("confidence_tier"),
        })
    # front end shows 9: highest-confidence first, keep every "watch" story
    tier = {"high": 0, "med": 1, "low": 2}
    metrics.sort(key=lambda m: (tier.get(m["confidence"], 3), m["status"] == "good"))
    out.write_text(json.dumps({"clip_id": str(clip), "metrics": metrics[:9]}, indent=1),
                   encoding="utf-8")


def build_explanation(clip: int, sc: dict, out: Path) -> None:
    cards = KB["indicators"]
    goods, watches = [], []
    for key, ind in sc["indicators"].items():
        card = cards.get(key, {})
        val, band = ind.get("value"), ind.get("pro_band")
        if val is None or not band or ind.get("confidence_tier") == "low":
            continue
        name = card.get("plain_name", key)
        if band[0] <= val <= band[1]:
            goods.append(name)
        else:
            side = "more than" if val > band[1] else "less than"
            watches.append(f"Your {name} measured {side} most tour swings.")
    n_notes = len(sc.get("feedback") or [])
    headline = ("A tour-shaped swing — the measurements mostly sit inside the pro bands."
                if n_notes <= 1 else
                "Solid foundations with a few clear places to look next.")
    sections = [{
        "title": "What lined up well", "tag": "good",
        "body": "Inside the typical tour spread: " + ", ".join(goods[:6]) + ".",
    }]
    if watches:
        sections.append({"title": "Worth a look", "tag": "watch",
                         "body": " ".join(watches[:4])})
    sections.append({
        "title": "How to read this", "tag": "info",
        "body": "Every number here is measured from your video against a 400-swing tour "
                "reference — nothing is estimated by the AI. Ask the coach about any "
                "metric for the full story.",
    })
    out.write_text(json.dumps({"clip_id": str(clip), "headline": headline,
                               "sections": sections}, indent=1), encoding="utf-8")


def build(clip: int) -> None:
    scp = ROOT / "Data" / "demo" / f"clip{clip}" / f"clip{clip}_scorecard.json"
    sc = json.loads(scp.read_text(encoding="utf-8"))
    dst = WEB / str(clip)
    dst.mkdir(parents=True, exist_ok=True)
    video = ROOT / "Data" / "videos_160" / f"{clip}.mp4"
    cap = cv2.VideoCapture(str(video)); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0; cap.release()

    h264(video, dst / "raw.mp4")
    print(f"[{clip}] raw.mp4")
    build_overlay(clip, dst / "overlay.mp4")
    print(f"[{clip}] overlay.mp4 (H.264)")
    build_replay(clip, dst / "replay_3d.json", fps)
    print(f"[{clip}] replay_3d.json")
    build_metrics(clip, sc, dst / "metrics.json")
    print(f"[{clip}] metrics.json")
    build_explanation(clip, sc, dst / "explanation.json")
    print(f"[{clip}] explanation.json")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        build(int(arg))
    print("done")
