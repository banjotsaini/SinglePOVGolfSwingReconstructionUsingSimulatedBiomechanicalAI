"""Shared builders for the JSON artifacts deploy/web/app.js consumes.

Extracted from build_web_assets.py (which bakes the demo clips) so the SAME
builders serve the real upload path: pipeline.py emits <stem>_replay_3d.json
next to its other outputs, and the processing Lambda runs this module as a CLI
step to emit <stem>_metrics.json + <stem>_explanation.json from the scorecard
(LLM-augmented by coaching_explain.py when the eval ran). Shapes match exactly
what app.js / replay3d.js already read — no UI changes needed.

CLI (the processing handler's post-scorecard step):
    python web_artifacts.py --scorecard <path>_scorecard.json --out-dir <dir> --stem <stem>
        [--parquet-3d <3d.parquet> --fps 30]   # backstop: also emit <stem>_replay_3d.json
                                               # if pipeline.py didn't already
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parent.parent

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


def load_kb(path: Path | str | None = None) -> dict:
    p = Path(path) if path else ROOT / "Data" / "coaching" / "indicator_kb.json"
    return json.loads(p.read_text(encoding="utf-8"))


def build_replay_json(parquet_3d: Path, out: Path, fps: float) -> None:
    """COCO-named raw 3D parquet -> replay_3d.json (h36m chain + clubhead).

    The cache holds RAW lifted 3D (events need raw); smooth for the replay so
    the viewer isn't jittery — same tuned One-Euro the coaching path uses.
    """
    import numpy as np
    import pandas as pd
    from smoothing import one_euro_filter, level_and_ground

    df = pd.read_parquet(parquet_3d)
    frames = []

    kp_names = sorted(df["kp_name"].unique())
    wide = df.pivot_table(index="frame", columns="kp_name", values=["x", "y", "z"],
                          sort=True)
    arr = np.stack([wide[ax].to_numpy(dtype=np.float32)[:, [kp_names.index(k) for k in kp_names]]
                    for ax in ("x", "y", "z")], axis=-1)  # (T, K, 3)
    arr = one_euro_filter(arr, fps=fps, min_cutoff=0.3, beta=0.4)
    # level the stance (undo camera pitch/roll) and pin the feet to y=0 so the
    # viewers can place the floor plane exactly instead of guessing from the
    # lowest joint (which floated one foot on ~59% of clips)
    arr, _lvl = level_and_ground(arr,
                                 l_ankle=kp_names.index("left_ankle"),
                                 r_ankle=kp_names.index("right_ankle"))
    smoothed = {(int(f), k): arr[i, j] for i, f in enumerate(wide.index)
                for j, k in enumerate(kp_names)}

    def mid(a, b):
        return tuple(round((a[i] + b[i]) / 2, 4) for i in range(3))

    for f, g in df.groupby("frame", sort=True):
        by = {r.kp_name: tuple(round(float(v), 4) for v in smoothed[(int(f), r.kp_name)])
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
        # stance leveled + feet pinned: the floor is exactly the y=0 plane.
        # Viewers use this instead of their min-joint heuristics; absence of
        # these keys falls back to the old behavior (pre-grounding assets).
        "grounded": True,
        "floor_y": 0.0,
        "leveled_tilt_deg": round(_lvl.get("tilt_deg", 0.0), 2),
        "joint_names": REPLAY_JOINTS + ["clubhead_est"],
        "bones": BONES,
        "frames": frames,
    }, separators=(",", ":")), encoding="utf-8")


def build_metrics_json(sc: dict, out: Path, clip_id: str, kb: dict) -> None:
    cards = kb["indicators"]
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
            "why": card.get("why", ""),
            "confidence": ind.get("confidence_tier"),
        })
    # front end shows 9: highest-confidence first, keep every "watch" story
    tier = {"high": 0, "med": 1, "low": 2}
    metrics.sort(key=lambda m: (tier.get(m["confidence"], 3), m["status"] == "good"))
    out.write_text(json.dumps({"clip_id": str(clip_id),
                               "events": sc.get("events", {}),
                               "metrics": metrics[:9]}, indent=1),
                   encoding="utf-8")


def build_explanation_json(sc: dict, out: Path, clip_id: str, kb: dict) -> None:
    cards = kb["indicators"]
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
    # the gated LLM explanation (coaching_explain.py writes it into the
    # scorecard) — include only when its grounding verification passed
    llm = (sc.get("llm_explanation") or "").strip()
    if llm and sc.get("llm_grounding", {}).get("grounded"):
        sections.append({"title": "Coach's read", "tag": "info", "body": llm})
    sections.append({
        "title": "How to read this", "tag": "info",
        "body": "Every number here is measured from your video against a 400-swing tour "
                "reference — nothing is estimated by the AI. Ask the coach about any "
                "metric for the full story.",
    })
    out.write_text(json.dumps({"clip_id": str(clip_id), "headline": headline,
                               "sections": sections}, indent=1), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scorecard", required=True, help="path to <stem>_scorecard.json")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--stem", required=True)
    p.add_argument("--kb", default=None, help="override indicator_kb.json path")
    p.add_argument("--parquet-3d", default=None,
                   help="raw 3D parquet — emit <stem>_replay_3d.json if not already present")
    p.add_argument("--fps", type=float, default=None,
                   help="replay fps for the --parquet-3d backstop "
                        "(default: <stem>_mocap.json's fps, else 30)")
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    sc = json.loads(Path(args.scorecard).read_text(encoding="utf-8"))
    kb = load_kb(args.kb)

    metrics_path = out_dir / f"{args.stem}_metrics.json"
    build_metrics_json(sc, metrics_path, args.stem, kb)
    print(f"[web] wrote {metrics_path.name}")
    expl_path = out_dir / f"{args.stem}_explanation.json"
    build_explanation_json(sc, expl_path, args.stem, kb)
    print(f"[web] wrote {expl_path.name}")

    if args.parquet_3d:
        replay_path = out_dir / f"{args.stem}_replay_3d.json"
        if replay_path.exists():
            print(f"[web] {replay_path.name} already present (pipeline emitted it) — skipping")
        else:
            fps = args.fps
            if fps is None:  # pipeline's canonical JSON records the real video fps
                mocap = out_dir / f"{args.stem}_mocap.json"
                if mocap.exists():
                    fps = json.loads(mocap.read_text(encoding="utf-8")).get("fps")
                fps = float(fps or 30.0)
            print(f"[web] {replay_path.name} missing — building from parquet at {fps:.2f} fps")
            build_replay_json(Path(args.parquet_3d), replay_path, fps=fps)
            print(f"[web] wrote {replay_path.name}")


if __name__ == "__main__":
    main()
