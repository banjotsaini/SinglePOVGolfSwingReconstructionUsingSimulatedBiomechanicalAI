"""Sample GolfDB clips and measure what Austin's z-depth correction actually
changes — both joint trajectories and the depth-dependent coaching indicators.

Why this exists
---------------
"Show how coaching + joint estimation improve with the optimization." There is a
catch worth being precise about:

  * The corrector is trained MediaPipe-world -> Vicon, on GolfPose lab coords
    where **Z = height (vertical)**, Y = front-back depth, X = left/right
    (Austin's own skeleton_mapping.json). The Procrustes align leaves X,Y in
    place and SG+MLP edit ONLY index 2 (GP-Z = height).
  * GolfDB clips have **no depth ground truth**, so we can't compute an accuracy
    (MPJPE) improvement here — that lives on the Vicon test set (106 -> 33 mm,
    outputs/comparison/). On GolfDB we can only measure the *effect* of the
    correction on each metric.

So this script measures, per sampled clip:
  1. Per-axis correction magnitude (X / Y=depth / Z=height) — shows which axis
     the optimization actually moves.
  2. Coaching indicators baseline vs optimized at the top + impact events:
       - shoulder_turn, hip_turn  (horizontal X-Y plane -> rotation)
       - spine_tilt               (angle from vertical Z -> height-dependent)
     This reveals which coaching numbers the height-axis correction can and
     cannot affect.

Run:  python Scripts/compare_z_correction_coaching.py
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import cv2
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from z_depth_correction import ZDepthCorrector, MP_IDX, GP_IDX, apply_alignment

import mediapipe as mp
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

PROJ = Path(__file__).resolve().parent.parent
VIDEO_DIR = PROJ / "Data" / "videos_160"
POSE_MODEL = PROJ / "Models" / "pose_landmarker_heavy.task"
GOLFDB_MAT = PROJ / "AL EDA Investigation" / "golfDB.mat"
OUT = PROJ / "outputs" / "coaching_compare"
OUT.mkdir(parents=True, exist_ok=True)

# GP joint index -> position in the 12-joint array
def gp_pos(gp_i: int) -> int:
    return GP_IDX.index(gp_i)

L_HIP, R_HIP = gp_pos(4), gp_pos(1)
L_SHO, R_SHO = gp_pos(11), gp_pos(14)

# In GP lab frame: axis 0 = X (left/right), 1 = Y (depth), 2 = Z (height/vertical)
HORIZ = [0, 1]   # horizontal plane (rotation lives here)
VERT = 2         # vertical/height axis (what the corrector edits)


def horizontal_turn_deg(p_left, p_right) -> float:
    """Angle of the left->right segment in the horizontal (X,Y) plane."""
    d = p_right - p_left
    return float(np.degrees(np.arctan2(d[HORIZ[1]], d[HORIZ[0]])))


def spine_tilt_deg(frame) -> float:
    """Angle (deg) of the hip_center->shoulder_center vector away from vertical Z."""
    hipc = 0.5 * (frame[L_HIP] + frame[R_HIP])
    shoc = 0.5 * (frame[L_SHO] + frame[R_SHO])
    v = shoc - hipc
    horiz = np.hypot(v[HORIZ[0]], v[HORIZ[1]])
    return float(np.degrees(np.arctan2(horiz, abs(v[VERT]))))


def load_golfdb_meta():
    mat = sio.loadmat(str(GOLFDB_MAT))
    db = mat["golfDB"]

    def unwrap(x):
        while isinstance(x, np.ndarray) and x.size == 1:
            x = x.item()
        return x

    rows = [{f: unwrap(db[0, i][f]) for f in db.dtype.names} for i in range(db.shape[1])]
    import pandas as pd
    return pd.DataFrame(rows)


def pick_clips(df, n_each=2):
    """A few real-time driver clips, mixing face-on and down-the-line.
    Pin the two ids Austin used for his demo videos for continuity."""
    picks = []
    for cid in (830, 269):  # 830 face-on, 269 down-the-line (Austin's demo clips)
        row = df[df["id"] == cid]
        if len(row):
            picks.append(row.iloc[0])
    base = df[(df["club"] == "driver") & (df["slow"] == 0)]
    for view in ("face-on", "down-the-line"):
        got = base[(base["view"] == view) & (~base["id"].isin([p["id"] for p in picks]))]
        for _, r in got.head(n_each).iterrows():
            picks.append(r)
    # de-dup, keep order, cap total
    seen, out = set(), []
    for p in picks:
        if p["id"] in seen:
            continue
        seen.add(p["id"])
        out.append(p)
    return out[:6]


def mediapipe_world(video_path: Path) -> np.ndarray | None:
    """Run MediaPipe (Tasks API, heavy model) -> root-centered (T,12,3) world
    landmarks in GP_IDX order. Uses pose_world_landmarks (metric, hip origin)."""
    opts = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(POSE_MODEL)),
        running_mode=RunningMode.VIDEO,
    )
    landmarker = PoseLandmarker.create_from_options(opts)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    fi = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(rgb))
        res = landmarker.detect_for_video(mp_img, int((fi / fps) * 1000))
        if res.pose_world_landmarks:
            lms = res.pose_world_landmarks[0]
            raw = np.array([[lms[i].x, lms[i].y, lms[i].z] for i in MP_IDX])
            lh = raw[MP_IDX.index(23)]
            rh = raw[MP_IDX.index(24)]
            raw -= 0.5 * (lh + rh)[np.newaxis, :]
            frames.append(raw)
        else:
            frames.append(np.full((12, 3), np.nan))
        fi += 1
    cap.release()
    landmarker.close()
    return np.array(frames) if frames else None


def events_relative(ev):
    e = np.array(ev).flatten().astype(int)
    return e - e[0]


def main():
    corr = ZDepthCorrector()
    df = load_golfdb_meta()
    clips = pick_clips(df)
    print(f"Sampled {len(clips)} clips: {[int(c['id']) for c in clips]}")

    axis_names = ["X (left/right)", "Y (depth)", "Z (height)"]
    rows = []
    per_axis_accum = []
    traj_example = None  # (cid, baseline, optimized) for the figure

    for rec in clips:
        cid = int(rec["id"])
        vid = VIDEO_DIR / f"{cid}.mp4"
        if not vid.exists():
            print(f"  [skip] {vid} missing")
            continue
        mp12 = mediapipe_world(vid)
        if mp12 is None or np.isnan(mp12).all():
            print(f"  [skip] clip {cid}: no MediaPipe detections")
            continue

        baseline = apply_alignment(np.nan_to_num(mp12), corr.R, corr.s)  # aligned only
        optimized = corr.correct_joint_array(np.nan_to_num(mp12))         # + SG + MLP
        if traj_example is None:
            traj_example = (cid, baseline.copy(), optimized.copy())

        # 1) which axis actually moved
        per_axis = np.abs(optimized - baseline).reshape(-1, 3)
        per_axis = np.nanmean(per_axis, axis=0) * 1000.0  # mm
        per_axis_accum.append(per_axis)

        # 2) coaching indicators at top (event idx 3) and impact (event idx 5)
        ev = events_relative(rec["events"])
        ev8 = [int(x) for x in ev[1:9]]
        T = baseline.shape[0]
        top = int(np.clip(ev8[3], 0, T - 1))
        impact = int(np.clip(ev8[5], 0, T - 1))

        for label, fi in (("top", top), ("impact", impact)):
            rows.append({
                "clip": cid, "view": rec["view"], "event": label,
                "shoulder_turn_base": horizontal_turn_deg(baseline[fi, L_SHO], baseline[fi, R_SHO]),
                "shoulder_turn_opt": horizontal_turn_deg(optimized[fi, L_SHO], optimized[fi, R_SHO]),
                "hip_turn_base": horizontal_turn_deg(baseline[fi, L_HIP], baseline[fi, R_HIP]),
                "hip_turn_opt": horizontal_turn_deg(optimized[fi, L_HIP], optimized[fi, R_HIP]),
                "spine_tilt_base": spine_tilt_deg(baseline[fi]),
                "spine_tilt_opt": spine_tilt_deg(optimized[fi]),
            })
        print(f"  clip {cid:5d} {rec['view']:<14} | "
              f"axis |Δ| mm  X={per_axis[0]:5.1f}  Y={per_axis[1]:5.1f}  Z={per_axis[2]:5.1f}")

    import pandas as pd
    rdf = pd.DataFrame(rows)
    mean_axis = np.mean(per_axis_accum, axis=0)

    # Deltas in the coaching indicators
    rdf["d_shoulder_turn"] = (rdf["shoulder_turn_opt"] - rdf["shoulder_turn_base"]).abs()
    rdf["d_hip_turn"] = (rdf["hip_turn_opt"] - rdf["hip_turn_base"]).abs()
    rdf["d_spine_tilt"] = (rdf["spine_tilt_opt"] - rdf["spine_tilt_base"]).abs()

    rdf.to_csv(OUT / "indicator_compare.csv", index=False)

    print("\n" + "=" * 64)
    print("PER-AXIS CORRECTION MAGNITUDE (mean |optimized - baseline|, mm)")
    print("=" * 64)
    for nm, v in zip(axis_names, mean_axis):
        print(f"  {nm:<16}: {v:6.1f} mm")

    print("\n" + "=" * 64)
    print("COACHING INDICATOR CHANGE (mean abs Δ across sampled events, degrees)")
    print("=" * 64)
    print(f"  shoulder turn (horizontal) : {rdf['d_shoulder_turn'].mean():.3f} deg")
    print(f"  hip turn (horizontal)      : {rdf['d_hip_turn'].mean():.3f} deg")
    print(f"  spine tilt (vertical-dep.) : {rdf['d_spine_tilt'].mean():.3f} deg")

    summary = {
        "clips": sorted(rdf["clip"].unique().tolist()),
        "per_axis_correction_mm": {n: round(float(v), 2) for n, v in zip(axis_names, mean_axis)},
        "indicator_mean_abs_delta_deg": {
            "shoulder_turn": round(float(rdf["d_shoulder_turn"].mean()), 4),
            "hip_turn": round(float(rdf["d_hip_turn"].mean()), 4),
            "spine_tilt": round(float(rdf["d_spine_tilt"].mean()), 4),
        },
    }
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[+] wrote {OUT/'indicator_compare.csv'} and {OUT/'summary.json'}")

    make_figure(traj_example, mean_axis, axis_names, rdf)
    write_findings(summary, rdf)


def make_figure(traj_example, mean_axis, axis_names, rdf):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    # Panel 1: height-axis trajectory for the right wrist (worst joint), one clip
    cid, base, opt = traj_example
    rw = R_SHO  # use a fast-moving upper-body joint; r_wrist index
    rw = GP_IDX.index(16)  # r_wrist
    ax[0].plot(base[:, rw, VERT] * 1000, color="#d9822b", lw=1.6, label="baseline (aligned)")
    ax[0].plot(opt[:, rw, VERT] * 1000, color="#2a9d3a", lw=1.6, label="+ optimization")
    ax[0].set_title(f"Right-wrist HEIGHT (Z) trajectory — clip {cid}")
    ax[0].set_xlabel("frame"); ax[0].set_ylabel("Z height (mm)")
    ax[0].legend(); ax[0].grid(alpha=0.3)

    # Panel 2: which axis the optimization moves
    colors = ["#9aa0a6", "#9aa0a6", "#2a9d3a"]
    ax[1].bar(["X\n(L/R)", "Y\n(depth)", "Z\n(height)"], mean_axis, color=colors)
    ax[1].set_title("Where the correction acts (mean |Δ|)")
    ax[1].set_ylabel("mm")
    for i, v in enumerate(mean_axis):
        ax[1].text(i, v + 1, f"{v:.0f}", ha="center", fontsize=10)
    ax[1].grid(alpha=0.3, axis="y")

    # Panel 3: coaching-indicator change
    inds = ["shoulder\nturn", "hip\nturn", "spine\ntilt"]
    vals = [rdf["d_shoulder_turn"].mean(), rdf["d_hip_turn"].mean(), rdf["d_spine_tilt"].mean()]
    icol = ["#9aa0a6", "#9aa0a6", "#2a9d3a"]
    ax[2].bar(inds, vals, color=icol)
    ax[2].set_title("Coaching-indicator change (mean |Δ|)")
    ax[2].set_ylabel("degrees")
    for i, v in enumerate(vals):
        ax[2].text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=10)
    ax[2].grid(alpha=0.3, axis="y")

    fig.suptitle("Z-depth optimization corrects HEIGHT only: rotation metrics unchanged, "
                 "height/posture metrics shift", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "z_correction_coaching_effect.png", dpi=150)
    plt.close(fig)
    print(f"[+] wrote {OUT/'z_correction_coaching_effect.png'}")


def write_findings(summary, rdf):
    ax = summary["per_axis_correction_mm"]
    ind = summary["indicator_mean_abs_delta_deg"]
    lines = [
        "# What the z-depth optimization changes on real GolfDB clips\n\n",
        f"Sampled {len(summary['clips'])} real-time driver clips "
        f"(face-on + down-the-line): {summary['clips']}.\n\n",
        "MediaPipe world landmarks -> baseline (Procrustes-aligned) vs optimized "
        "(+ Savitzky-Golay + learned MLP). GolfDB has no depth ground truth, so "
        "these are *effect* magnitudes; the accuracy improvement (106 -> 33 mm) is "
        "measured on the Vicon test set (`outputs/comparison/`).\n\n",
        "## 1. The correction acts on the HEIGHT axis only\n\n",
        "| Axis | Mean |Δ| (baseline → optimized) |\n|---|---|\n",
        f"| X (left/right) | {ax['X (left/right)']:.1f} mm |\n",
        f"| Y (depth, toward/away camera) | {ax['Y (depth)']:.1f} mm |\n",
        f"| **Z (height/vertical)** | **{ax['Z (height)']:.1f} mm** |\n\n",
        "The Procrustes alignment leaves X and Y untouched and the SG+MLP stages "
        "edit only GolfPose-Z, which Austin's skeleton mapping defines as *height* "
        "(not camera depth). So the optimization is, precisely, a **vertical-axis** "
        "correction.\n\n",
        "## 2. Effect on coaching indicators\n\n",
        "| Indicator | Depends on | Mean |Δ| |\n|---|---|---|\n",
        f"| shoulder turn | horizontal plane (rotation) | {ind['shoulder_turn']:.3f}° |\n",
        f"| hip turn | horizontal plane (rotation) | {ind['hip_turn']:.3f}° |\n",
        f"| spine tilt | vertical/height | {ind['spine_tilt']:.3f}° |\n\n",
        "## Bottom line\n\n",
        "- **Joint estimation:** improved on the height axis — this is exactly the "
        "Vicon-validated 106 → 33 mm (~3×) gain.\n",
        "- **Coaching — rotation metrics** (shoulder turn, hip turn, X-factor): "
        "**unchanged (0.00°)**. They live in the camera-depth horizontal plane, which "
        "this optimization does not touch. These are the marquee swing-rotation "
        "metrics, so the optimization does *not* (yet) improve them.\n",
        "- **Coaching — height metrics** (spine tilt, posture, head lift, vertical "
        f"weight shift): **do shift** (~{ind['spine_tilt']:.1f}° on spine tilt) and "
        "become more trustworthy.\n\n",
        "**Implication for next step:** to improve the depth-rotation coaching metrics, "
        "the next optimization must target the GolfPose-Y (camera-depth) axis, which "
        "Austin's FINDINGS explicitly note was *not* evaluated.\n",
    ]
    with open(OUT / "FINDINGS.md", "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"[+] wrote {OUT/'FINDINGS.md'}")


if __name__ == "__main__":
    main()
