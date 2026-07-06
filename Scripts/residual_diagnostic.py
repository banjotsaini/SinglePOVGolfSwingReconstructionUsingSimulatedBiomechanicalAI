"""Residual-structure diagnostic: is there anything for a learned correction to
learn ON TOP OF MixSTE?

Pairs golf-fine-tuned MixSTE (17+0) predictions with GolfPose Vicon 3D GT,
Procrustes-aligns per swing, and characterizes the residual R = GT - aligned_pred
by joint, swing phase, and axis. Produces per-joint tables, residual-vs-phase and
histogram figures, and the numbers behind a GO / NO-GO call.

NOT a training pipeline. Read-only. Does not touch the default lifter path.

Run:  python3 Scripts/residual_diagnostic.py
      python3 Scripts/camera_depth_decomp.py    # camera-frame (depth) decomposition
      python3 Scripts/transfer_check.py         # cross-subject transfer proxies
      python3 Scripts/make_final_figures.py     # figures

Local reproduction setup (NOT committed — reproduce once, then everything caches):
  1. pip install torch scikit-learn einops timm
  2. fetch the MixSTE model def to scratchpad/golfpose_repo/common/model_cross.py
     (official repo: JinluZhang1126/MixSTE), + empty stubs for its unused
     rela/routing_transformer/linearattention sibling imports.
  3. aws s3 cp s3://motioncaddie-capstone-data-lj-2026/01_inputs/Models/golfpose/
     golfpose_17plus0.bin  Models/golfpose/   (gitignored, 405 MB)
The first run lifts G1-G6 and caches predictions to
scratchpad/mixste_preds_G1-G6.pkl; later runs load the cache in seconds.
"""
from __future__ import annotations
import sys, json, pickle, warnings
from pathlib import Path
import numpy as np
from scipy import linalg
warnings.filterwarnings("ignore")

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ / "scratchpad" / "golfpose_repo"))
OUT = PROJ / "outputs" / "residual_diagnostic"
OUT.mkdir(parents=True, exist_ok=True)
CKPT = PROJ / "Models" / "golfpose" / "golfpose_17plus0.bin"
PRED_CACHE = PROJ / "scratchpad" / "mixste_preds_G1-G6.pkl"

CAM_W, CAM_H, RF = 1280, 720, 243

# H36M-17 / GolfPose body-17 ordering (verbatim from coaching_accuracy_benchmark.py)
JOINT_NAMES = ["hip", "r_hip", "r_knee", "r_ankle", "l_hip", "l_knee", "l_ankle",
               "spine", "neck", "nose", "head", "l_shoulder", "l_elbow", "l_wrist",
               "r_shoulder", "r_elbow", "r_wrist"]
L_WRI, R_WRI = 13, 16
VERT = 2  # GolfPose lab vertical/height axis; horizontal plane = (0,1)

# Split (comparable to Austin's baseline): train G1-G4, held-out test G5-G6.
# Image folders S5/S6 are scrambled vs npz (S5<->G6, S6<->G5) in golfpose_eval.py,
# but this script lifts the npz 2D GT directly, so it works entirely in npz G-space
# where 2D and 3D share a key -> the S<->G image scramble cannot mis-pair anything.
TRAIN_SUBJ = ("G1", "G2", "G3", "G4")
TEST_SUBJ = ("G5", "G6")


def load_vicon():
    d3 = np.load(PROJ / "golfswing" / "data_3d_golf_gt.npz", allow_pickle=True)["positions_3d"].item()
    d2 = np.load(PROJ / "golfswing" / "data_2d_golf_gt.npz", allow_pickle=True)["positions_2d"].item()
    data = {}
    for s in d3:
        if s not in d2:
            continue
        for sw in d3[s]:
            if sw not in d2[s]:
                continue
            gt = np.asarray(d3[s][sw], np.float32)
            xy = d2[s][sw]
            xy = np.asarray(xy[0] if isinstance(xy, list) else xy, np.float32)
            if gt.shape[1] < 17 or xy.shape[1] < 17:
                continue
            data.setdefault(s, {})[sw] = (gt[:, :17, :], xy[:, :17, :])
    return data


def build_lifter():
    import torch
    from common.model_cross import MixSTE2
    m = MixSTE2(num_frame=243, num_joints=17, in_chans=2, embed_dim_ratio=512, depth=8,
                num_heads=8, mlp_ratio=2.0, qkv_bias=True, qk_scale=None, drop_path_rate=0.0)
    st = torch.load(str(CKPT), map_location="cpu", weights_only=False)["model_pos"]
    miss, unexp = m.load_state_dict({k.replace("module.", ""): v for k, v in st.items()}, strict=False)
    assert len(miss) == 0 and len(unexp) == 0, f"checkpoint mismatch miss={miss} unexp={unexp}"
    m.eval()

    def norm(xy):
        o = xy.astype(np.float32) / CAM_W * 2.0
        o[..., 0] -= 1.0
        o[..., 1] -= CAM_H / CAM_W
        return o

    def lift(xy):
        x = norm(xy)
        T = x.shape[0]
        if T < RF:
            pl = (RF - T) // 2; pr = RF - T - pl
            xp = np.concatenate([np.repeat(x[:1], pl, 0), x, np.repeat(x[-1:], pr, 0)])
            with torch.no_grad():
                p = m(torch.from_numpy(xp[None]))[0].numpy()
            return p[pl:pl + T]
        out = np.zeros((T, 17, 3), np.float32); cnt = np.zeros(T, np.int32); step = RF // 2
        pos = list(range(0, T - RF + 1, step))
        if pos[-1] != T - RF:
            pos.append(T - RF)
        for s0 in pos:
            with torch.no_grad():
                p = m(torch.from_numpy(x[s0:s0 + RF][None]))[0].numpy()
            out[s0:s0 + RF] += p; cnt[s0:s0 + RF] += 1
        return out / np.maximum(cnt[:, None, None], 1)
    return lift


def procrustes_swing(pred, gt):
    """One rotation+scale per SWING (the coaching benchmark's alignment)."""
    pc = pred - pred[:, :1]; gc = gt - gt[:, :1]
    A = gc.reshape(-1, 3); B = pc.reshape(-1, 3)
    R, _ = linalg.orthogonal_procrustes(B, A); BR = B @ R
    s = np.sum(A * BR) / np.sum(BR ** 2)
    aligned = (s * (pc.reshape(-1, 3) @ R)).reshape(pred.shape)
    rot_deg = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    return aligned, gc, rot_deg, s


def procrustes_frame(pred, gt):
    """Rotation+scale per FRAME (the spec). Removes all global pose transform,
    leaving pure per-pose SHAPE residual — what a residual MLP could learn."""
    pc = pred - pred[:, :1]; gc = gt - gt[:, :1]
    aligned = np.empty_like(pc)
    scales = np.empty(len(pc))
    for t in range(len(pc)):
        B = pc[t]; A = gc[t]
        R, _ = linalg.orthogonal_procrustes(B, A); BR = B @ R
        s = np.sum(A * BR) / (np.sum(BR ** 2) + 1e-12)
        aligned[t] = s * (B @ R); scales[t] = s
    return aligned, gc, float(scales.mean())


# default alignment used by the diagnostic
def procrustes_align(pred, gt):
    aligned, gc, s = procrustes_frame(pred, gt)
    # report the per-swing global rotation separately (quirk #2 magnitude)
    _, _, rot_deg, _ = procrustes_swing(pred, gt)
    return aligned, gc, rot_deg, s


def detect_phases(gt_rel):
    """Kinematic phase labels per frame from Vicon GT hand motion. 6 phases.
    Robust proxies: top = max hand height; impact = max hand speed after top."""
    hands = 0.5 * (gt_rel[:, L_WRI] + gt_rel[:, R_WRI])
    T = len(hands)
    height = hands[:, VERT]
    lo, hi = int(0.10 * T), int(0.90 * T)
    top = lo + int(np.argmax(height[lo:hi])) if hi > lo else T // 2
    speed = np.linalg.norm(np.diff(hands, axis=0, prepend=hands[:1]), axis=-1)
    after = np.zeros(T, bool); after[top:] = True
    impact = top + int(np.argmax(speed[top:])) if top < T - 1 else T - 1
    labels = np.empty(T, object)
    a_end = int(0.10 * T)
    for i in range(T):
        if i < a_end:
            labels[i] = "address"
        elif i < top - max(2, int(0.03 * T)):
            labels[i] = "backswing"
        elif i <= top + max(2, int(0.03 * T)):
            labels[i] = "top"
        elif i < impact - max(2, int(0.03 * T)):
            labels[i] = "downswing"
        elif i <= impact + max(2, int(0.03 * T)):
            labels[i] = "impact"
        else:
            labels[i] = "follow_through"
    return labels, top, impact


PHASES = ["address", "backswing", "top", "downswing", "impact", "follow_through"]


def main():
    data = load_vicon()
    print("[data] subjects:", {s: len(data[s]) for s in sorted(data)})

    if PRED_CACHE.exists():
        preds = pickle.load(open(PRED_CACHE, "rb"))
        print(f"[preds] loaded cache {PRED_CACHE.name}")
    else:
        lift = build_lifter()
        preds = {}
        for s in sorted(data):
            for sw in sorted(data[s]):
                gt, xy = data[s][sw]
                preds[(s, sw)] = lift(xy)
                print(f"  lifted {s} {sw}  T={xy.shape[0]}")
        pickle.dump(preds, open(PRED_CACHE, "wb"))
        print(f"[preds] cached -> {PRED_CACHE.name}")

    # accumulate residuals: per (split, joint, axis) and per (split, phase)
    rows = []              # long-form residual records (subsampled per joint stats)
    rot_report = []
    per = {"train": [], "test": []}         # list of (T,17,3) residuals
    phase_res = {"train": {p: [] for p in PHASES}, "test": {p: [] for p in PHASES}}
    joint_axis = {"train": np.zeros((17, 3)), "test": np.zeros((17, 3))}  # placeholder
    hist_store = {"train": {}, "test": {}}

    swing_vs_frame = {"train": {"swing_mm": [], "frame_mm": []},
                      "test": {"swing_mm": [], "frame_mm": []}}
    for (s, sw), pred in preds.items():
        gt, xy = data[s][sw]
        aligned, gc, rot_deg, scale = procrustes_align(pred, gt)
        R = gc - aligned                      # residual GT - aligned_pred, (T,17,3), root-rel
        sp_aligned, _, _, _ = procrustes_swing(pred, gt)
        split_k = "train" if s in TRAIN_SUBJ else "test"
        swing_vs_frame[split_k]["swing_mm"].append(float(np.linalg.norm(gc - sp_aligned, axis=-1).mean() * 1000))
        swing_vs_frame[split_k]["frame_mm"].append(float(np.linalg.norm(R, axis=-1).mean() * 1000))
        split = "train" if s in TRAIN_SUBJ else "test"
        rot_report.append({"subj": s, "swing": sw, "procrustes_rot_deg": round(rot_deg, 1),
                           "scale": round(scale, 3), "T": int(len(R)),
                           "mpjpe_mm": round(float(np.linalg.norm(R, axis=-1).mean() * 1000), 1)})
        per[split].append(R)
        labels, top, impact = detect_phases(gc)
        for p in PHASES:
            sel = labels == p
            if sel.any():
                phase_res[split][p].append(np.linalg.norm(R[sel], axis=-1).reshape(-1))

    diag = {"split_def": {"train": TRAIN_SUBJ, "test": TEST_SUBJ},
            "model": "golfpose MixSTE 17+0", "note_club": "17+0 checkpoint has NO club joints; wrists are the closest club proxy.",
            "per_swing_alignment": rot_report}

    # ---- per-joint residual magnitude (mm), mean & std, per split ----
    out_tables = {}
    for split in ("train", "test"):
        Rall = np.concatenate(per[split], axis=0)         # (Ntot,17,3), meters
        mag = np.linalg.norm(Rall, axis=-1) * 1000.0       # (Ntot,17) mm
        signed = Rall * 1000.0                              # mm, per axis
        tab = []
        for j in range(17):
            tab.append({
                "joint": JOINT_NAMES[j],
                "mean_mm": round(float(mag[:, j].mean()), 1),
                "std_mm": round(float(mag[:, j].std()), 1),
                "bias_x_mm": round(float(signed[:, j, 0].mean()), 1),
                "bias_y_mm": round(float(signed[:, j, 1].mean()), 1),
                "bias_z_mm": round(float(signed[:, j, 2].mean()), 1),
                "std_x_mm": round(float(signed[:, j, 0].std()), 1),
                "std_y_mm": round(float(signed[:, j, 1].std()), 1),
                "std_z_mm": round(float(signed[:, j, 2].std()), 1),
            })
        out_tables[split] = tab
        # axis energy share across all joints
        var_axis = signed.reshape(-1, 3).var(axis=0)
        diag[f"axis_energy_share_{split}"] = {ax: round(float(v / var_axis.sum()), 3)
                                              for ax, v in zip(["x", "y_lab", "z_vert"], var_axis)}
    diag["per_joint"] = out_tables

    # ---- structure metric: |bias| / std  per joint (how much residual is systematic) ----
    struct = {}
    for split in ("train", "test"):
        s_rows = []
        for r in out_tables[split]:
            bias_mag = (r["bias_x_mm"]**2 + r["bias_y_mm"]**2 + r["bias_z_mm"]**2) ** 0.5
            s_rows.append({"joint": r["joint"], "bias_mag_mm": round(bias_mag, 1),
                           "rand_mm": r["std_mm"],
                           "systematic_frac": round(bias_mag / (r["mean_mm"] + 1e-9), 2)})
        struct[split] = s_rows
    diag["structure"] = struct

    # ---- residual by phase (mean mm) per split ----
    phase_tab = {}
    for split in ("train", "test"):
        phase_tab[split] = {p: (round(float(np.concatenate(phase_res[split][p]).mean() * 1000), 1)
                                 if phase_res[split][p] else None) for p in PHASES}
    diag["residual_by_phase_mm"] = phase_tab

    diag["align_compare_mm"] = {
        split: {"per_swing_procrustes": round(float(np.mean(swing_vs_frame[split]["swing_mm"])), 1),
                "per_frame_procrustes": round(float(np.mean(swing_vs_frame[split]["frame_mm"])), 1)}
        for split in ("train", "test")}

    json.dump(diag, open(OUT / "diagnostic.json", "w"), indent=2)
    print("\n[+] wrote", OUT / "diagnostic.json")

    # ---- figures ----
    make_figures(per, phase_res, out_tables)
    print_summary(diag)
    return diag


def make_figures(per, phase_res, out_tables):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Fig 1: per-joint residual mean+std, test split (held-out)
    tab = out_tables["test"]
    order = np.argsort([r["mean_mm"] for r in tab])[::-1]
    names = [tab[i]["joint"] for i in order]
    means = [tab[i]["mean_mm"] for i in order]
    stds = [tab[i]["std_mm"] for i in order]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(names, means, yerr=stds, color="#0C5C3A", ecolor="#E8B23E", capsize=3)
    ax.set_ylabel("residual |GT - aligned MixSTE| (mm)")
    ax.set_title("Per-joint residual on held-out G5/G6 (mean ± std) — after per-swing Procrustes")
    ax.tick_params(axis="x", rotation=45)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    fig.tight_layout(); fig.savefig(OUT / "per_joint_residual.png", dpi=140); plt.close(fig)

    # Fig 2: residual vs phase, worst joints, test split
    fig, ax = plt.subplots(figsize=(10, 4.5))
    Rall = np.concatenate(per["test"], axis=0)
    # rebuild phase means per joint using stored per-frame? we only stored magnitude across joints.
    # Recompute phase means per joint from per['test'] requires labels; instead show overall phase curve.
    xs = list(range(len(PHASES)))
    means = [np.concatenate(phase_res["test"][p]).mean() * 1000 if phase_res["test"][p] else np.nan
             for p in PHASES]
    ax.plot(xs, means, "-o", color="#0C5C3A", lw=2)
    ax.set_xticks(xs); ax.set_xticklabels(PHASES, rotation=20)
    ax.set_ylabel("mean residual (mm)")
    ax.set_title("Residual by swing phase — held-out G5/G6")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "residual_by_phase.png", dpi=140); plt.close(fig)

    # Fig 3: histograms of per-axis residual for the 3 worst joints (test)
    worst = [np.argsort([r["mean_mm"] for r in out_tables["test"]])[::-1][:3]][0]
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    axn = ["X (lab)", "Y (lab)", "Z (vertical)"]
    for col, j in enumerate(worst):
        for a in range(3):
            axs[col].hist(Rall[:, j, a] * 1000, bins=60, alpha=0.5, label=axn[a])
        axs[col].set_title(f"{JOINT_NAMES[j]} residual (mm)")
        axs[col].axvline(0, color="k", lw=0.8)
        axs[col].legend(fontsize=8); axs[col].grid(alpha=0.3)
    fig.suptitle("Per-axis residual distribution, 3 highest-error joints (held-out G5/G6)")
    fig.tight_layout(); fig.savefig(OUT / "residual_histograms.png", dpi=140); plt.close(fig)
    print("[+] wrote 3 figures to", OUT)


def print_summary(diag):
    print("\n" + "=" * 68)
    print("RESIDUAL DIAGNOSTIC SUMMARY  (held-out G5/G6 unless noted)")
    print("=" * 68)
    rot = np.mean([r["procrustes_rot_deg"] for r in diag["per_swing_alignment"]])
    print(f"mean per-swing Procrustes rotation to lab frame : {rot:.1f} deg  (quirk #2: global axis offset)")
    print(f"align compare (mean residual mm): {diag['align_compare_mm']}")
    print(f"axis energy share (test, per-frame residual) : {diag['axis_energy_share_test']}")
    print("\nper-joint residual (test), sorted worst-first:")
    for r in sorted(diag["per_joint"]["test"], key=lambda x: -x["mean_mm"]):
        print(f"  {r['joint']:<11} mean={r['mean_mm']:5.1f}  std={r['std_mm']:5.1f}  "
              f"bias(x,y,z)=({r['bias_x_mm']:5.1f},{r['bias_y_mm']:5.1f},{r['bias_z_mm']:5.1f})")
    print("\nresidual by phase (mm):")
    for split in ("train", "test"):
        print(f"  {split}: " + "  ".join(f"{p}={diag['residual_by_phase_mm'][split][p]}" for p in PHASES))
    print("\nstructure (systematic bias fraction, test):")
    for r in sorted(diag["structure"]["test"], key=lambda x: -x["bias_mag_mm"]):
        print(f"  {r['joint']:<11} bias={r['bias_mag_mm']:5.1f}mm  frac_systematic={r['systematic_frac']}")


if __name__ == "__main__":
    main()
