"""Resolve the physical meaning of the dominant residual axis: is MixSTE's
leftover error along CAMERA DEPTH (the ill-posed monocular direction) or in-plane?

Recovers a weak-perspective camera per swing from the Vicon 2D<->3D GT
(2D_px ~ scale * (Rcam @ X_lab)[:2] + t), so Rcam[2] is the lab-space viewing
(depth) axis. Then projects the per-frame Procrustes residual onto image-plane
vs depth. Loads cached MixSTE preds; read-only.
"""
from __future__ import annotations
import sys, pickle, json, warnings
from pathlib import Path
import numpy as np
from scipy import linalg
warnings.filterwarnings("ignore")

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ / "Scripts"))
from residual_diagnostic import (load_vicon, procrustes_frame, JOINT_NAMES,
                                 TRAIN_SUBJ, TEST_SUBJ, detect_phases, PHASES)

preds = pickle.load(open(PROJ / "scratchpad" / "mixste_preds_G1-G6.pkl", "rb"))
data = load_vicon()


def recover_camera(xy_px, X_lab):
    """Weak-perspective fit. xy_px:(T,17,2) pixels, X_lab:(T,17,3) meters.
    Returns unit lab-space depth axis (viewing direction)."""
    # center both
    xy = xy_px.reshape(-1, 2); X = X_lab.reshape(-1, 3)
    xy = xy - xy.mean(0); X = X - X.mean(0)
    # solve xy ~ X @ M   (M is 3x2), columns are image x,y directions in lab space
    M, *_ = np.linalg.lstsq(X, xy, rcond=None)          # (3,2)
    ex = M[:, 0] / (np.linalg.norm(M[:, 0]) + 1e-9)
    ey = M[:, 1] / (np.linalg.norm(M[:, 1]) + 1e-9)
    depth = np.cross(ex, ey)
    depth /= (np.linalg.norm(depth) + 1e-9)
    # re-orthogonalize image plane basis
    ey2 = np.cross(depth, ex); ey2 /= np.linalg.norm(ey2) + 1e-9
    return np.stack([ex, ey2, depth])                    # rows: img-x, img-y, depth (lab coords)


out = {"train": {"inplane": [], "depth": []}, "test": {"inplane": [], "depth": []}}
depth_by_phase = {"train": {p: [] for p in PHASES}, "test": {p: [] for p in PHASES}}
frac_depth_joint = {"train": {j: [] for j in range(17)}, "test": {j: [] for j in range(17)}}

for (s, sw), pred in preds.items():
    gt, xy = data[s][sw]
    aligned, gc, _ = procrustes_frame(pred, gt)
    R = gc - aligned                                     # (T,17,3) lab frame, meters
    B = recover_camera(xy, gt[:, :17, :])                # (3,3) rows img-x,img-y,depth
    Rcam = R @ B.T                                       # residual in camera frame: [...,2]=depth
    split = "train" if s in TRAIN_SUBJ else "test"
    inplane = np.linalg.norm(Rcam[..., :2], axis=-1)     # (T,17)
    depth = np.abs(Rcam[..., 2])
    out[split]["inplane"].append(inplane.reshape(-1))
    out[split]["depth"].append(depth.reshape(-1))
    labels, _, _ = detect_phases(gc)
    for p in PHASES:
        sel = labels == p
        if sel.any():
            depth_by_phase[split][p].append(depth[sel].reshape(-1))
    for j in range(17):
        e2 = Rcam[:, j] ** 2
        frac_depth_joint[split][j].append((e2[:, 2].sum(), e2.sum()))

res = {}
for split in ("train", "test"):
    ip = np.concatenate(out[split]["inplane"]) * 1000
    dp = np.concatenate(out[split]["depth"]) * 1000
    # energy share
    ip2 = (ip ** 2).sum(); dp2 = (dp ** 2).sum()
    res[split] = {
        "mean_inplane_mm": round(float(ip.mean()), 1),
        "mean_depth_mm": round(float(dp.mean()), 1),
        "depth_energy_share": round(float(dp2 / (ip2 + dp2)), 3),
    }
    res[split]["depth_by_phase_mm"] = {
        p: round(float(np.concatenate(depth_by_phase[split][p]).mean() * 1000), 1)
        if depth_by_phase[split][p] else None for p in PHASES}
    res[split]["depth_frac_by_joint"] = {}
    for j in range(17):
        num = sum(a for a, _ in frac_depth_joint[split][j])
        den = sum(b for _, b in frac_depth_joint[split][j]) + 1e-12
        res[split]["depth_frac_by_joint"][JOINT_NAMES[j]] = round(float(num / den), 2)

json.dump(res, open(PROJ / "outputs" / "residual_diagnostic" / "camera_depth.json", "w"), indent=2)

print("=" * 62)
print("CAMERA-FRAME residual decomposition (per-frame Procrustes residual)")
print("=" * 62)
for split in ("train", "test"):
    r = res[split]
    print(f"\n[{split}] in-plane={r['mean_inplane_mm']}mm  depth={r['mean_depth_mm']}mm  "
          f"depth_energy_share={r['depth_energy_share']}")
    print("  depth residual by phase (mm): " +
          "  ".join(f"{p}={r['depth_by_phase_mm'][p]}" for p in PHASES))
print("\ndepth fraction of residual energy, per joint (test), worst-first:")
for j, f in sorted(res["test"]["depth_frac_by_joint"].items(), key=lambda x: -x[1]):
    print(f"  {j:<11} {f}")
