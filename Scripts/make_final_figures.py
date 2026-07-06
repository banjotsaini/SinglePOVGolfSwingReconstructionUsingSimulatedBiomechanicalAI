"""Consolidated diagnostic figures (camera-depth framing)."""
import sys, pickle, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore")
PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ / "Scripts"))
from residual_diagnostic import load_vicon, procrustes_frame, JOINT_NAMES, PHASES, detect_phases
from camera_depth_decomp import recover_camera
OUT = PROJ / "outputs" / "residual_diagnostic"
preds = pickle.load(open(PROJ / "scratchpad" / "mixste_preds_G1-G6.pkl", "rb")); data = load_vicon()
GREEN, AMBER, GRAY = "#0C5C3A", "#E8B23E", "#9aa0a6"

def collect(subjs):
    dep, inp, labs = [], [], []
    for (s, sw), pred in preds.items():
        if s not in subjs: continue
        gt, xy = data[s][sw]; aligned, gc, _ = procrustes_frame(pred, gt); R = gc - aligned
        B = recover_camera(xy, gt[:, :17, :]); Rc = R @ B.T
        dep.append(Rc[..., 2]); inp.append(np.linalg.norm(Rc[..., :2], axis=-1))
        lab, _, _ = detect_phases(gc); labs.append(lab)
    return np.concatenate(dep, 0), np.concatenate(inp, 0), np.concatenate(labs, 0)

dep_te, inp_te, lab_te = collect({"G5", "G6"})
dep_tr, inp_tr, lab_tr = collect({"G1", "G2", "G3", "G4"})

# Fig 1: per-joint depth vs in-plane RMSE (test)
depR = np.sqrt((dep_te**2).mean(0))*1000; inpR = np.sqrt((inp_te**2).mean(0))*1000
order = np.argsort(depR)[::-1]; names = [JOINT_NAMES[i] for i in order]
x = np.arange(17); w = 0.4
fig, ax = plt.subplots(figsize=(12, 4.6))
ax.bar(x-w/2, depR[order], w, label="camera-DEPTH residual", color=GREEN)
ax.bar(x+w/2, inpR[order], w, label="in-plane residual", color=GRAY)
ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right")
ax.set_ylabel("residual RMSE (mm)")
ax.set_title("MixSTE residual is camera-depth, concentrated at extremities — held-out G5/G6\n(after per-frame Procrustes; 92% of residual energy is along camera depth)")
ax.legend(); ax.grid(alpha=0.3, axis="y")
fig.tight_layout(); fig.savefig(OUT / "fig1_depth_vs_inplane_perjoint.png", dpi=140); plt.close(fig)

# Fig 2: depth residual by phase, train vs test
def phase_depth(dep, lab):
    return [np.sqrt((dep[lab == p]**2).mean())*1000 if (lab == p).any() else np.nan for p in PHASES]
fig, ax = plt.subplots(figsize=(9, 4.6))
ax.plot(PHASES, phase_depth(dep_te, lab_te), "-o", color=GREEN, lw=2, label="test G5/G6 (held out)")
ax.plot(PHASES, phase_depth(dep_tr, lab_tr), "-o", color=AMBER, lw=2, label="train G1-G4 (in-sample)")
ax.set_ylabel("camera-depth residual RMSE (mm)")
ax.set_title("Depth error peaks at top-of-backswing; 4x train/test gap = generalization, not structure")
ax.legend(); ax.grid(alpha=0.3); ax.tick_params(axis="x", rotation=20)
fig.tight_layout(); fig.savefig(OUT / "fig2_depth_by_phase.png", dpi=140); plt.close(fig)

# Fig 3: transfer crux — signed bias (no transfer) vs unsigned profile (transfers)
bias_tr = dep_tr.mean(0)*1000; bias_te = dep_te.mean(0)*1000
rmse_tr = np.sqrt((dep_tr**2).mean(0))*1000; rmse_te = np.sqrt((dep_te**2).mean(0))*1000
m = [j for j in range(17) if j != 0]
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].scatter(bias_tr[m], bias_te[m], color=GREEN)
for j in m: axs[0].annotate(JOINT_NAMES[j], (bias_tr[j], bias_te[j]), fontsize=7)
axs[0].axhline(0, color="k", lw=.6); axs[0].axvline(0, color="k", lw=.6)
axs[0].set_xlabel("train signed depth bias (mm)"); axs[0].set_ylabel("test signed depth bias (mm)")
axs[0].set_title(f"SIGNED correction does NOT transfer  (r = {np.corrcoef(bias_tr[m], bias_te[m])[0,1]:.2f})")
axs[0].grid(alpha=0.3)
axs[1].scatter(rmse_tr[m], rmse_te[m], color=AMBER)
for j in m: axs[1].annotate(JOINT_NAMES[j], (rmse_tr[j], rmse_te[j]), fontsize=7)
axs[1].set_xlabel("train depth RMSE (mm)"); axs[1].set_ylabel("test depth RMSE (mm)")
axs[1].set_title(f"WHERE error lives DOES transfer  (r = {np.corrcoef(rmse_tr[m], rmse_te[m])[0,1]:.2f})")
axs[1].grid(alpha=0.3)
fig.suptitle("Transfer crux: a corrector can learn WHICH joints are bad, but not HOW to fix them across subjects")
fig.tight_layout(); fig.savefig(OUT / "fig3_transfer.png", dpi=140); plt.close(fig)
print("wrote fig1_depth_vs_inplane_perjoint.png, fig2_depth_by_phase.png, fig3_transfer.png")
# clean up the earlier lab-axis figures that are now superseded
for old in ["per_joint_residual.png", "residual_by_phase.png", "residual_histograms.png"]:
    p = OUT / old
    if p.exists(): p.unlink()
