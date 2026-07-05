"""Is the MixSTE depth residual TRANSFERABLE across subjects? (the GO/NO-GO crux)

The only Vicon-labeled subjects are G1-G6, and MixSTE was trained on G1-G4 (so its
in-sample residual there is tiny). A learned corrector is only worth building if the
depth-error STRUCTURE learned on some subjects transfers to unseen ones.

Two cheap, no-training proxies (leave-subject-out on the held-out pair, plus a
pose-conditioned bound):
  1. per-joint constant depth debias learned on train subjects, applied to test.
  2. per-joint x PHASE depth debias (pose-conditioned) learned on 1 test subject,
     applied to the other test subject (true leave-one-subject-out).
If a fixed per-joint (and phase) depth shift measurably cuts held-out depth error,
a richer pose-conditioned MLP has real signal to learn. If not, the residual is
subject-specific noise -> NO-GO.
"""
from __future__ import annotations
import sys, pickle, json, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ / "Scripts"))
from residual_diagnostic import load_vicon, procrustes_frame, JOINT_NAMES, PHASES, detect_phases
from camera_depth_decomp import recover_camera

preds = pickle.load(open(PROJ / "scratchpad" / "mixste_preds_G1-G6.pkl", "rb"))
data = load_vicon()

# collect per-frame camera-depth residual + joint + phase, tagged by subject
recs = {}   # subj -> list of (depth_resid (17,), phase-label array per frame)
for (s, sw), pred in preds.items():
    gt, xy = data[s][sw]
    aligned, gc, _ = procrustes_frame(pred, gt)
    R = gc - aligned
    B = recover_camera(xy, gt[:, :17, :])
    depth = (R @ B.T)[..., 2]             # (T,17) signed camera-depth residual (m)
    labels, _, _ = detect_phases(gc)
    recs.setdefault(s, []).append((depth, labels))


def stack(subjects):
    d = np.concatenate([r[0] for s in subjects for r in recs[s]], axis=0)  # (N,17)
    lab = np.concatenate([r[1] for s in subjects for r in recs[s]], axis=0)  # (N,)
    return d, lab


def rmse(x):
    return float(np.sqrt(np.mean(x ** 2)) * 1000)


# ---------- Proxy 1: per-joint constant depth debias, train G1-G4 -> test G5/G6 ----------
train_d, _ = stack(["G1", "G2", "G3", "G4"])
test_d, test_lab = stack(["G5", "G6"])
bias_joint = train_d.mean(axis=0)                       # (17,) learned on train
before = rmse(test_d)
after_const = rmse(test_d - bias_joint[None, :])

# ---------- Proxy 2: pose-conditioned (joint x phase) debias, leave-one-subject-out ----------
# learn on G5, apply to G6, and vice versa (true unseen-subject transfer)
def phase_joint_bias(subj):
    d, lab = stack([subj])
    tab = np.zeros((len(PHASES), 17))
    for pi, p in enumerate(PHASES):
        sel = lab == p
        tab[pi] = d[sel].mean(0) if sel.any() else 0.0
    return tab

def apply_phase_bias(subj, tab):
    d, lab = stack([subj])
    corr = d.copy()
    for pi, p in enumerate(PHASES):
        sel = lab == p
        corr[sel] -= tab[pi][None, :]
    return rmse(d), rmse(corr)

b5 = phase_joint_bias("G5"); b6 = phase_joint_bias("G6")
# apply the OTHER subject's learned bias (unseen transfer)
g6_before, g6_after = apply_phase_bias("G6", b5)
g5_before, g5_after = apply_phase_bias("G5", b6)

# consistency: correlation of per-joint constant depth bias, train vs test
test_bias_joint = test_d.mean(0)
corr = float(np.corrcoef(bias_joint, test_bias_joint)[0, 1])

res = {
    "proxy1_perjoint_const_debias_train->test": {
        "test_depth_rmse_before_mm": round(before, 1),
        "test_depth_rmse_after_mm": round(after_const, 1),
        "reduction_pct": round(100 * (before - after_const) / before, 1),
    },
    "proxy2_pose_conditioned_leave_one_subject_out": {
        "learn_G5_apply_G6": {"before_mm": round(g6_before, 1), "after_mm": round(g6_after, 1),
                              "reduction_pct": round(100 * (g6_before - g6_after) / g6_before, 1)},
        "learn_G6_apply_G5": {"before_mm": round(g5_before, 1), "after_mm": round(g5_after, 1),
                              "reduction_pct": round(100 * (g5_before - g5_after) / g5_before, 1)},
    },
    "per_joint_depth_bias_corr_train_vs_test": round(corr, 2),
    "per_joint_depth_bias_train_mm": {JOINT_NAMES[j]: round(float(bias_joint[j] * 1000), 1) for j in range(17)},
    "per_joint_depth_bias_test_mm": {JOINT_NAMES[j]: round(float(test_bias_joint[j] * 1000), 1) for j in range(17)},
}
json.dump(res, open(PROJ / "outputs" / "residual_diagnostic" / "transfer_check.json", "w"), indent=2)

print("=" * 64)
print("TRANSFERABILITY OF THE DEPTH RESIDUAL  (the GO/NO-GO crux)")
print("=" * 64)
p1 = res["proxy1_perjoint_const_debias_train->test"]
print(f"\nProxy 1 - per-joint const depth debias, train G1-4 -> test G5/6:")
print(f"  depth RMSE {p1['test_depth_rmse_before_mm']}mm -> {p1['test_depth_rmse_after_mm']}mm "
      f"({p1['reduction_pct']:+.1f}%)")
p2 = res["proxy2_pose_conditioned_leave_one_subject_out"]
print(f"\nProxy 2 - pose-conditioned (joint x phase) debias, leave-one-subject-out:")
for k, v in p2.items():
    print(f"  {k}: {v['before_mm']}mm -> {v['after_mm']}mm ({v['reduction_pct']:+.1f}%)")
print(f"\nper-joint depth-bias correlation train vs test: r = {res['per_joint_depth_bias_corr_train_vs_test']}")
print("\nper-joint signed depth bias (mm)   train  |  test")
for j in range(17):
    n = JOINT_NAMES[j]
    print(f"  {n:<11} {bias_joint[j]*1000:7.1f}  | {test_bias_joint[j]*1000:7.1f}")
