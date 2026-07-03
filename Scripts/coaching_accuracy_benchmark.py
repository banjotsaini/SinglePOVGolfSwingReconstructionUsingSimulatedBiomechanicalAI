"""Ground-truth benchmark: how accurate are the coaching MEASUREMENTS?

The coaching app derives biomechanical indicators (shoulder/hip turn, spine tilt,
arm bend, knee flex) from estimated 3D pose. Until now we had no way to ask
"are those numbers right?" because GolfDB has no 3D ground truth.

GolfPose Vicon (golfpose_data/) does. This harness:
  1. lifts the Vicon 2D ground-truth keypoints to 3D with the real lifter
     (GolfPose MixSTE, golf-fine-tuned; the coaching pipeline's MotionBERT is a
     swappable backend),
  2. best-fit aligns the prediction to the Vicon 3D (one rotation+scale per swing,
     Procrustes — so we measure residual *shape* error, not a global frame offset),
  3. computes the same geometric quantities the coaching indicators use, from BOTH
     the prediction and the Vicon truth, every frame,
  4. reports mean-abs measurement error per quantity (degrees).

That error is the thing every optimization must reduce. Interventions (jitter
smoothing, z-depth correction, ...) are applied to the predicted 3D via --intervention
and benchmarked against the untouched baseline.

Split follows Austin's eval: train subjects G1-G4 (available for learned fixes),
report on test subjects G5/G6.

Run:  python Scripts/coaching_accuracy_benchmark.py --backend mixste
"""
from __future__ import annotations

import sys
import json
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
from scipy import linalg

PROJ = Path(__file__).resolve().parent.parent
GP_DATA = PROJ / "golfpose_data"
OUT = PROJ / "outputs" / "coaching_accuracy"
OUT.mkdir(parents=True, exist_ok=True)

TRAIN_SUBJ = ("G1", "G2", "G3", "G4")
TEST_SUBJ = ("G5", "G6")

# H36M-17 indices (GolfPose body-17 share this exact ordering)
HIP_C = 0
R_HIP, R_KNE, R_ANK = 1, 2, 3
L_HIP, L_KNE, L_ANK = 4, 5, 6
SPINE, NECK, HEAD = 7, 8, 10
L_SHO, L_ELB, L_WRI = 11, 12, 13
R_SHO, R_ELB, R_WRI = 14, 15, 16

CAM_W, CAM_H = 1280, 720  # GolfPose camera-0 intrinsics (from sanity check)
VERT = 2  # GolfPose lab vertical/height axis; horizontal plane = axes (0,1)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_vicon():
    d3 = np.load(GP_DATA / "data_3d_golf_gt.npz", allow_pickle=True)["positions_3d"].item()
    d2 = np.load(GP_DATA / "data_2d_golf_gt.npz", allow_pickle=True)["positions_2d"].item()
    data = {}
    for subj in d3:
        if subj not in d2:
            continue
        for sw in d3[subj]:
            if sw not in d2[subj]:
                continue
            gt = np.asarray(d3[subj][sw], dtype=np.float32)
            xy = d2[subj][sw]
            xy = np.asarray(xy[0] if isinstance(xy, list) else xy, dtype=np.float32)
            if gt.shape[1] < 17 or xy.shape[1] < 17:
                continue
            data.setdefault(subj, {})[sw] = (gt[:, :17, :], xy[:, :17, :])
    return data


# ---------------------------------------------------------------------------
# lifter backends
# ---------------------------------------------------------------------------
class MixSTELifter:
    name = "golfpose_mixste"

    def __init__(self, device="cuda"):
        sys.path.insert(0, r"C:\dev\golfpose-repo")
        from common.model_cross import MixSTE2
        import torch
        self.torch = torch
        self.device = device
        self.rf = 243
        self.model = MixSTE2(num_frame=243, num_joints=17, in_chans=2,
                             embed_dim_ratio=512, depth=8, num_heads=8,
                             mlp_ratio=2.0, qkv_bias=True, qk_scale=None, drop_path_rate=0.0)
        ckpt = torch.load(str(PROJ / "Models" / "golfpose" / "golfpose_17plus0.bin"),
                          map_location="cpu", weights_only=False)
        state = ckpt.get("model_pos", ckpt)
        self.model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()}, strict=False)
        self.model = self.model.to(device).eval()

    def _norm(self, xy):
        out = xy.astype(np.float32) / CAM_W * 2.0
        out[..., 0] -= 1.0
        out[..., 1] -= CAM_H / CAM_W
        return out

    def lift(self, xy2d):
        torch = self.torch
        x = self._norm(xy2d)
        T, rf = x.shape[0], self.rf
        if T < rf:
            pl = (rf - T) // 2; pr = rf - T - pl
            xp = np.concatenate([np.repeat(x[:1], pl, 0), x, np.repeat(x[-1:], pr, 0)])
            with torch.no_grad():
                p = self.model(torch.from_numpy(xp[None]).to(self.device))[0].cpu().numpy()
            return p[pl:pl + T]
        out = np.zeros((T, 17, 3), np.float32); cnt = np.zeros(T, np.int32)
        step = rf // 2
        pos = list(range(0, T - rf + 1, step))
        if pos[-1] != T - rf: pos.append(T - rf)
        for s0 in pos:
            with torch.no_grad():
                p = self.model(torch.from_numpy(x[s0:s0 + rf][None]).to(self.device))[0].cpu().numpy()
            out[s0:s0 + rf] += p; cnt[s0:s0 + rf] += 1
        return out / np.maximum(cnt[:, None, None], 1)


class MotionBERTLifter:
    name = "motionbert_full"

    def __init__(self, device="cuda"):
        sys.path.insert(0, str(PROJ / "Scripts"))
        from adapters.motionbert_adapter import MotionBERTAdapter
        self.ad = MotionBERTAdapter(upstream_2d_cache_dir=PROJ, variant="full", device=device)

    def lift(self, xy2d):
        x = self.ad._prepare_input(xy2d, CAM_W, CAM_H)  # 2D already H36M-17
        return self.ad._infer(x)


# ---------------------------------------------------------------------------
# alignment + geometry
# ---------------------------------------------------------------------------
def procrustes_align(pred, gt):
    """One rotation+scale per swing aligning pred -> gt (root-centered)."""
    pc = pred - pred[:, HIP_C:HIP_C + 1, :]
    gc = gt - gt[:, HIP_C:HIP_C + 1, :]
    A = gc.reshape(-1, 3); B = pc.reshape(-1, 3)
    R, _ = linalg.orthogonal_procrustes(B, A)
    BR = B @ R
    s = np.sum(A * BR) / np.sum(BR ** 2)
    return (s * (pc.reshape(-1, 3) @ R)).reshape(pred.shape)


def _horiz_angle(v):
    """Orientation of a vector in the horizontal plane (axes 0,1), degrees."""
    return np.degrees(np.arctan2(v[..., 1], v[..., 0]))


def _interior_angle(a, b, c):
    ba, bc = a - b, c - b
    cos = np.sum(ba * bc, -1) / (np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def _tilt_from_vertical(vec):
    horiz = np.hypot(vec[..., 0], vec[..., 1])
    return np.degrees(np.arctan2(horiz, np.abs(vec[..., VERT])))


def quantities(xyz):
    """Per-frame coaching geometric quantities. xyz: (T,17,3)."""
    sho = xyz[:, R_SHO] - xyz[:, L_SHO]
    hip = xyz[:, R_HIP] - xyz[:, L_HIP]
    hip_c = 0.5 * (xyz[:, L_HIP] + xyz[:, R_HIP])
    sho_c = 0.5 * (xyz[:, L_SHO] + xyz[:, R_SHO])
    return {
        "shoulder_line_deg": _horiz_angle(sho),
        "hip_line_deg": _horiz_angle(hip),
        "spine_tilt_deg": _tilt_from_vertical(sho_c - hip_c),
        "left_arm_bend_deg": _interior_angle(xyz[:, L_SHO], xyz[:, L_ELB], xyz[:, L_WRI]),
        "right_arm_bend_deg": _interior_angle(xyz[:, R_SHO], xyz[:, R_ELB], xyz[:, R_WRI]),
        "left_knee_flex_deg": _interior_angle(xyz[:, L_HIP], xyz[:, L_KNE], xyz[:, L_ANK]),
        "right_knee_flex_deg": _interior_angle(xyz[:, R_HIP], xyz[:, R_KNE], xyz[:, R_ANK]),
    }


def ang_err(pred_deg, gt_deg, wrap=False):
    d = pred_deg - gt_deg
    if wrap:
        d = (d + 180) % 360 - 180
    return np.abs(d)


WRAP = {"shoulder_line_deg", "hip_line_deg"}  # orientation angles wrap at 360


# ---------------------------------------------------------------------------
# intervention hooks (applied to predicted 3D before alignment)
# ---------------------------------------------------------------------------
def intervention_none(pred, ctx):
    return pred


def get_intervention(name):
    from benchmark_interventions import INTERVENTIONS  # iteration-specific, lazy
    return INTERVENTIONS[name]


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
def run(backend="mixste", intervention="none", tag=None, noise_px=0.0):
    data = load_vicon()
    lifter = {"mixste": MixSTELifter, "motionbert": MotionBERTLifter}[backend]()
    print(f"[lifter] {lifter.name}   [intervention] {intervention}   [noise] {noise_px}px")

    if intervention == "none":
        fn = intervention_none
    else:
        fn = get_intervention(intervention)

    quants = list(quantities(np.zeros((1, 17, 3))).keys())
    err_accum = {q: [] for q in quants}
    n_swings = 0

    for subj in TEST_SUBJ:
        for sw, (gt, xy) in sorted(data.get(subj, {}).items()):
            if noise_px > 0:
                # reproducible per-swing jitter mimicking 2D detection noise
                rng = np.random.default_rng(abs(hash((subj, sw))) % (2**32))
                xy = xy + rng.normal(0.0, noise_px, size=xy.shape).astype(np.float32)
            pred = lifter.lift(xy)
            pred = fn(pred, {"subj": subj, "swing": sw})
            pred_a = procrustes_align(pred, gt)
            qp, qg = quantities(pred_a), quantities(gt)
            for q in quants:
                err_accum[q].append(ang_err(qp[q], qg[q], wrap=q in WRAP))
            n_swings += 1
    print(f"[benchmark] {n_swings} test swings (G5/G6)")

    summary = {}
    for q in quants:
        e = np.concatenate(err_accum[q])
        summary[q] = round(float(np.mean(e)), 2)
    summary["_overall_mean_deg"] = round(float(np.mean(list(summary.values()))), 2)
    summary["_meta"] = {"backend": lifter.name, "intervention": intervention, "n_swings": n_swings}

    summary["_meta"]["noise_px"] = noise_px
    tag = tag or f"{backend}_{intervention}" + (f"_n{noise_px:g}" if noise_px else "")
    with open(OUT / f"{tag}.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nMean-abs measurement error vs Vicon truth (degrees):")
    for q in quants:
        print(f"  {q:<22}: {summary[q]:6.2f}")
    print(f"  {'OVERALL':<22}: {summary['_overall_mean_deg']:6.2f}")
    print(f"\n[+] wrote {OUT / f'{tag}.json'}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="mixste", choices=["mixste", "motionbert"])
    ap.add_argument("--intervention", default="none")
    ap.add_argument("--noise", type=float, default=0.0, help="2D jitter std in px")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    run(a.backend, a.intervention, a.tag, noise_px=a.noise)
