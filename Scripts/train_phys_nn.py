"""Phys-NN training experiment — McNally et al. (CVPR 2023W) reproduced on
public data (tim-blackmore/launch-monitor-regression, ~10k TrackMan shots).

Implements the paper's final model (Phys-NN-pM-cQ + Lh loss): a differentiable
batch Euler flight sim whose per-step aerodynamic coefficients come from an MLP
    CL', CD', CQ' = N(v/89.4, w/2094);  CL=sig/2, CD=sig/2, CQ=sig,  CM=pM*S
trained on (launch conditions -> flat-ground landing position + max height).

Ship gate (BALL_FLIGHT_PLAN.md phase 3): the NN must beat the SHIPPED Phys-Q
polynomial baseline on held-out mean landing error, else we keep Phys-Q.
On success, weights are exported to Data/coaching/ball_flight_nn.json for a
pure-Python forward pass (no torch in the Lambda).

    .venv/Scripts/python.exe Scripts/train_phys_nn.py --csv <data.csv> [--epochs 50]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
import ball_flight as BF

YD, FT = 0.9144, 0.3048
MPH = 0.44704
RPM = 2 * math.pi / 60.0
DT, TSIM = 0.1, 10.0          # paper §2.3.2
N_STEPS = int(TSIM / DT)
RHO, G = 1.225, 9.81
M, R = BF.BALL_MASS_KG, BF.BALL_RADIUS_M
A, I, D = BF.BALL_AREA_M2, BF.BALL_INERTIA, 2 * BF.BALL_RADIUS_M


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #

REQ = ["Ball Speed", "Launch Angle", "Launch Direction", "Spin Rate", "Spin Axis",
       "Carry Flat - Length", "Carry Flat - Side", "Max Height - Height"]


def load(csv: Path) -> pd.DataFrame:
    df = pd.read_csv(csv, skiprows=[1])          # row 1 is a units row
    df = df[REQ + ["Club"]].apply(lambda c: pd.to_numeric(c, errors="coerce")
                                  if c.name != "Club" else c)
    df = df.dropna(subset=REQ)
    # physical-sanity gates (mirror ball_flight bounds; drop mishit junk like
    # the paper's <8 yd max-height filter)
    df = df[(df["Ball Speed"].between(30, 220)) & (df["Launch Angle"].between(1, 50))
            & (df["Spin Rate"].between(500, 13000))
            & (df["Max Height - Height"] * FT / YD >= 8 * 0.9144 / YD)  # >= 8 yd
            & (df["Carry Flat - Length"].between(20, 400))]
    return df.reset_index(drop=True)


def to_tensors(df: pd.DataFrame, dev):
    """Initial state + targets, all in meters / SI (paper coordinates:
    y toward target, x right of target, z up)."""
    v = torch.tensor(df["Ball Speed"].values * MPH, dtype=torch.float32)
    la = torch.deg2rad(torch.tensor(df["Launch Angle"].values, dtype=torch.float32))
    az = torch.deg2rad(torch.tensor(df["Launch Direction"].values, dtype=torch.float32))
    w = torch.tensor(df["Spin Rate"].values * RPM, dtype=torch.float32)
    ax = torch.deg2rad(torch.tensor(df["Spin Axis"].values, dtype=torch.float32))
    v0 = torch.stack([v * torch.cos(la) * torch.sin(az),
                      v * torch.cos(la) * torch.cos(az),
                      v * torch.sin(la)], 1)
    # paper §3.1: w_x = |w| cos(psi) (back), w_z = -|w| sin(psi) (side)
    w0 = torch.stack([w * torch.cos(ax), torch.zeros_like(w), -w * torch.sin(ax)], 1)
    tgt = torch.stack([torch.tensor(df["Carry Flat - Side"].values * YD, dtype=torch.float32),
                       torch.tensor(df["Carry Flat - Length"].values * YD, dtype=torch.float32)], 1)
    hmax = torch.tensor(df["Max Height - Height"].values * FT, dtype=torch.float32)
    return v0.to(dev), w0.to(dev), tgt.to(dev), hmax.to(dev)


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #

class CoeffNet(nn.Module):
    """Paper base MLP: 256/128 ReLU; outputs CL', CD', CQ' (pM-cQ variant)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6, 256), nn.ReLU(),
                                 nn.Linear(256, 128), nn.ReLU(),
                                 nn.Linear(128, 3))
        self.pM = nn.Parameter(torch.tensor(0.01))

    def forward(self, v, w):
        x = torch.cat([v / 89.4, w / 2094.0], 1)
        cl, cd, cq = self.net(x).unbind(1)
        return torch.sigmoid(cl) / 2, torch.sigmoid(cd) / 2, torch.sigmoid(cq)


def simulate_batch(model: CoeffNet | None, v0, w0):
    """Differentiable Euler sim. model=None -> shipped Phys-Q polynomials.
    Returns (r_land [B,2] x/y, apex [B])."""
    B = v0.shape[0]
    v, w = v0.clone(), w0.clone()
    r = torch.zeros(B, 3, device=v0.device); r[:, 2] = 0.01
    zs, xs, ys = [], [], []
    for _ in range(N_STEPS):
        vmag = v.norm(dim=1, keepdim=True).clamp_min(1e-6)
        wmag = w.norm(dim=1, keepdim=True).clamp_min(1e-6)
        S = (R * wmag / vmag).squeeze(1)
        if model is None:
            CD = 0.1304 + 0.9287 * S - 0.8259 * S * S
            CL = 0.0504 + 1.2031 * S - 1.1490 * S * S
            CM = 0.01 * S
            q = 0.5 * RHO * vmag.squeeze(1) ** 2
        else:
            CL, CD, CQ = model(v, w)
            CM = model.pM * S
            q = CQ * RHO * vmag.squeeze(1) ** 2          # paper eq. 28
        lift_dir = torch.cross(w, v, dim=1)
        lift_dir = lift_dir / lift_dir.norm(dim=1, keepdim=True).clamp_min(1e-6)
        F = (CL * q * A).unsqueeze(1) * lift_dir \
            - (CD * q * A).unsqueeze(1) * (v / vmag)
        F = F - torch.tensor([0.0, 0.0, M * G], device=v.device)
        T = (CM * q * D * A).unsqueeze(1) * (w / wmag)
        v = v + DT * F / M
        w = w - DT * T / I
        r = r + DT * v
        zs.append(r[:, 2]); xs.append(r[:, 0]); ys.append(r[:, 1])
    Z = torch.stack(zs, 1)                                # [B, N]
    X, Y = torch.stack(xs, 1), torch.stack(ys, 1)
    i_start = int(1.0 / DT)                               # paper's i_s = t=1s
    logits = -Z[:, i_start:].abs() * 20.0                 # soft-argmax sharpness
    wgt = torch.softmax(logits, 1)
    x_land = (wgt * X[:, i_start:]).sum(1)
    y_land = (wgt * Y[:, i_start:]).sum(1)
    return torch.stack([x_land, y_land], 1), Z.max(1).values


def eval_landing(model, v0, w0, tgt, hmax, bs=2048):
    errs, herrs = [], []
    with torch.no_grad():
        for i in range(0, v0.shape[0], bs):
            land, apex = simulate_batch(model, v0[i:i+bs], w0[i:i+bs])
            errs.append((land - tgt[i:i+bs]).norm(dim=1))
            herrs.append((apex - hmax[i:i+bs]).abs())
    e = torch.cat(errs) / YD
    h = torch.cat(herrs) / YD
    return e.mean().item(), h.mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--out", default=str(Path(__file__).parent.parent /
                                         "Data" / "coaching" / "ball_flight_nn.json"))
    args = ap.parse_args()

    torch.manual_seed(0)
    dev = "cpu"
    df = load(Path(args.csv))
    n_val = max(1, len(df) // 10)
    train_df, val_df = df.iloc[:-n_val], df.iloc[-n_val:]   # temporal split (paper)
    print(f"[data] {len(df)} shots after cleaning -> {len(train_df)} train / {len(val_df)} val")
    print(f"[data] clubs: {df['Club'].value_counts().to_dict()}")

    v0t, w0t, tgtt, hmt = to_tensors(train_df, dev)
    v0v, w0v, tgtv, hmv = to_tensors(val_df, dev)

    base_p, base_h = eval_landing(None, v0v, w0v, tgtv, hmv)
    print(f"[baseline] shipped Phys-Q: mean landing err {base_p:.2f} yd, height err {base_h:.2f} yd")

    model = CoeffNet().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best = (float("inf"), None)
    for ep in range(1, args.epochs + 1):
        perm = torch.randperm(v0t.shape[0])
        tot = 0.0
        model.train()
        for i in range(0, len(perm), args.batch):
            idx = perm[i:i + args.batch]
            land, apex = simulate_batch(model, v0t[idx], w0t[idx])
            loss = (land - tgtt[idx]).norm(dim=1).mean() + (apex - hmt[idx]).abs().mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * len(idx)
        vp, vh = eval_landing(model, v0v, w0v, tgtv, hmv)
        if vp < best[0]:
            best = (vp, {k: v.detach().cpu().numpy().tolist() for k, v in model.state_dict().items()})
        print(f"  epoch {ep:3d}: train loss {tot/len(perm)/YD:6.2f} yd | "
              f"val landing {vp:5.2f} yd, height {vh:5.2f} yd {'*' if vp == best[0] else ''}")

    print(f"\n[result] best val landing error: Phys-NN {best[0]:.2f} yd vs shipped Phys-Q {base_p:.2f} yd")
    if best[0] < base_p:
        w = best[1]
        out = {
            "_about": "Phys-NN coefficients net (McNally CVPR'23W, pM-cQ variant) trained on "
                      "tim-blackmore/launch-monitor-regression TrackMan data",
            "val_landing_err_yd": round(best[0], 2),
            "baseline_physq_err_yd": round(base_p, 2),
            "layers": [[w["net.0.weight"], w["net.0.bias"]],
                       [w["net.2.weight"], w["net.2.bias"]],
                       [w["net.4.weight"], w["net.4.bias"]]],
            "pM": w["pM"],
        }
        Path(args.out).write_text(json.dumps(out))
        print(f"[result] NN BEATS baseline -> exported weights to {args.out}")
    else:
        print("[result] NN does not beat the shipped Phys-Q baseline -> keeping Phys-Q (per plan).")


if __name__ == "__main__":
    main()
