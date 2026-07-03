"""Train a 1D-CNN swing-event detector on MotionBERT-Full 3D trajectories.

This is the 'LLM-Augmented Pipeline' made real: instead of the heuristic
wrist-Y argmin/argmax detector (PCE@5 = 0.170), we train a temporal CNN
that takes the 3D landmark trajectory and predicts the 8 swing events
per frame — the SwingNet idea, but operating on our 3D landmarks instead
of raw video frames.

Pipeline:
  input  = MotionBERT-Full 3D landmarks (T, 17, 3) + velocities  -> (T, 102)
  target = per-frame class in {0=no-event, 1..8=the 8 swing events}
  model  = dilated 1D-CNN, per-frame 9-way softmax
  eval   = for each event class, argmax frame -> PCE@1/3/5

Train/test split uses GolfDB's `split` column (player-grouped, no leakage).
"""
from __future__ import annotations

import sys
import argparse
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from eval_utils import SWING_EVENTS

PROJECT_ROOT = Path(__file__).parent.parent
# Which lifter's cached 3D to train on. Overridable via --lifter so we can
# retrain/compare detectors across lifters (e.g. motionbert vs golfpose MixSTE).
CACHE_3D = PROJECT_ROOT / "Data" / "eval_runs" / "motionbert_full_from_mediapipe_lite"
GOLFDB_PKL = PROJECT_ROOT / "golfdb" / "golfDB.pkl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_JOINTS = 17
N_CLASSES = 9          # 0 = no-event, 1..8 = swing events
TOLERANCES = (1, 3, 5)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def events_clip_local(ev) -> np.ndarray:
    ev = np.asarray(ev)
    return ev[1:9] - ev[0]


def load_clip_xyz(clip_id: int) -> np.ndarray | None:
    p = CACHE_3D / f"{clip_id}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    T = int(df["frame"].max()) + 1
    xyz = np.zeros((T, N_JOINTS, 3), dtype=np.float32)
    for r in df.itertuples(index=False):
        if 0 <= r.kp_idx < N_JOINTS:
            xyz[r.frame, r.kp_idx] = (r.x, r.y, r.z)
    return xyz


def build_features(xyz: np.ndarray) -> np.ndarray:
    """(T,17,3) -> (T, 102): position (root-relative) + velocity."""
    T = xyz.shape[0]
    # root-relative (hip_center is COCO idx... we use mean of hips since COCO)
    # The cache is COCO-17: left_hip=11, right_hip=12
    root = (xyz[:, 11] + xyz[:, 12]) / 2.0
    pos = xyz - root[:, None, :]
    pos = pos.reshape(T, -1)                      # (T, 51)
    vel = np.zeros_like(pos)
    vel[1:] = pos[1:] - pos[:-1]                  # (T, 51)
    return np.concatenate([pos, vel], axis=1)     # (T, 102)


def build_targets(T: int, events_local: np.ndarray) -> np.ndarray:
    """Per-frame class labels. Frame at event[i] gets class i+1; rest 0."""
    y = np.zeros(T, dtype=np.int64)
    for i, ef in enumerate(events_local):
        if 0 <= ef < T:
            y[int(ef)] = i + 1
    return y


def load_dataset():
    df = pd.read_pickle(GOLFDB_PKL).set_index("id")
    samples = []
    for cid in df.index:
        xyz = load_clip_xyz(int(cid))
        if xyz is None or xyz.shape[0] < 30:
            continue
        ev = events_clip_local(df.loc[cid, "events"])
        if (ev < 0).any() or (ev >= xyz.shape[0]).any():
            continue
        feats = build_features(xyz)
        tgt = build_targets(xyz.shape[0], ev)
        samples.append({
            "clip_id": int(cid),
            "split": int(df.loc[cid, "split"]),
            "feats": feats,
            "target": tgt,
            "events": ev,
        })
    return samples


# ---------------------------------------------------------------------------
# Model — dilated temporal 1D-CNN (TCN-lite), no batchnorm (batch=1 friendly)
# ---------------------------------------------------------------------------

class EventTCN(nn.Module):
    def __init__(self, in_ch=102, hidden=128, n_classes=N_CLASSES, n_blocks=5, p_drop=0.2):
        super().__init__()
        self.inp = nn.Conv1d(in_ch, hidden, kernel_size=1)
        blocks = []
        for i in range(n_blocks):
            d = 2 ** i
            blocks.append(nn.Sequential(
                nn.Conv1d(hidden, hidden, kernel_size=3, padding=d, dilation=d),
                nn.GELU(),
                nn.GroupNorm(8, hidden),
                nn.Dropout(p_drop),
            ))
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Conv1d(hidden, n_classes, kernel_size=1)

    def forward(self, x):                 # x: (B, in_ch, T)
        h = self.inp(x)
        for blk in self.blocks:
            h = h + blk(h)                # residual
        return self.head(h)              # (B, n_classes, T)


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

def predict_events(logits: np.ndarray, T: int) -> dict:
    """logits: (n_classes, T). For each event class 1..8, argmax frame."""
    probs = logits  # already softmax-able; argmax over time per channel
    out = {}
    for i, name in enumerate(SWING_EVENTS):
        ch = i + 1
        out[name] = int(np.argmax(probs[ch]))
    return out


def score_pce(pred: dict, events_local: np.ndarray) -> dict:
    hits = {t: 0 for t in TOLERANCES}
    for i, name in enumerate(SWING_EVENTS):
        err = abs(pred[name] - int(events_local[i]))
        for t in TOLERANCES:
            if err <= t:
                hits[t] += 1
    return {t: hits[t] / 8.0 for t in TOLERANCES}


def main():
    global CACHE_3D
    ap = argparse.ArgumentParser()
    ap.add_argument("--lifter", default="motionbert_full_from_mediapipe_lite",
                    help="eval_runs cache dir name to train on")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "Models" / "event_detector_tcn.pt"))
    ap.add_argument("--tag", default=None, help="leaderboard model tag; default derived from lifter")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--no-leaderboard", action="store_true")
    args = ap.parse_args()

    CACHE_3D = PROJECT_ROOT / "Data" / "eval_runs" / args.lifter
    tag = args.tag or f"event_cnn_from_{args.lifter.split('_from_')[0]}"

    torch.manual_seed(0)
    np.random.seed(0)
    print(f"[train] device={DEVICE}")
    print(f"[train] loading dataset (3D cache: {args.lifter} + GolfDB labels)...")
    samples = load_dataset()
    print(f"[train] {len(samples)} usable clips")

    # GolfDB split: test = split 1, train = 2/3/4 (player-grouped, no leakage)
    train = [s for s in samples if s["split"] != 1]
    test  = [s for s in samples if s["split"] == 1]
    print(f"[train] train={len(train)}  test={len(test)}")

    # Class weights — events are 8 frames out of ~300, hugely imbalanced
    class_counts = np.zeros(N_CLASSES)
    for s in train:
        for c in range(N_CLASSES):
            class_counts[c] += (s["target"] == c).sum()
    weights = class_counts.sum() / (N_CLASSES * np.maximum(class_counts, 1))
    weights = np.clip(weights, 1.0, 200.0)
    w = torch.tensor(weights, dtype=torch.float32, device=DEVICE)
    print(f"[train] class weights (no-event vs events): {weights[0]:.2f} vs ~{weights[1:].mean():.1f}")

    model = EventTCN().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] model params: {n_params/1e6:.2f}M")

    N_EPOCHS = args.epochs
    for epoch in range(N_EPOCHS):
        model.train()
        np.random.shuffle(train)
        tot_loss = 0.0
        for s in train:
            x = torch.tensor(s["feats"].T[None], device=DEVICE)      # (1,102,T)
            y = torch.tensor(s["target"][None], device=DEVICE)       # (1,T)
            logits = model(x)                                        # (1,9,T)
            loss = F.cross_entropy(logits, y, weight=w)
            opt.zero_grad(); loss.backward(); opt.step()
            tot_loss += loss.item()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            # quick eval on test
            pce = evaluate(model, test)
            print(f"  epoch {epoch+1:2}/{N_EPOCHS}  loss={tot_loss/len(train):.3f}  "
                  f"PCE@5={pce[5]:.3f} PCE@3={pce[3]:.3f} PCE@1={pce[1]:.3f}")

    pce = evaluate(model, test, verbose=True)
    print()
    print("=" * 56)
    print(f"  TRAINED 1D-CNN EVENT DETECTOR — held-out test (n={len(test)})")
    print(f"    PCE@5 = {pce[5]:.3f}")
    print(f"    PCE@3 = {pce[3]:.3f}")
    print(f"    PCE@1 = {pce[1]:.3f}")
    print("=" * 56)
    print(f"  vs heuristic detector (same model's 3D): PCE@5 = 0.170")
    print(f"  vs Codex LLM (n=32):                      PCE@5 = 0.258")

    # Persist the model + write a leaderboard row
    out_model = Path(args.out)
    out_model.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), out_model)
    print(f"\n[train] saved model -> {out_model}")

    if not args.no_leaderboard:
        write_leaderboard_row(test, model, pce, tag=tag)


@torch.no_grad()
def evaluate(model, test, verbose=False):
    model.eval()
    agg = {t: [] for t in TOLERANCES}
    for s in test:
        x = torch.tensor(s["feats"].T[None], device=DEVICE)
        logits = model(x)[0].cpu().numpy()            # (9, T)
        pred = predict_events(logits, s["feats"].shape[0])
        pce = score_pce(pred, s["events"])
        for t in TOLERANCES:
            agg[t].append(pce[t])
    return {t: float(np.mean(agg[t])) for t in TOLERANCES}


def write_leaderboard_row(test, model, pce, tag="event_cnn_from_motionbert_full"):
    """Append per-clip rows for the trained detector into all_metrics.parquet
    so it shows up in the leaderboard alongside the other pipelines."""
    metrics_path = PROJECT_ROOT / "Data" / "all_metrics.parquet"
    if not metrics_path.exists():
        return
    df = pd.read_parquet(metrics_path)
    df = df[df["model"] != tag]  # idempotent

    rows = []
    model.eval()
    with torch.no_grad():
        for s in test:
            x = torch.tensor(s["feats"].T[None], device=DEVICE)
            logits = model(x)[0].cpu().numpy()
            pred = predict_events(logits, s["feats"].shape[0])
            p = score_pce(pred, s["events"])
            rows.append({
                "model": tag,
                "clip_id": s["clip_id"],
                "pce_at_5": p[5], "pce_at_3": p[3], "pce_at_1": p[1],
                "detection_rate": float("nan"), "bone_cv_mean": float("nan"),
                "jitter_mean_px": float("nan"), "implausible_frac_mean": float("nan"),
                "fps_inference": float("nan"),
            })
    new = pd.DataFrame(rows)
    combined = pd.concat([df, new], ignore_index=True)
    combined.to_parquet(metrics_path, index=False)
    print(f"[train] wrote {len(new)} rows to all_metrics.parquet as '{tag}'")


if __name__ == "__main__":
    main()
