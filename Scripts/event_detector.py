"""Inference wrapper for the trained 1D-CNN swing-event detector.

Loads Models/event_detector_tcn.pt and predicts the 8 swing-event frames
from a clip's 3D landmark trajectory (COCO-17, the MotionBERT cache format).
This replaces the heuristic wrist-Y argmin/argmax detector in production.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)  # tiny model; avoids Windows thread-teardown segfaults

sys.path.insert(0, str(Path(__file__).parent))
from train_event_detector import EventTCN, build_features
from eval_utils import SWING_EVENTS

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_WEIGHTS = PROJECT_ROOT / "Models" / "event_detector_tcn.pt"


class EventDetector:
    def __init__(self, weights: str | Path = DEFAULT_WEIGHTS, device: str | None = None):
        # Tiny model (~0.26M params): CPU is instant and avoids CUDA teardown
        # segfaults seen when loading it standalone on Windows.
        self.device = device or "cpu"
        self.model = EventTCN()
        state = torch.load(str(weights), map_location="cpu")
        self.model.load_state_dict(state)
        self.model = self.model.to(self.device).eval()

    @torch.no_grad()
    def predict(self, xyz_coco: np.ndarray) -> dict:
        """xyz_coco: (T, 17, 3) COCO-17 3D landmarks. Returns
        {event_name: frame_idx} for the 8 swing events, monotonic-enforced."""
        feats = build_features(xyz_coco)                       # (T, 102)
        x = torch.tensor(feats.T[None], dtype=torch.float32, device=self.device)
        logits = self.model(x)[0].cpu().numpy()                # (9, T)
        T = xyz_coco.shape[0]

        # per-event argmax over time on its channel
        raw = {}
        for i, name in enumerate(SWING_EVENTS):
            raw[name] = int(np.argmax(logits[i + 1]))

        # enforce monotonic order (address < ... < finish). If a later event
        # lands before an earlier one, nudge it forward to preserve sequence.
        ordered = []
        prev = -1
        for name in SWING_EVENTS:
            f = max(raw[name], prev + 1)
            f = min(f, T - 1)
            ordered.append(f)
            prev = f
        return {name: ordered[i] for i, name in enumerate(SWING_EVENTS)}


if __name__ == "__main__":
    import argparse, pandas as pd
    p = argparse.ArgumentParser()
    p.add_argument("--clip", type=int, default=0)
    args = p.parse_args()
    cache = PROJECT_ROOT / "Data" / "eval_runs" / "golfpose3d_from_mediapipe_lite" / f"{args.clip}.parquet"
    df = pd.read_parquet(cache)
    T = int(df["frame"].max()) + 1
    xyz = np.zeros((T, 17, 3), dtype=np.float32)
    for r in df.itertuples(index=False):
        if 0 <= r.kp_idx < 17:
            xyz[r.frame, r.kp_idx] = (r.x, r.y, r.z)
    det = EventDetector()
    print(det.predict(xyz))
