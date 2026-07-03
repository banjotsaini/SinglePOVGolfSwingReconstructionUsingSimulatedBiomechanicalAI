"""Derive per-indicator confidence weights from Austin Lee's depth-reliability EDA.

Austin's `pose_consistency_by_player_view.csv` (AL EDA Investigation/) holds
per-(player, view, event, joint) 3D landmark positions across 86 driver clips.
The within-player standard deviation of the DEPTH (z) coordinate at a given
(event, joint) is an empirical measure of how reliably that joint's depth can
be recovered from single-POV video — the exact thing that makes our rotation
indicators trustworthy or not.

We aggregate that into a reliability score per (event, joint), then map each of
our 15 coaching indicators onto the joints+events it depends on to produce a
confidence weight in [0,1]. The scorecard uses this to label / gate feedback so
we never raise a coaching note off an unreliable measurement.

Attribution: depth-reliability empirics are from Austin Lee's Phase-2 driver EDA.
Caveat: his sample is 3 players / 86 clips (driver) — this is a reliability
PRIOR, not a population statistic. Documented in
Checkpoint 1/reports/AUSTIN_FINDINGS_INTEGRATION.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
AUSTIN_CSV = PROJECT_ROOT / "AL EDA Investigation" / "outputs" / "driver_eda" / "pose_consistency_by_player_view.csv"
OUT = PROJECT_ROOT / "Data" / "coaching" / "indicator_confidence.json"

# MediaPipe-33 indices (Austin's joint_idx convention)
MP = dict(nose=0, l_sho=11, r_sho=12, l_elb=13, r_elb=14, l_wri=15, r_wri=16,
          l_hip=23, r_hip=24, l_kne=25, r_kne=26, l_ank=27, r_ank=28)

# Each indicator -> the (event_name, [joint_idxs]) it depends on. Confidence
# follows the depth-reliability of these joints at these events.
INDICATOR_DEPS = {
    "shoulder_turn_top_deg":     [("Top", [MP["l_sho"], MP["r_sho"]]), ("Address", [MP["l_sho"], MP["r_sho"]])],
    "hip_turn_top_deg":          [("Top", [MP["l_hip"], MP["r_hip"]]), ("Address", [MP["l_hip"], MP["r_hip"]])],
    "x_factor_top_deg":          [("Top", [MP["l_sho"], MP["r_sho"], MP["l_hip"], MP["r_hip"]])],
    "hip_turn_impact_deg":       [("Impact", [MP["l_hip"], MP["r_hip"]])],
    "spine_tilt_address_deg":    [("Address", [MP["l_sho"], MP["r_sho"], MP["l_hip"], MP["r_hip"]])],
    "spine_tilt_impact_deg":     [("Impact", [MP["l_sho"], MP["r_sho"], MP["l_hip"], MP["r_hip"]])],
    "posture_loss_deg":          [("Address", [MP["l_sho"], MP["r_sho"]]), ("Impact", [MP["l_sho"], MP["r_sho"]])],
    "head_sway_max_pct":         [("Top", [MP["nose"]]), ("Impact", [MP["nose"]])],
    "head_lift_max_pct":         [("Top", [MP["nose"]]), ("Mid-downswing", [MP["nose"]])],
    "left_arm_bend_top_deg":     [("Top", [MP["l_sho"], MP["l_elb"], MP["l_wri"]])],
    "right_arm_bend_top_deg":    [("Top", [MP["r_sho"], MP["r_elb"], MP["r_wri"]])],
    "lead_knee_flex_address_deg":[("Address", [MP["l_hip"], MP["l_kne"], MP["l_ank"]])],
    "lead_knee_flex_impact_deg": [("Impact", [MP["l_hip"], MP["l_kne"], MP["l_ank"]])],
    "hip_lateral_shift_pct":     [("Address", [MP["l_hip"], MP["r_hip"]]), ("Impact", [MP["l_hip"], MP["r_hip"]])],
    "tempo_ratio":               [],  # pure event-timing, no joint depth involved -> always high confidence
}


def build_reliability_table() -> pd.DataFrame:
    """Per-(event_name, joint_idx): mean within-player std of the z (depth)
    coordinate. Lower = more reliable depth."""
    df = pd.read_csv(AUSTIN_CSV)
    # within-player-view std of z across that player's clips, then average
    g = (df.groupby(["player", "view", "event_name", "joint_idx"])["z"]
           .std()
           .reset_index()
           .groupby(["event_name", "joint_idx"])["z"]
           .mean()
           .reset_index()
           .rename(columns={"z": "z_std"}))
    return g


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rel = build_reliability_table()
    rel_lookup = {(r.event_name, int(r.joint_idx)): float(r.z_std) for r in rel.itertuples(index=False)}

    # Normalize z_std -> confidence in [0,1]. Use the global distribution of
    # z_std across all (event,joint) to set the scale: best (lowest std)->1,
    # worst (highest)->~0, via a robust min-max on log scale.
    all_std = np.array(list(rel_lookup.values()))
    lo, hi = np.percentile(all_std, 5), np.percentile(all_std, 95)

    def to_conf(z_std: float) -> float:
        c = 1.0 - (z_std - lo) / (hi - lo + 1e-9)
        return float(np.clip(c, 0.05, 1.0))

    confidence = {}
    for ind, deps in INDICATOR_DEPS.items():
        if not deps:  # tempo: timing only
            confidence[ind] = {"confidence": 1.0, "tier": "high",
                                "basis": "event-timing only (no depth)"}
            continue
        stds = []
        for event_name, joints in deps:
            for j in joints:
                if (event_name, j) in rel_lookup:
                    stds.append(rel_lookup[(event_name, j)])
        if not stds:
            confidence[ind] = {"confidence": 0.5, "tier": "med", "basis": "no reliability data"}
            continue
        # an indicator is only as reliable as its WORST constituent joint/event
        worst = max(stds)
        conf = to_conf(worst)
        tier = "high" if conf >= 0.66 else ("med" if conf >= 0.4 else "low")
        confidence[ind] = {"confidence": round(conf, 3), "tier": tier,
                            "worst_joint_z_std": round(worst, 4)}

    with open(OUT, "w") as f:
        json.dump({
            "_source": "Austin Lee Phase-2 driver EDA (pose_consistency_by_player_view.csv)",
            "_caveat": "Reliability prior from 3 players / 86 driver clips; not a population statistic.",
            "indicators": confidence,
        }, f, indent=1)

    print(f"[reliability] wrote {OUT}")
    print(f"{'indicator':30} {'conf':>5}  tier")
    for ind, c in sorted(confidence.items(), key=lambda kv: kv[1]["confidence"]):
        print(f"  {ind:30} {c['confidence']:.2f}  {c['tier']}")


if __name__ == "__main__":
    main()
