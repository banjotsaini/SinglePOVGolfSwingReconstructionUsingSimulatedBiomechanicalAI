"""Biomechanical swing indicators from single-POV 3D landmarks.

TASK #2 (indicator set) + #13 (what's credibly measurable) + #3 (extraction)
made concrete. Each indicator below was chosen because it is (a) meaningful
to a golfer and (b) actually recoverable from monocular 3D pose — the 3D is
what makes the *rotation* metrics (shoulder/hip turn, X-factor) possible at
all; 2D can't see rotation about the vertical axis.

Honest scope (per professor guidance — interpretive, NOT prescriptive):
  - These are INDICATORS derived from estimated 3D pose, not lab-grade
    measurements. We report them relative to the GolfDB pro distribution,
    never as absolute "correct" targets.
  - MotionBERT output is camera-relative + normalized, so absolute metric
    units aren't meaningful; everything is either an angle (degrees) or a
    body-scale-normalized ratio.

Coordinate convention (our MotionBERT parquet, COCO-17):
  x = left/right,  y = vertical (DOWN is +),  z = depth (toward/away camera)
  rotation about the vertical axis  ->  angle in the (x, z) horizontal plane

COCO-17 indices:
  0 nose 1 Leye 2 Reye 3 Lear 4 Rear 5 Lsho 6 Rsho 7 Lelb 8 Relb
  9 Lwri 10 Rwri 11 Lhip 12 Rhip 13 Lkne 14 Rkne 15 Lank 16 Rank
"""
from __future__ import annotations

import numpy as np

NOSE = 0
L_SHO, R_SHO = 5, 6
L_ELB, R_ELB = 7, 8
L_WRI, R_WRI = 9, 10
L_HIP, R_HIP = 11, 12
L_KNE, R_KNE = 13, 14
L_ANK, R_ANK = 15, 16

EVENTS = ["address", "toe_up", "mid_backswing", "top",
          "mid_downswing", "impact", "mid_follow_through", "finish"]


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------

def _angle(a, b, c) -> float:
    """Interior angle at vertex b of a-b-c, in degrees."""
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def _horizontal_angle(vec) -> float:
    """Angle (deg) of a vector projected onto the horizontal (x,z) plane.
    Used for rotation about the vertical axis."""
    return float(np.degrees(np.arctan2(vec[2], vec[0])))


def _signed_turn(angle_now: float, angle_ref: float) -> float:
    """Shortest signed angular difference, deg in (-180, 180]."""
    d = (angle_now - angle_ref + 180) % 360 - 180
    return abs(d)  # magnitude of turn


def _body_scale(frame_xyz) -> float:
    """Torso length (hip-center to shoulder-center) as a scale unit so
    displacements are body-relative (scale-invariant)."""
    hip_c = (frame_xyz[L_HIP] + frame_xyz[R_HIP]) / 2
    sho_c = (frame_xyz[L_SHO] + frame_xyz[R_SHO]) / 2
    return float(np.linalg.norm(sho_c - hip_c) + 1e-6)


# ---------------------------------------------------------------------------
# main: indicators for one clip
# ---------------------------------------------------------------------------

def compute_indicators(xyz: np.ndarray, events_local: np.ndarray) -> dict:
    """xyz: (T, 17, 3) 3D landmarks. events_local: 8 frame indices.
    Returns a flat dict of named indicators."""
    ev = {name: int(np.clip(events_local[i], 0, xyz.shape[0] - 1))
          for i, name in enumerate(EVENTS)}

    def F(event):           # frame at an event
        return xyz[ev[event]]

    scale = _body_scale(F("address"))
    out: dict[str, float] = {}

    # ---- rotation metrics (the 3D-only payoff) ----
    sho_vec = lambda fr: fr[R_SHO] - fr[L_SHO]
    hip_vec = lambda fr: fr[R_HIP] - fr[L_HIP]
    sho_addr = _horizontal_angle(sho_vec(F("address")))
    hip_addr = _horizontal_angle(hip_vec(F("address")))

    out["shoulder_turn_top_deg"] = _signed_turn(_horizontal_angle(sho_vec(F("top"))), sho_addr)
    out["hip_turn_top_deg"]      = _signed_turn(_horizontal_angle(hip_vec(F("top"))), hip_addr)
    out["x_factor_top_deg"]      = out["shoulder_turn_top_deg"] - out["hip_turn_top_deg"]
    out["hip_turn_impact_deg"]   = _signed_turn(_horizontal_angle(hip_vec(F("impact"))), hip_addr)

    # ---- posture / spine ----
    def spine_tilt(fr):
        hip_c = (fr[L_HIP] + fr[R_HIP]) / 2
        sho_c = (fr[L_SHO] + fr[R_SHO]) / 2
        spine = sho_c - hip_c
        vertical = np.array([0.0, -1.0, 0.0])  # up (y is down-positive)
        cos = np.dot(spine, vertical) / (np.linalg.norm(spine) + 1e-9)
        return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
    out["spine_tilt_address_deg"] = spine_tilt(F("address"))
    out["spine_tilt_impact_deg"]  = spine_tilt(F("impact"))
    out["posture_loss_deg"]       = abs(out["spine_tilt_impact_deg"] - out["spine_tilt_address_deg"])

    # ---- head movement (sway + lift), body-scale normalized ----
    head_addr = F("address")[NOSE]
    sway = []   # horizontal displacement through the swing
    lift = []   # vertical
    for e in ["toe_up", "top", "mid_downswing", "impact"]:
        d = F(e)[NOSE] - head_addr
        sway.append(np.linalg.norm([d[0], d[2]]))
        lift.append(abs(d[1]))
    out["head_sway_max_pct"] = float(max(sway) / scale * 100)
    out["head_lift_max_pct"] = float(max(lift) / scale * 100)

    # ---- lead-arm bend at top (both arms; lead picked downstream by handedness) ----
    out["left_arm_bend_top_deg"]  = _angle(F("top")[L_SHO], F("top")[L_ELB], F("top")[L_WRI])
    out["right_arm_bend_top_deg"] = _angle(F("top")[R_SHO], F("top")[R_ELB], F("top")[R_WRI])

    # ---- knee flex maintenance ----
    def knee(fr, side):
        h, k, a = (L_HIP, L_KNE, L_ANK) if side == "L" else (R_HIP, R_KNE, R_ANK)
        return _angle(fr[h], fr[k], fr[a])
    out["lead_knee_flex_address_deg"] = knee(F("address"), "L")
    out["lead_knee_flex_impact_deg"]  = knee(F("impact"), "L")

    # ---- weight shift proxy: lateral hip-center travel address->impact ----
    hip_c_addr = (F("address")[L_HIP] + F("address")[R_HIP]) / 2
    hip_c_imp  = (F("impact")[L_HIP] + F("impact")[R_HIP]) / 2
    out["hip_lateral_shift_pct"] = float(np.linalg.norm(
        [(hip_c_imp - hip_c_addr)[0], (hip_c_imp - hip_c_addr)[2]]) / scale * 100)

    # ---- tempo: backswing : downswing frame ratio ----
    backswing = max(1, ev["top"] - ev["address"])
    downswing = max(1, ev["impact"] - ev["top"])
    out["tempo_ratio"] = float(backswing / downswing)

    return out


INDICATOR_NAMES = [
    "shoulder_turn_top_deg", "hip_turn_top_deg", "x_factor_top_deg", "hip_turn_impact_deg",
    "spine_tilt_address_deg", "spine_tilt_impact_deg", "posture_loss_deg",
    "head_sway_max_pct", "head_lift_max_pct",
    "left_arm_bend_top_deg", "right_arm_bend_top_deg",
    "lead_knee_flex_address_deg", "lead_knee_flex_impact_deg",
    "hip_lateral_shift_pct", "tempo_ratio",
]
