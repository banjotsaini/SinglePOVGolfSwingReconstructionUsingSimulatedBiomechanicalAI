"""Golf ball flight simulation — physics baseline from McNally et al. (CVPR 2023W).

Implements the paper's physics model ("Phys-Q" baseline): Newton + Magnus lift +
drag + spin-decay torque, forward-Euler integrated, with the published polynomial
aerodynamic coefficients from Ferguson, McNally & McPhee (2022,
doi 10.5703/1288284317493) parameterized by spin ratio S = R*|w|/|v|:

    CD = 0.1304 + 0.9287*S - 0.8259*S^2
    CL = 0.0504 + 1.2031*S - 1.1490*S^2
    CM = 0.01*S

The paper's Phys-NN variant (an MLP predicting the coefficients per time step)
is ALSO available here: trained on public TrackMan data (tim-blackmore/
launch-monitor-regression, 7.9k shots; Scripts/train_phys_nn.py) it reached a
1.84 yd held-out mean landing error vs 7.11 yd for Phys-Q on the same split.
Weights live in Data/coaching/ball_flight_nn.json. The NN only runs when (a)
numpy + the weights file are available and (b) the launch conditions sit inside
the training envelope (stored in the JSON) — everything else falls back to the
Phys-Q polynomials. The result's "engine" field says which one ran.

Sign conventions (right-handed golfer):
    +y = toward the target, +z = up, +x = right of the target line.
    sidespin_rpm > 0 curves the ball right (a fade); < 0 curves left (a draw).

Validated against TrackMan published tour averages (carry within ~2-5%):
driver 288 sim / 282 published, 7-iron 167/172, PW 135/136.

Pure stdlib — safe to import in the chat Lambda. Everything is deterministic.
"""
from __future__ import annotations

import math

# golf ball physical constants
BALL_MASS_KG = 0.04593
BALL_RADIUS_M = 0.021335
BALL_AREA_M2 = math.pi * BALL_RADIUS_M ** 2
BALL_INERTIA = 0.4 * BALL_MASS_KG * BALL_RADIUS_M ** 2   # 2/5 m R^2
AIR_DENSITY = 1.225
GRAVITY = 9.81

MPH_TO_MS = 0.44704
RPM_TO_RADS = 2 * math.pi / 60.0
M_TO_YD = 1.0936133

# sane physical bounds for inputs (launch monitors top out well inside these)
_BOUNDS = {
    "ball_speed_mph": (30.0, 220.0),
    "launch_angle_deg": (1.0, 50.0),
    "backspin_rpm": (500.0, 13000.0),
    "sidespin_rpm": (-4000.0, 4000.0),
}

# --------------------------------------------------------------------------- #
# Optional Phys-NN engine (paper's pM-cQ variant; see module docstring)
# --------------------------------------------------------------------------- #

_NN_PATH = __import__("pathlib").Path(__file__).parent.parent / "Data" / "coaching" / "ball_flight_nn.json"
_NN_CACHE: dict = {"loaded": False, "nn": None}


def _load_nn():
    """Weights + envelope, or None when numpy/the file is unavailable."""
    if _NN_CACHE["loaded"]:
        return _NN_CACHE["nn"]
    _NN_CACHE["loaded"] = True
    try:
        import json as _json
        import numpy as np
        raw = _json.loads(_NN_PATH.read_text(encoding="utf-8"))
        _NN_CACHE["nn"] = {
            "layers": [(np.array(W, dtype=np.float64), np.array(b, dtype=np.float64))
                       for W, b in raw["layers"]],
            "pM": float(raw["pM"]),
            "envelope": raw["envelope"],
            "np": np,
        }
    except Exception:
        _NN_CACHE["nn"] = None
    return _NN_CACHE["nn"]


def nn_in_envelope(ball_speed_mph: float, launch_angle_deg: float,
                   backspin_rpm: float, sidespin_rpm: float = 0.0) -> bool:
    """True when the launch conditions sit inside the NN's training envelope
    (ball STATE, not club — aerodynamics don't know which club was swung)."""
    nn = _load_nn()
    if nn is None:
        return False
    spin = math.hypot(backspin_rpm, sidespin_rpm)
    axis = math.degrees(math.atan2(sidespin_rpm, backspin_rpm))
    env = nn["envelope"]
    checks = [("ball_speed_mph", ball_speed_mph), ("launch_angle_deg", launch_angle_deg),
              ("spin_rpm", spin), ("spin_axis_deg", axis)]
    return all(env[k][0] <= v <= env[k][1] for k, v in checks)


def _simulate_nn(ball_speed_mph: float, launch_angle_deg: float, backspin_rpm: float,
                 sidespin_rpm: float = 0.0, azimuth_deg: float = 0.0,
                 trajectory_points: int = 0) -> dict:
    """NN-coefficient flight sim. dt matches the 0.1 s the network was TRAINED
    with (the learned coefficients compensate that integrator; see paper §3.5)."""
    nn = _load_nn()
    np = nn["np"]
    dt, tmax = 0.1, 10.0
    v = ball_speed_mph * MPH_TO_MS
    la, az = math.radians(launch_angle_deg), math.radians(azimuth_deg)
    vel = np.array([v * math.cos(la) * math.sin(az),
                    v * math.cos(la) * math.cos(az),
                    v * math.sin(la)])
    w = np.array([backspin_rpm * RPM_TO_RADS, 0.0, -sidespin_rpm * RPM_TO_RADS])
    r = np.array([0.0, 0.0, 0.01])
    apex, t = r[2], 0.0
    traj = [(r[1], r[2], r[0])]
    (W0, b0), (W1, b1), (W2, b2) = nn["layers"]
    while t < tmax:
        vmag = float(np.linalg.norm(vel)) or 1e-9
        wmag = float(np.linalg.norm(w))
        S = BALL_RADIUS_M * wmag / vmag
        x = np.concatenate([vel / 89.4, w / 2094.0])
        h = np.maximum(W0 @ x + b0, 0.0)
        h = np.maximum(W1 @ h + b1, 0.0)
        o = 1.0 / (1.0 + np.exp(-(W2 @ h + b2)))
        CL, CD, CQ = o[0] / 2, o[1] / 2, o[2]
        CM = nn["pM"] * S
        q = CQ * AIR_DENSITY * vmag * vmag          # paper eq. 28 (cQ variant)
        lift = np.cross(w, vel)
        lift_n = float(np.linalg.norm(lift)) or 1.0
        F = CL * q * BALL_AREA_M2 * lift / lift_n \
            - CD * q * BALL_AREA_M2 * vel / vmag \
            - np.array([0.0, 0.0, BALL_MASS_KG * GRAVITY])
        if wmag > 0:
            w = w - dt * (CM * q * (2 * BALL_RADIUS_M) * BALL_AREA_M2 / BALL_INERTIA) * (w / wmag)
        vel = vel + dt * F / BALL_MASS_KG
        r_prev = r.copy()
        r = r + dt * vel
        apex = max(apex, r[2])
        t += dt
        if r[2] <= 0.0 and t > 0.5:
            frac = r_prev[2] / (r_prev[2] - r[2])
            r = r_prev + frac * (r - r_prev)
            traj.append((r[1], 0.0, r[0]))
            break
        traj.append((r[1], r[2], r[0]))
    out = {"carry_yd": r[1] * M_TO_YD, "side_yd": r[0] * M_TO_YD,
           "apex_yd": apex * M_TO_YD, "flight_time_s": t}
    if trajectory_points > 0:
        step = max(1, len(traj) // trajectory_points)
        keep = traj[::step]
        if keep[-1] != traj[-1]:
            keep.append(traj[-1])
        out["trajectory"] = [[round(p[0] * M_TO_YD, 1), round(p[1] * M_TO_YD, 1),
                              round(p[2] * M_TO_YD, 1)] for p in keep]
    return out


def simulate_flight(ball_speed_mph: float, launch_angle_deg: float,
                    backspin_rpm: float, sidespin_rpm: float = 0.0,
                    azimuth_deg: float = 0.0, dt: float = 0.01,
                    max_time_s: float = 15.0, trajectory_points: int = 0) -> dict:
    """Integrate one ball flight; returns carry/side/apex/flight time (floats).

    With trajectory_points > 0, also returns ~that many [downrange_yd, height_yd,
    side_yd] samples spanning launch to landing (for UI rendering).

    Magnus lift acts along w x v (the paper writes v x w but uses the opposite
    spin-axis sign convention; w x v is what reproduces real tour carries —
    with v x w a driver flight dives at ~83 yd).
    """
    v = ball_speed_mph * MPH_TO_MS
    la, az = math.radians(launch_angle_deg), math.radians(azimuth_deg)
    vx = v * math.cos(la) * math.sin(az)
    vy = v * math.cos(la) * math.cos(az)
    vz = v * math.sin(la)
    # backspin about +x; positive sidespin (fade, curves right) needs spin about -z
    wx, wy, wz = backspin_rpm * RPM_TO_RADS, 0.0, -sidespin_rpm * RPM_TO_RADS
    x, y, z = 0.0, 0.0, 0.01                      # 1 cm tee height (paper §2.3.1)
    apex, t = z, 0.0
    traj = [(y, z, x)]
    while t < max_time_s:
        vmag = math.sqrt(vx * vx + vy * vy + vz * vz)
        wmag = math.sqrt(wx * wx + wy * wy + wz * wz)
        S = BALL_RADIUS_M * wmag / vmag if vmag > 0 else 0.0
        CD = 0.1304 + 0.9287 * S - 0.8259 * S * S
        CL = 0.0504 + 1.2031 * S - 1.1490 * S * S
        CM = 0.01 * S
        q = 0.5 * AIR_DENSITY * vmag * vmag
        cx = wy * vz - wz * vy
        cy = wz * vx - wx * vz
        cz = wx * vy - wy * vx
        cmag = math.sqrt(cx * cx + cy * cy + cz * cz) or 1.0
        FL = CL * q * BALL_AREA_M2
        FD = CD * q * BALL_AREA_M2
        fx = FL * cx / cmag - FD * vx / vmag
        fy = FL * cy / cmag - FD * vy / vmag
        fz = FL * cz / cmag - FD * vz / vmag - BALL_MASS_KG * GRAVITY
        torque = CM * q * (2 * BALL_RADIUS_M) * BALL_AREA_M2
        if wmag > 0:
            decay = dt * torque / BALL_INERTIA / wmag
            wx -= decay * wx
            wy -= decay * wy
            wz -= decay * wz
        vx += dt * fx / BALL_MASS_KG
        vy += dt * fy / BALL_MASS_KG
        vz += dt * fz / BALL_MASS_KG
        x_prev, y_prev, z_prev = x, y, z
        x += dt * vx
        y += dt * vy
        z += dt * vz
        apex = max(apex, z)
        t += dt
        if z <= 0.0 and t > 0.5:
            frac = z_prev / (z_prev - z)           # interpolate to ground plane
            x = x_prev + frac * (x - x_prev)
            y = y_prev + frac * (y - y_prev)
            traj.append((y, 0.0, x))
            break
        traj.append((y, z, x))
    out = {
        "carry_yd": y * M_TO_YD,
        "side_yd": x * M_TO_YD,
        "apex_yd": apex * M_TO_YD,
        "flight_time_s": t,
    }
    if trajectory_points > 0:
        step = max(1, len(traj) // trajectory_points)
        keep = traj[::step]
        if keep[-1] != traj[-1]:
            keep.append(traj[-1])                  # always end at the landing point
        out["trajectory"] = [[round(p[0] * M_TO_YD, 1), round(p[1] * M_TO_YD, 1),
                              round(p[2] * M_TO_YD, 1)] for p in keep]
    return out


# --------------------------------------------------------------------------- #
# Typical launch conditions by club and skill tier.
# tour/lpga rows are TrackMan published tour averages; amateur rows are
# mid-handicap approximations from the same public sources. Values are
# (ball_speed_mph, launch_angle_deg, backspin_rpm).
# --------------------------------------------------------------------------- #

CLUB_DEFAULTS: dict[str, dict[str, tuple[float, float, float]]] = {
    "driver":  {"tour": (167.0, 10.9, 2686), "lpga": (140.0, 13.2, 2611), "amateur": (133.0, 12.6, 3275)},
    "3_wood":  {"tour": (158.0, 9.2, 3655),  "lpga": (132.0, 11.2, 2704), "amateur": (125.0, 11.5, 3600)},
    "hybrid":  {"tour": (146.0, 10.2, 4437), "lpga": (122.0, 12.5, 4501), "amateur": (118.0, 12.0, 4500)},
    "5_iron":  {"tour": (132.0, 12.1, 5361), "lpga": (112.0, 14.8, 5081), "amateur": (108.0, 13.5, 5500)},
    "7_iron":  {"tour": (120.0, 16.3, 7097), "lpga": (104.0, 19.0, 6699), "amateur": (98.0, 18.0, 7000)},
    "9_iron":  {"tour": (109.0, 20.4, 8647), "lpga": (93.0, 23.9, 7589),  "amateur": (88.0, 23.0, 8200)},
    "wedge":   {"tour": (102.0, 24.2, 9304), "lpga": (86.0, 25.6, 8403),  "amateur": (80.0, 26.0, 8700)},
}

# scorecard meta.club vocabulary (GolfDB: "driver"/"iron") + natural phrasings
CLUB_ALIASES = {
    "iron": "7_iron", "mid iron": "7_iron", "7 iron": "7_iron", "seven iron": "7_iron",
    "wood": "3_wood", "3 wood": "3_wood", "fairway wood": "3_wood",
    "5 iron": "5_iron", "long iron": "5_iron",
    "9 iron": "9_iron", "short iron": "9_iron",
    "pitching wedge": "wedge", "pw": "wedge",
}

TIERS = ("tour", "lpga", "amateur")


def normalize_club(club: str | None) -> str | None:
    if not club:
        return None
    c = str(club).strip().lower().replace("-", " ").replace("_", " ")
    c_key = c.replace(" ", "_")
    if c_key in CLUB_DEFAULTS:
        return c_key
    return CLUB_ALIASES.get(c)


def estimate_for_club(club: str | None, tier: str = "tour",
                      overrides: dict | None = None,
                      speed_scale: float = 1.0) -> dict:
    """Simulate a typical flight for `club`, with optional launch overrides.

    speed_scale nudges the tier-default ball speed for what we measured of THIS
    swing (hand speed vs tour-typical). Clamped to ±12% — it is a low-confidence
    single-camera hint, not a launch monitor. A golfer-stated ball_speed_mph
    override always wins over the scale.

    Returns a JSON-able dict whose display numbers are pre-rounded — the chat
    grounding verifier matches answer numbers against tool-result numbers, so
    the tool must return exactly what the model should quote.
    """
    club_key = normalize_club(club)
    if club_key is None:
        return {"estimated": False,
                "note": "No club was recorded for this swing, so a flight can't be "
                        "simulated. Ask the golfer what club they hit (and any launch "
                        "numbers they know) and call this tool again with overrides."}
    if tier not in TIERS:
        tier = "tour"
    ball_speed, launch, backspin = CLUB_DEFAULTS[club_key][tier]
    sidespin = 0.0
    scale = min(max(float(speed_scale), 0.88), 1.12) if speed_scale else 1.0
    ball_speed *= scale

    used_overrides = []
    ov = overrides or {}
    for name, default in (("ball_speed_mph", ball_speed), ("launch_angle_deg", launch),
                          ("backspin_rpm", backspin), ("sidespin_rpm", sidespin)):
        val = ov.get(name)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            lo, hi = _BOUNDS[name]
            clamped = min(max(float(val), lo), hi)
            if name == "ball_speed_mph":
                ball_speed = clamped
            elif name == "launch_angle_deg":
                launch = clamped
            elif name == "backspin_rpm":
                backspin = clamped
            else:
                sidespin = clamped
            used_overrides.append(name)

    if "ball_speed_mph" in used_overrides:
        scale = 1.0                      # a stated ball speed makes the nudge moot
    use_nn = nn_in_envelope(ball_speed, launch, backspin, sidespin)
    sim = _simulate_nn if use_nn else simulate_flight
    r = sim(ball_speed, launch, backspin, sidespin, trajectory_points=24)
    side = round(r["side_yd"])
    shape = "straight" if abs(side) < 3 else ("right (fade side)" if side > 0 else "left (draw side)")
    return {
        "estimated": True,
        "club": club_key.replace("_", " "),
        "skill_tier": tier,
        "assumed_launch": {
            "ball_speed_mph": round(ball_speed, 1),
            "launch_angle_deg": round(launch, 1),
            "backspin_rpm": round(backspin),
            "sidespin_rpm": round(sidespin),
            "overridden_by_golfer": used_overrides,
        },
        "carry_yd": round(r["carry_yd"]),
        "side_yd": side,
        "curve": shape,
        "apex_yd": round(r["apex_yd"]),
        "flight_time_s": round(r["flight_time_s"], 1),
        # phys_nn = paper's deep-learning coefficients (trained on real TrackMan
        # shots, 1.8 yd val landing err); phys_q = published polynomial fallback
        # for launch conditions outside the NN's training envelope
        "engine": "phys_nn" if use_nn else "phys_q",
        "note": "Simulated with a physics model from typical launch conditions for this "
                "club — an estimate, not a measurement of the ball in the video.",
        # "_ui"-prefixed keys are stripped from what the model sees (and from the
        # grounding pool) but kept in tool_log so the web UI can draw the arc.
        "_ui_trajectory": r["trajectory"],
    }


if __name__ == "__main__":
    for club in CLUB_DEFAULTS:
        for tier in TIERS:
            e = estimate_for_club(club, tier)
            print(f"{club:8s} {tier:8s} carry={e['carry_yd']:>4d} yd  "
                  f"apex={e['apex_yd']:>3d} yd  t={e['flight_time_s']:.1f}s")
