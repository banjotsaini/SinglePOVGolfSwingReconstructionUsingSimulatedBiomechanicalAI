"""R6 prototype: physics-fit 3D uplift — fit simulate_flight launch params so the
projected trajectory matches the measured 2D ball track."""
import sys, math, csv
import numpy as np
sys.path.insert(0, r"C:\dev\golf-capstone\Scripts")
sys.path.insert(0, r"C:\Users\Banjo\AppData\Local\Temp\claude\C--dev-golf-capstone\8a3e149f-899d-44af-ba03-276ee427faa8\scratchpad")
from ball_flight import simulate_flight, CLUB_DEFAULTS
from proto_ball_track import detect, load_pose

YD = 0.9144  # m per yard

def golfer_height_px(pose, frame):
    # standing height: 95th percentile of per-frame pixel spans (impact frame is
    # crouched/rotated and under-measures, which inflates the depth anchor)
    spans = []
    for pts in pose.values():
        ys = [p[1] for p in pts]
        spans.append(max(ys) - min(ys))
    return float(np.percentile(spans, 95))

def fit_flight(track, impact, fps, origin_px, pose, W, H,
               f_px=910.0, golfer_m=1.75, horizon_y=None, club="driver"):
    """track: [(frame, x_px, y_px)]. Returns fitted launch + flight numbers."""
    cx, cy = W / 2, H / 2
    pitch = 0.0 if horizon_y is None else math.atan((cy - horizon_y) / f_px)
    # depth anchor: golfer height in px at impact
    h_px = golfer_height_px(pose, impact)
    Zg = f_px * golfer_m / h_px                     # camera->golfer depth (m)
    # tee 3D position in camera coords (x right, y down, z forward)
    ox, oy = origin_px
    T = np.array([(ox - cx) * Zg / f_px, (oy - cy) * Zg / f_px, Zg])

    spin_b, spin_s = CLUB_DEFAULTS[club]["amateur"][2], 0.0

    def project(params, times):
        speed, launch, azim = params
        r = simulate_flight(speed, launch, spin_b, spin_s, 0.0,
                            trajectory_points=240)
        traj = np.array(r["trajectory"])            # [downrange_yd, height_yd, side_yd]
        tt = np.linspace(0, r["flight_time_s"], len(traj))
        # launch-frame (m): x=downrange, y=up, z=side(right)
        P = np.stack([traj[:, 0] * YD, traj[:, 1] * YD, traj[:, 2] * YD], axis=1)
        # rotate downrange direction by azimuth around vertical, into camera frame:
        # camera frame: X right, Y down, Z forward (pitch-corrected)
        a = math.radians(azim)
        Xc = P[:, 0] * math.sin(a) + P[:, 2] * math.cos(a)
        Zc = P[:, 0] * math.cos(a) - P[:, 2] * math.sin(a)
        Yc = -P[:, 1]
        # apply camera pitch (rotate about X axis)
        cp, sp = math.cos(pitch), math.sin(pitch)
        Zc2 = Zc * cp - Yc * sp
        Yc2 = Zc * sp + Yc * cp
        pts = np.stack([Xc, Yc2, Zc2], axis=1) + T
        # sample at observation times
        out = []
        for t in times:
            k = np.searchsorted(tt, t)
            k = min(max(k, 1), len(tt) - 1)
            w = (t - tt[k - 1]) / (tt[k] - tt[k - 1] + 1e-9)
            p = pts[k - 1] * (1 - w) + pts[k] * w
            if p[2] <= 0.3:
                out.append((1e6, 1e6))
                continue
            out.append((cx + f_px * p[0] / p[2], cy + f_px * p[1] / p[2]))
        return np.array(out), r

    times = np.array([(f - impact) / fps for f, _, _ in track])
    obs = np.array([(x, y) for _, x, y in track])

    def loss(params):
        if not (30 <= params[0] <= 200 and 2 <= params[1] <= 45 and -60 <= params[2] <= 60):
            return 1e9
        proj, _ = project(params, times)
        return float(np.mean(np.linalg.norm(proj - obs, axis=1)))

    # coarse grid then Nelder-Mead
    best, bl = None, 1e18
    for s in (90, 110, 130, 150):
        for la in (8, 14, 20):
            for az in (-20, -5, 5, 20):
                l = loss((s, la, az))
                if l < bl:
                    best, bl = [s, la, az], l
    try:
        from scipy.optimize import minimize
        res = minimize(loss, best, method="Nelder-Mead",
                       options={"maxiter": 400, "xatol": 0.2, "fatol": 0.05})
        best, bl = list(res.x), float(res.fun)
    except ImportError:
        pass
    proj, r = project(best, times)
    return {"ball_speed_mph": round(best[0], 1), "launch_deg": round(best[1], 1),
            "azimuth_deg": round(best[2], 1), "resid_px": round(bl, 1),
            "carry_yd": round(r["carry_yd"]), "apex_yd": round(r["apex_yd"]),
            "side_yd": round(r["side_yd"]), "time_s": round(r["flight_time_s"], 1),
            "pitch_deg": round(math.degrees(pitch), 1), "depth_m": round(Zg, 2)}

if __name__ == "__main__":
    jobs = [(r"C:\Users\Banjo\Downloads\IMG_3434.MOV", 287,
             r"C:\dev\golf-capstone\Data\demo\IMG_3434\IMG_3434_landmarks_2d.csv",
             "IMG_3434", 560.0),
            (r"C:\Users\Banjo\Downloads\IMG_8107.MOV", 130,
             r"C:\dev\golf-capstone\Data\demo\IMG_8107\IMG_8107_landmarks_2d.csv",
             "IMG_8107", 610.0)]
    for video, impact, csv2d, name, horizon in jobs:
        track, origin, _ = detect(video, impact, csv2d)
        pose = load_pose(csv2d)
        print(f"\n=== {name}: track {len(track)} pts f{track[0][0]}..f{track[-1][0]}")
        for f_try, tag in ((910.0, "f=910"), (773.0, "f=773 (-15%)"), (1046.0, "f=1046 (+15%)")):
            fit = fit_flight(track, impact, 30.0, origin, pose, 720, 1280,
                             f_px=f_try, horizon_y=horizon)
            print(f"  [{tag}] {fit}")
