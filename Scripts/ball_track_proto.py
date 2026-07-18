"""R5 prototype v3: median-background diff + golfer mask + direction-gated linking."""
import csv, math, time
import cv2
import numpy as np

def load_pose(csv_path):
    per_frame = {}
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            f = int(row["frame"])
            per_frame.setdefault(f, []).append((float(row["x"]), float(row["y"])))
    return per_frame

def detect(video, impact, csv2d, n_after=80):
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    hi = min(total, impact + n_after)
    grays = {}
    for i in range(max(0, impact - 8), hi):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        if ok:
            grays[i] = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    ref = np.float32(grays[impact])
    aligned = {}
    for i, g in grays.items():
        shift, _ = cv2.phaseCorrelate(ref, np.float32(g))
        M = np.float32([[1, 0, -shift[0]], [0, 1, -shift[1]]])
        aligned[i] = cv2.warpAffine(g, M, (W, H))
    frames_sorted = sorted(aligned)
    med = np.median(np.stack([aligned[i] for i in frames_sorted]), axis=0).astype(np.int16)

    pose = load_pose(csv2d)
    def golfer_bbox(f):
        pts = pose.get(f) or pose.get(min(pose, key=lambda k: abs(k - f)))
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        return (min(xs) - 70, min(ys) - 70, max(xs) + 70, max(ys) + 70)

    cands = {}
    prev = None
    for i in frames_sorted:
        d = np.abs(aligned[i].astype(np.int16) - med).astype(np.uint8)
        _, th = cv2.threshold(d, 16, 255, cv2.THRESH_BINARY)
        # moving foreground only: must ALSO differ from the previous frame
        if prev is not None:
            dc = np.abs(aligned[i].astype(np.int16) - prev).astype(np.uint8)
            _, thc = cv2.threshold(dc, 12, 255, cv2.THRESH_BINARY)
            thc = cv2.dilate(thc, np.ones((5, 5), np.uint8))
            th = cv2.bitwise_and(th, thc)
        prev = aligned[i].astype(np.int16)
        x0, y0, x1, y1 = golfer_bbox(i)
        th[max(0, int(y0)):int(y1), max(0, int(x0)):int(x1)] = 0
        th[int(0.62 * H):, :] = 0          # below horizon: mat/field clutter
        th[:int(0.06 * H), :] = 0          # net/roof top edge
        th[:, :12] = 0; th[:, -12:] = 0    # stabilization warp border bands
        n, _, stats, cents = cv2.connectedComponentsWithStats(th)
        out = []
        for j in range(1, n):
            bx, by, bw, bh, area = stats[j]
            if 2 <= area <= 80 and bw <= 16 and bh <= 16:
                out.append((float(cents[j][0]), float(cents[j][1]), int(area)))
        cands[i] = out

    kp_pose = pose.get(impact, [])
    # launch origin: ball at impact ~ feet level, offset toward camera side of stance
    ay = max(p[1] for p in kp_pose) if kp_pose else 0.75 * H
    wx = np.mean([p[0] for p in kp_pose]) if kp_pose else 0.5 * W
    origin = (wx, ay)

    def chase(f0, x0, y0):
        track = [(f0, x0, y0)]
        vel = None
        fi, xi, yi = f0, x0, y0
        gaps = 0
        for fn in range(f0 + 1, frames_sorted[-1] + 1):
            if fn not in cands:
                continue
            px, py = (xi + vel[0] * (fn - fi), yi + vel[1] * (fn - fi)) if vel else (xi, yi)
            gate = 45 if vel is None else max(8, 0.7 * math.hypot(*vel) * (fn - fi) + 6)
            pick, bd = None, gate
            for (x, y, a) in cands[fn]:
                dd = math.hypot(x - px, y - py)
                if dd < bd:
                    nv = ((x - xi) / (fn - fi), (y - yi) / (fn - fi))
                    if vel is None and nv[1] > -2:
                        continue
                    pick, bd = (x, y, nv), dd
            if pick:
                if math.hypot(*pick[2]) < 1.5:      # ball never hovers early
                    gaps += 1
                    if gaps > 8:
                        break
                    continue
                xi, yi, fi = pick[0], pick[1], fn
                vel = pick[2]
                track.append((fn, xi, yi))
                gaps = 0
            else:
                gaps += 1
                if gaps > 8:
                    break
        return track

    def truncate_ballistic(track):
        """Cut the track where it stops behaving like a receding ball:
        a stall (near-zero step speed) or a dominant-axis reversal = club/junk tail."""
        if len(track) < 4:
            return track
        steps = [((b[1] - a[1]) / (b[0] - a[0]), (b[2] - a[2]) / (b[0] - a[0]))
                 for a, b in zip(track, track[1:])]
        vx0 = np.mean([s[0] for s in steps[:3]]); vy0 = np.mean([s[1] for s in steps[:3]])
        ax = 0 if abs(vx0) >= abs(vy0) else 1
        sgn = 1 if (vx0 if ax == 0 else vy0) >= 0 else -1
        stall = rev = 0
        for k, s in enumerate(steps):
            stall = stall + 1 if math.hypot(*s) < 1.2 else 0
            rev = rev + 1 if s[ax] * sgn < 0 else 0
            if stall >= 3 or rev >= 3:
                return track[:k + 2 - max(stall, rev)]
        return track

    best, best_score = None, 1e18
    dbg = []
    for f0 in range(impact + 1, impact + 10):
        for (x0, y0, a0) in cands.get(f0, []):
            track = truncate_ballistic(chase(f0, x0, y0))
            if len(track) < 8:
                continue
            head = track[:8]
            sp = [math.hypot(b[1] - a[1], b[2] - a[2]) / (b[0] - a[0])
                  for a, b in zip(head, head[1:])]
            if sp[0] < 4 or sp[0] * 1.15 < sp[-1]:
                continue
            xs = np.array([p[1] for p in head]); ys = np.array([p[2] for p in head])
            if xs.ptp() >= ys.ptp():
                A = np.vstack([xs, np.ones_like(xs)]).T
                m, c = np.linalg.lstsq(A, ys, rcond=None)[0]
                rms = float(np.sqrt(np.mean((ys - (m * xs + c)) ** 2)))
            else:
                A = np.vstack([ys, np.ones_like(ys)]).T
                m, c = np.linalg.lstsq(A, xs, rcond=None)[0]
                rms = float(np.sqrt(np.mean((xs - (m * ys + c)) ** 2)))
            if rms > 5.0:
                continue
            # direction consistency: origin->start vs initial velocity within 50 deg
            ovx, ovy = x0 - origin[0], y0 - origin[1]
            tvx = np.mean([b[1] - a[1] for a, b in zip(head[:4], head[1:4])])
            tvy = np.mean([b[2] - a[2] for a, b in zip(head[:4], head[1:4])])
            dot = (ovx * tvx + ovy * tvy) / (math.hypot(ovx, ovy) * math.hypot(tvx, tvy) + 1e-9)
            if ovy > 0 or dot < math.cos(math.radians(40)):
                continue
            # anti-causality: OUR ball cannot exist before impact. Back-extrapolate
            # the launch velocity into pre-impact frames; if moving blobs sit there,
            # this is a neighbor bay's ball already in flight.
            f0_, vx0, vy0 = track[0][0], (head[1][1] - head[0][1]) / (head[1][0] - head[0][0]), \
                            (head[1][2] - head[0][2]) / (head[1][0] - head[0][0])
            pre_hits = 0
            for fb in range(impact - 6, impact):
                if fb not in cands:
                    continue
                k = f0_ - fb
                bxp, byp = x0 - vx0 * k, y0 - vy0 * k
                if math.hypot(bxp - origin[0], byp - origin[1]) < 220:
                    continue   # club region pre-impact, not a neighbor ball
                if any(math.hypot(x - bxp, y - byp) < 25 for x, y, a in cands[fb]):
                    pre_hits += 1
            if pre_hits >= 2:
                continue
            score = 10 * rms - 5 * min(len(track), 18) - 2 * sp[0] - 40 * dot
            dbg.append((f0, x0, y0, len(track), round(rms, 1), round(sp[0], 1), round(dot, 2), round(score)))
            if score < best_score:
                best, best_score = track, score
    return best, origin, dbg

GT = {
    "IMG_3434": {291: (652, 476), 292: (659, 445), 293: (665, 420),
                 294: (671, 400), 295: (676, 383), 296: (680, 369), 299: (690, 340),
                 300: (694, 333), 301: (696, 328), 305: (705, 314), 309: (714, 310)},
    "IMG_8107": {137: (568, 598), 139: (578, 578), 141: (586, 566),
                 143: (592, 556), 145: (598, 550)},
}

if __name__ == "__main__":
    jobs = [(r"C:\Users\Banjo\Downloads\IMG_3434.MOV", 287,
             r"C:\dev\golf-capstone\Data\demo\IMG_3434\IMG_3434_landmarks_2d.csv", "IMG_3434"),
            (r"C:\Users\Banjo\Downloads\IMG_8107.MOV", 130,
             r"C:\dev\golf-capstone\Data\demo\IMG_8107\IMG_8107_landmarks_2d.csv", "IMG_8107")]
    for video, impact, csv2d, name in jobs:
        t0 = time.time()
        track, origin, dbg = detect(video, impact, csv2d)
        dt = time.time() - t0
        print(f"\n=== {name}  impact={impact}  origin=({origin[0]:.0f},{origin[1]:.0f})  {dt:.1f}s")
        print(f"  candidates surviving gates: {len(dbg)}")
        for row in dbg[:6]:
            print("   ", row)
        if not track:
            print("  NO TRACK")
            continue
        print(f"  best track: {len(track)} pts, frames {track[0][0]}..{track[-1][0]}")
        for f, x, y in track[:16]:
            print(f"    f{f}  ({x:.0f},{y:.0f})")
        gt = GT[name]
        hits = sum(1 for f, x, y in track if f in gt and math.hypot(x - gt[f][0], y - gt[f][1]) <= 8)
        print(f"  GT overlap: {hits}/{len(gt)} within 8px")
