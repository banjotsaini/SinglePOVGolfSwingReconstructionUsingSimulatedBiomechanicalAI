# What the z-depth optimization changes on real GolfDB clips

Sampled 6 real-time driver clips (face-on + down-the-line): [0, 2, 8, 10, 269, 830].

MediaPipe world landmarks -> baseline (Procrustes-aligned) vs optimized (+ Savitzky-Golay + learned MLP). GolfDB has no depth ground truth, so these are *effect* magnitudes; the accuracy improvement (106 -> 33 mm) is measured on the Vicon test set (`outputs/comparison/`).

## 1. The correction acts on the HEIGHT axis only

| Axis | Mean |Δ| (baseline → optimized) |
|---|---|
| X (left/right) | 0.0 mm |
| Y (depth, toward/away camera) | 0.0 mm |
| **Z (height/vertical)** | **110.3 mm** |

The Procrustes alignment leaves X and Y untouched and the SG+MLP stages edit only GolfPose-Z, which Austin's skeleton mapping defines as *height* (not camera depth). So the optimization is, precisely, a **vertical-axis** correction.

## 2. Effect on coaching indicators

| Indicator | Depends on | Mean |Δ| |
|---|---|---|
| shoulder turn | horizontal plane (rotation) | 0.000° |
| hip turn | horizontal plane (rotation) | 0.000° |
| spine tilt | vertical/height | 3.958° |

## Bottom line

- **Joint estimation:** improved on the height axis — this is exactly the Vicon-validated 106 → 33 mm (~3×) gain.
- **Coaching — rotation metrics** (shoulder turn, hip turn, X-factor): **unchanged (0.00°)**. They live in the camera-depth horizontal plane, which this optimization does not touch. These are the marquee swing-rotation metrics, so the optimization does *not* (yet) improve them.
- **Coaching — height metrics** (spine tilt, posture, head lift, vertical weight shift): **do shift** (~4.0° on spine tilt) and become more trustworthy.

**Implication for next step:** to improve the depth-rotation coaching metrics, the next optimization must target the GolfPose-Y (camera-depth) axis, which Austin's FINDINGS explicitly note was *not* evaluated.
