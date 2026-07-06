# Can a learned correction stack on top of MixSTE? — residual-structure diagnostic

**Verdict: NO-GO** for a deterministic pose-conditioned residual MLP trained on the
current GolfPose Vicon data. The MixSTE residual is real, large, and depth-shaped —
but the part that a corrector could actually *learn and transfer* is small (≤19 %
of depth error even with oracle test labels) and its **sign does not transfer across
subjects** (r ≈ 0). Details and the one caveat that could change this below.

No training was done. Read-only. The default lifter path is untouched.

---

## What was built

- Golf-fine-tuned **MixSTE 17+0** (`golfpose_17plus0.bin`, pulled from S3
  `01_inputs/Models/golfpose/`) run on the **Vicon 2D ground-truth** keypoints for
  **all six subjects G1–G6** (38 swings), paired frame-for-frame with Vicon 3D GT.
- Predictions cached to `scratchpad/mixste_preds_G1-G6.pkl` (re-runs are instant).
- Residual `R = GT − aligned_MixSTE` after **per-frame Procrustes** (rotation+scale),
  which is what isolates genuine per-pose *shape* error (see quirk #2).
- Split matches the earlier baseline: **train G1–G4, held-out test G5–G6.**

### Validation gate (so the residuals are trustworthy)
- Checkpoint loads with **0 missing / 0 unexpected** keys against the official MixSTE2
  architecture (embed 512, depth 8, 17 joints, 243-frame window).
- **Procrustes-aligned MPJPE = 23.9 mm on held-out G5/G6** (PA-MPJPE, below the
  paper's 35.6 mm Protocol-1 number) — the weights behave. Raw MPJPE was 590 mm
  *before* alignment, which is the coordinate quirk, not a bug (below).

## Data quirks — checked, asserted
- **Subject scramble (image S5↔npz G6, S6↔npz G5):** avoided entirely. This
  diagnostic lifts the **npz 2D GT** and compares to **npz 3D GT**, both keyed by the
  same `G#`/swing, so 2D and 3D can't be mis-paired. The S↔G scramble only affects the
  *image-folder* pipeline (`golfpose_eval.py`), which is not used here. The 23.9 mm
  PA-MPJPE confirms 2D↔3D pairing is correct.
- **Coordinate-axis mismatch (quirk #2):** confirmed and quantified. MixSTE output
  sits in its own frame, **~112° rotated** from Vicon lab space per swing. That is why
  raw MPJPE is 590 mm but Procrustes MPJPE is 24 mm — a pure global rotation. All
  residual analysis is done **after** alignment.
- **Club keypoints:** the 17+0 checkpoint has **no club joints at all** (17 body
  joints only). Club depth cannot be assessed with this model; the **wrists are the
  closest club-control proxy** — and they are among the highest-error joints (below).

---

## Result 1 — the residual is real and it is camera-DEPTH (not structureless)

After per-frame Procrustes, on held-out G5/G6:

| quantity | value |
|---|---|
| residual energy along **camera depth** | **92.4 %** |
| mean depth residual | **20.5 mm** |
| mean in-plane residual | 6.4 mm |

Every body joint except the hip root carries **87–98 %** of its residual energy in the
camera viewing direction — the ill-posed monocular axis. So the naive "MixSTE already
absorbed all depth error" is **too strong**: there is a large, clearly depth-shaped
residual. *(fig1_depth_vs_inplane_perjoint.png)*

## Result 2 — it is joint- and phase-conditioned (the hypothesized pocket exists)

- **Worst joints (depth RMSE, test):** r_wrist 43.0, l_wrist 42.5, head 40.8,
  l_elbow 38.7, l_ankle 39.2 mm — **extremities and the head/spine chain.** Hips/root
  are ~5 mm. Exactly the distal, self-occluded joints.
- **Worst phase:** **top-of-backswing = 28.2 mm** and follow-through 25.2 mm, vs
  ~18–21 mm mid-swing — i.e. worst where self-occlusion is worst, as hypothesized.
  *(fig2_depth_by_phase.png)*

So far this reads like a GO. Result 3 is why it isn't.

## Result 3 — the structure does NOT transfer across subjects (the crux)

MixSTE was **trained on G1–G4**, so its in-sample depth residual there is only
**5.7 mm** — tiny and nearly structureless. The 20.5 mm on G5/G6 is a **generalization
gap**, and a corrector can only exploit it if the error pattern is subject-invariant.
It largely isn't:

| test | result |
|---|---|
| per-joint const depth debias, learn G1–G4 → apply G5/G6 | **−2.7 %** (slightly worse) |
| pose-conditioned (joint×phase) debias, leave-one-subject-out | **+8.4 % / −2.4 %** (a wash) |
| **signed** per-joint depth-bias correlation, train vs test | **r ≈ 0.0** |
| **oracle** joint×phase depth debias fit *on the test set itself* (upper bound) | **only −19 %** |

- The **sign** of the needed correction doesn't transfer (r ≈ 0): train biases are
  ~+2 mm; test biases are −12 to −25 mm. A corrector trained on the only clean
  supervision (G1–G4) learns the wrong direction for unseen subjects.
- What *does* transfer is only **which joints are bad** (unsigned depth-RMSE profile,
  **r = 0.87**) — useful for *where* to look, useless for *how much/which way* to move.
  *(fig3_transfer.png)*
- Even an **oracle** (joint×phase) corrector fit directly on the test data removes just
  **19 %** of depth RMSE (29.2 → 23.7 mm). The other ~81 % is finer-grained /
  per-sample variation not captured by a (joint, phase) function — effectively noise
  at the granularity a deterministic MLP would operate on.

The dominant error on unseen subjects behaves like a **subject/camera-level depth
mis-calibration**, not a pose-conditioned pattern a deterministic ΔZ MLP can fix from
this supervision.

---

## Verdict — NO-GO (with one caveat)

**NO-GO.** Do not build a deterministic pose-conditioned residual MLP on top of MixSTE
using this Vicon data. Evidence:
1. The only clean supervision (G1–G4) is *in-sample* for MixSTE → ~5.7 mm residual,
   almost nothing to learn from.
2. The large held-out residual (20.5 mm) is a subject-level depth offset whose **signed
   structure does not transfer** (r ≈ 0); simple transfer proxies net ≈ 0.
3. The **ceiling** for any joint×phase-conditioned corrector is ~19 % depth-RMSE
   reduction *even with oracle labels* — and depth-RMSE ≠ the marquee coaching angles,
   which are horizontal-plane rotations largely orthogonal to this depth error (see
   `outputs/coaching_accuracy/FINDINGS.md`). Expected joint-angle payoff is small.

**The honest pocket, if anything:** the transferable signal is *location* — distal
joints (wrists ≈ club control, elbows, head, lead ankle) at **top-of-backswing**. If
you want to pursue a correction, that is the only place with subject-invariant
structure, and the mechanism that fits the actual failure (a per-subject depth
mis-calibration) is **not** a pose→ΔZ MLP but either (a) per-sequence depth
anchoring/normalization, or (b) genuine multi-view / weak-depth supervision.

**Caveat that could flip this:** the split has only **2 held-out subjects (8 swings)**,
and 4 of 6 subjects are in-sample for MixSTE. That confound — not a true absence of the
phenomenon — is what kills transfer here. With more Vicon subjects held fully out of
MixSTE training, the r=0.87 location signal could become a learnable correction. On
*current* data, it is a NO-GO.

## Files
| file | contents |
|---|---|
| `diagnostic.json` | per-joint residual mean/std/bias, per-phase, axis energy, align compare |
| `camera_depth.json` | camera-frame decomposition (depth vs in-plane, by phase, by joint) |
| `transfer_check.json` | train→test and leave-one-subject-out transfer proxies |
| `fig1_depth_vs_inplane_perjoint.png` | per-joint depth vs in-plane residual (test) |
| `fig2_depth_by_phase.png` | depth residual by swing phase, train vs test |
| `fig3_transfer.png` | signed (no transfer, r≈0) vs unsigned profile (transfers, r=0.87) |
| `Scripts/residual_diagnostic.py` · `camera_depth_decomp.py` · `transfer_check.py` · `make_final_figures.py` | re-runnable analysis (read-only; local setup in the `residual_diagnostic.py` header) |
