# Ball Tracking — Research & Implementation Plan

Goal: a ball-tracking stage that runs in **parallel with the 2D skeleton stage**, followed by a
**3D uplift of the ball trajectory**, wired end-to-end into the deployed pipeline and validated on
`C:\Users\Banjo\Downloads\IMG_3434.MOV` and `C:\Users\Banjo\Downloads\IMG_8107.MOV`.

Loop state: **research iteration 7 of ~8 complete** (then 7 implement→test→deploy iterations,
then final phase: git commit, deploy demo+dev pages, e2e retest, LLM ball-data tool).

## Iteration log

### R1 (2026-07-17) — pipeline map + model survey
**Pipeline integration points (from codebase exploration):**
- Stage sequencer: `deploy/processing/processing_handler.py:_pipeline` (line 111) shells out to
  `Scripts/*.py` subprocesses; local mirror is `Scripts/demo.py` (`python Scripts/demo.py video.mov`,
  defaults `mediapipe_lite` + `golfpose3d`).
- 2D pose stage: `Scripts/adapters/mediapipe_adapter.py` via `BaseAdapter` ABC
  (`Scripts/eval_utils.py:216`) — `predict(video) -> DataFrame[frame,kp_idx,kp_name,x,y,conf]` in
  pixel coords, free parquet caching via `predict_and_cache()`. Registry:
  `Scripts/adapters/__init__.py:69`. **A ball detector fits this contract as a 1-keypoint adapter.**
- Frames are re-decoded per stage (`iter_frames`, `eval_utils.py:278`) — parallel ball stage can
  simply re-decode; no shared-decode refactor needed for v1.
- 3D frame: lifter output is hip-relative, unit-normalized, `h36m-camera` axes; **no camera
  intrinsics exist; absolute meters NOT recoverable**. Floor/leveling: `smoothing.level_and_ground`
  (`Scripts/smoothing.py:208`) — need to expose its rotation R to place ball in same frame.
  Body-scale (from `coaching_indicators.py`) is the only pixel→physical anchor.
- Sim ball flight to feed/replace: `Scripts/ball_flight.py` (`_ui_trajectory` at line 350),
  chat tool `estimate_ball_flight` (`Scripts/coaching_chat.py:236-294`), UI arc
  `deploy/web/chat.js:495 flightArc`.
- Artifacts: emit `<stem>_ball_3d.json`; cloud list `ARTIFACTS` at `processing_handler.py:50`.
- No existing ball/vision-detection code anywhere (ultralytics 8.4.53 already in local .venv;
  cloud image is CPU torch 2.4.1, numpy<2).

**Model candidates (initial survey):**
1. **WASB-SBDT** (BMVC2023, nttcom) — multi-sport ball detect+track strong baseline, pretrained
   weights in MODEL_ZOO (tennis/badminton/soccer etc., no golf) — https://github.com/nttcom/WASB-SBDT
2. **TrackNet family** (V1 tennis; V3 shuttlecock qaz812345/TrackNetV3; V4 motion-attention,
   arXiv 2409.14543) — heatmap nets on consecutive frames, built for tiny fast blurred balls.
3. **YOLO-based golf ball detection** — lit: Faster R-CNN vs YOLOv3/tiny (0.78 mAP, 357 fps,
   researchgate 347630649). Ultralytics already installed; Roboflow-style golf-ball datasets +
   nano model fine-tune is the low-lift trainable path.
4. **Classical motion baseline** — stabilize (phaseCorrelate) + frame-diff + physics-gated blob
   linking. Proven manually on IMG_3434.MOV (ball tracked frames 290–314, launch direction
   measurable). Zero deps, CPU-cheap — strong v1 / fallback candidate consistent with the
   pipeline's dependency-light ethos.

**3D uplift thought (to develop in R3+):** monocular 3D of a golf ball is ill-posed (ball too
small for size cue). Practical path: fit a physics trajectory (projectile + drag/Magnus from
`ball_flight.py`) whose camera projection matches the observed 2D track — i.e., 3D uplift =
constrained physics fit, anchored by body-scale + impact position, not a learned lifter.

### R2 (2026-07-17) — video ground truth + ball visibility ✅
- Ran `Scripts/demo.py` on both videos; bundles in `Data/demo/IMG_3434/` and `Data/demo/IMG_8107/`.
- **BUG found (fix in I-phase):** IMG_3434's 3D cache landed in
  `Data/eval_runs/golfpose3d_from_mediapipe_lite_lrfix/` (L/R repair fired) but `demo.py`'s
  scorecard-step cache lookup only checks the non-lrfix dir → scorecard skipped for IMG_3434.
  Cloud handler (`processing_handler.py:124-130`) prefers lrfix correctly; demo.py doesn't.
- **Ball visibility (both 720x1280@30fps, Topgolf, down-the-line):**
  - IMG_3434: impact f287; ball visible f290–314 (~0.8s, 24 frames), 3–6 px dot, drifts right,
    exits right frame edge near apex.
  - IMG_8107: impact f130 (event detector agrees — events: address 91 … impact 130); ball
    visible ~f131–147 (~0.5s), 3–5 px, climbs up-right, lost in sky/pole clutter after f147.
- **Detector implications:**
  - Ball is a 3–6 px backlit dot → generic YOLO at 640-px input sees a ~2 px target: hopeless
    without tiling. Need native-res inference (tiles/crops) or heatmap multi-frame nets
    (TrackNet/WASB) designed for tiny objects.
  - Stabilized frame-diff + blob linking works (verified manually on both clips) but needs
    physics-gated track initiation to reject clutter (flags, other bays' balls, birds).
  - **Synergy with existing stages:** event detector supplies the impact frame; pose supplies
    wrist/clubhead region at impact → constrain track initiation to that time+place. This is
    the key advantage over generic trackers.
  - Only ~15–25 frames of flight are observable → enough for launch direction + initial pixel
    speed, NOT for carry. Carry requires physics extrapolation (→ R3 physics-fit uplift).
### R3 (2026-07-17) — 3D uplift design (physics-fit) ✅
**Core idea: 3D uplift = constrained physics fit, not a learned lifter.** `simulate_flight()`
(`Scripts/ball_flight.py:157`) is already a full forward model — params
(ball_speed_mph, launch_angle_deg, azimuth_deg, backspin, sidespin) → 3D track
`[downrange_yd, height_yd, side_yd]` sampled over time. Fit: project the simulated 3D track
through a camera model and minimize pixel reprojection error against the observed 2D ball
track over the ~15–25 visible frames (t_i = (frame − impact_frame)/fps).

**Camera model (v1):** static pinhole (stabilization removes shake), principal point = image
center, focal length f_px from typical phone FOV prior (portrait 720x1280, main lens ≈ 26 mm
equiv) — treat f as fixed with a sensitivity band, or a 4th fit param with tight prior.
Camera yaw vs target line is unobservable → report azimuth RELATIVE TO CAMERA AXIS and label
it so (down-the-line assumption). No intrinsics exist anywhere in the repo (R1) — this is new.

**Identifiability:** early flight (~0.5 s) has little spin curvature → spin params stay at
club-default priors (envelope-gated, same philosophy as Phys-NN); fit only
(ball_speed, launch_angle, azimuth). ~20 obs / 3 params + physics priors = tractable.
Optimizer: tiny hand-rolled Nelder-Mead (repo is dependency-light; ball_flight is stdlib;
avoid scipy unless R6 shows it's already in the cloud image).

**Scale anchor:** ball diameter (42.7 mm, 3–6 px) is too small to use. Anchor = golfer:
pixel torso length at address ÷ assumed torso meters (~0.50 m; user height later) → depth of
launch point; launch origin = ball/tee position at impact (from pose feet/club region +
event detector impact frame). Carry scales linearly with this anchor → publish an
uncertainty band, not a point estimate.

**Output artifact `<stem>_ball_3d.json` (draft schema):**
- `track_2d`: [[frame, x_px, y_px, conf], …] (raw measured)
- `launch_measured`: {elevation_deg, azimuth_deg_rel_camera, n_frames, residual_px, quality}
- `fit`: {ball_speed_mph, launch_angle_deg, azimuth_deg, spin_assumed:{...}, carry_yd,
  side_yd, apex_yd, flight_time_s, engine:"physfit-v1", confidence, carry_range_yd:[lo,hi]}
- `trajectory_world`: simulate_flight trajectory points (feeds/replaces `_ui_trajectory` arc)
- `trajectory_replay`: ball points mapped into the replay frame — world yards → replay units
  via (pose torso units / torso meters), then apply the same leveling rotation + floor shift
  as the skeleton. **Refactor needed:** `level_and_ground` (`Scripts/smoothing.py:208`)
  returns only an info dict — expose R + shift in `info` so the ball step can reuse them.

**Renderable:** add optional `ball` key to `replay_3d.json` (viewers draw polyline+dot;
`grounded/floor_y` already in schema); chat `flightArc` (`deploy/web/chat.js:495`) renders the
measured arc with a "measured" badge vs today's "simulated".

**Fallback:** if no confident track (occlusion, face-on camera, indoor net), stage emits
`quality:"none"` and chat keeps current simulate-only behavior. Validation targets for the two
clips: IMG_3434 → azimuth clearly right-of-camera-axis (matches observed push/fade);
IMG_8107 → up-and-right, plausible Topgolf carry (150–230 restricted-flight).
### R4 (2026-07-17) — detector decision ✅
**Decision: v1 detector = classical stabilized frame-diff + physics-gated track linking.**
Rationale matrix:
| Candidate | Fit | Verdict |
|---|---|---|
| Classical diff+linking | Proven manually on BOTH clips; zero new deps (cv2 only); CPU-ms cheap; exploits impact-frame + launch-origin priors no generic model gets | **v1** |
| YOLO fine-tune | Roboflow golf datasets are small (100–185 imgs, mostly close-ups — not 3–6 px in-flight dots); ultralytics NOT in cloud image (new heavy dep); 640px input kills tiny targets without tiling | upgrade path |
| WASB / TrackNet | No golf weights; torch port + fine-tune data needed; strongest long-term for cluttered/low-contrast cases | upgrade path (v2 if v1 recall disappoints) |

Datasets noted for the upgrade path: Roboflow Universe "golf ball detection" (appleroot,
185 imgs), "golf ball tracker" (StandardData, 100 imgs), search `class:golfball`.

**v1 algorithm (BallDetector2D):**
1. Read impact_frame from scorecard events (or estimate from motion burst if absent);
   launch origin ≈ lowest wrist/club region at impact from 2D pose (already cached parquet).
2. Stabilize frames impact→+2.5 s vs impact frame via `cv2.phaseCorrelate` (proven here).
3. Consecutive-frame abs-diff, threshold, connected components ≤ ~30 px wide.
4. Track initiation: candidate chain must start within a cone from launch origin within
   ~5 frames of impact; growth by nearest-neighbor gated on smooth pixel velocity
   (monotone rise early, bounded accel) — rejects flags/birds/other bays' balls.
5. Output per-frame (x, y, conf) + quality (n_frames, gaps, residual to smooth poly).
**Eval metric (both clips):** manual GT tracks from today's analysis (IMG_3434 f290–314,
IMG_8107 f137–147 verified points) — require ≥80% of GT frames within 4 px, zero false
tracks; runtime < 2 s/clip CPU. Bake-off vs deep models deferred until v1 measured.
### R5 (2026-07-17) — prototype WORKS on both clips ✅
Prototype: scratchpad `proto_ball_track.py` (session 8a3e149f…; port to
`Scripts/adapters/ball_detector.py` in I1). Results vs manual GT:
- IMG_3434: **10/11 GT frames within 8 px**, 20-pt track f290–309 (real ball, club rejected)
- IMG_8107: **5/5 within 8 px**, 8-pt track f137–145; zero false tracks on either clip
- Runtime ~5 s/clip CPU (bar was 2 s — optimize by estimating phaseCorrelate shift on
  downscaled frames in I-phase; not blocking).

**The discriminator stack that made it work (each was added to kill a real observed failure):**
1. Foreground = median-background diff ∧ consecutive-frame diff (kills static clutter AND
   double-image artifacts; consecutive-diff alone produces ghost pairs).
2. Masks: golfer bbox from cached 2D pose (+70 px), below-horizon (0.62H), roof (0.06H),
   stabilization border bands (12 px — 30 px ate the ball's exit path).
3. Initiation: impact+1..impact+9 only (neighbor-bay tracks start late), first step upward,
   ≥4 px/frame, decelerating over first 8 pts.
4. Straightness: line-fit RMS ≤5 px over first 8 pts (kills club arcs, body parts).
5. Launch cone: origin→start direction within 40° of initial track velocity.
6. Anti-causality: back-extrapolate launch velocity into impact−6..impact−1; ≥2 moving-blob
   matches ⇒ neighbor bay's ball already in flight ⇒ reject. Exempt positions within 220 px
   of origin (that's the pre-impact club, not a neighbor ball).
7. Ballistic truncation: cut track at 3 consecutive near-zero steps (stall) or 3 consecutive
   dominant-axis sign reversals — the clubhead launches ball-like from the origin at impact+1
   but always stalls/reverses at the top of follow-through; a receding ball never does.
8. Score: 10·rms − 5·min(len,18) − 2·speed₀ − 40·cone_dot (min wins).
Launch origin heuristic: (mean pose x, max pose y) at impact — adequate; refine with
clubhead/tee detection later if needed.
### R6 (2026-07-17) — physics-fit uplift prototype WORKS (constrained form) ✅
Prototype: scratchpad `proto_phys_fit.py` (port in I-phase). Camera: pinhole, f_px≈910
(iPhone 26 mm-equiv prior, 720x1280), pitch from horizon line, depth anchored by golfer
standing height (95th-pct pixel span; impact-frame height is crouched and inflates depth).

**Key finding — speed is NOT identifiable monocularly:** the free 3-param fit
(speed, launch, azimuth) trades speed against depth/f: ±15% focal prior → 89–199 mph.
Fundamental scale ambiguity in a ~0.5 s 2D track; this is why launch monitors use radar.

**Production formulation — constrained fit:** fix ball speed from the existing envelope
(club default × hand-speed nudge, exactly what `estimate_for_club(speed_scale=…)` does),
fit ONLY (launch_angle, azimuth). Result is stable across ±15% focal prior:
- IMG_3434: launch 26°±3, azimuth +21..27° (right push — matches video), carry 206–212 yd,
  apex ~53 yd, ~7.6 s, resid 7–12 px
- IMG_8107: launch 10.4°±1 (flat), azimuth +17..21°, carry 196–202 yd, apex ~19 yd, ~5.3 s,
  resid 4–7 px
So: **measured = track + launch angle + azimuth (bands); simulated-but-informed = carry/apex
via envelope speed + measured shape; report carry as a band.** Azimuth is relative to camera
axis (labeled). scipy Nelder-Mead used locally — R7 must check the cloud image for scipy,
else hand-roll (~30 lines).
### R7 (2026-07-17) — speed IS extractable with subpixel + enough points ✅
(user direction: extract speed/launch confidently and estimate distance + other features)
Experiment: subpixel centroid refinement (intensity-weighted, median-diff patch) + bootstrap
CI (n=14, resampled points × ±8% focal jitter) on the FREE 3-param fit:
- IMG_3434, 20 pts: **speed 150–160 mph, launch 24.0–26.5°, azim 22–25°, carry 237–257 yd**
  — tight. Gravity curvature over 20 subpixel points pins the scale. (Chatbot's hand-speed
  sim guessed 241 — inside the band.)
- IMG_8107, 8 pts: speed 119–159 mph — too wide. **Track length is the gate.**
**Production rule:** ≥15 subpixel points → free fit, report speed/launch/azim/carry as
measured WITH CI band; <15 points → constrained fit (envelope speed, measured shape only),
label carry "estimated". Quality tiers: `measured` / `partial` / `simulated`.
**To widen the ≥15-pt regime (I-phase):** extend the chase past current cutoffs with
predictive search + threshold decay (8107's ball is visible past f147); more points also
narrow the CI. **To cut anchor bias:** user profile height (DB/product), iPhone focal-length
metadata from the .MOV container if present (probe in I-phase), mat-size scale reference.
**Other features to emit:** apex band, hang time, azimuth (push/pull direction), smash-factor
cross-check vs hand_speed indicator, sidespin/curvature fit once tracks are long enough
(needs the longer-chase work), descent angle at landing.
### R8 (pending) — consolidate: cloud-image constraints (scipy present? runtime budget,
numpy<2), artifact/DB wiring, chat tool + UI shape, implementation spec + I1–I7 task list.

## Implement→test→deploy loop (after research)
I1–I7 (pending): adapter + registry, uplift step, pipeline.py + processing_handler wiring,
artifacts/UI, end-to-end on both videos locally, deploy image rebuild, cloud validation.

## Final phase (user-added 2026-07-17, after I1–I7)
- Commit work to git.
- Deploy to BOTH the demo site and the dev portal (`/portal.html`) pages.
- Re-test everything end to end after deploy.
- **LLM tool access**: measured ball-tracking/estimation data must be exposed to the coaching
  LLM as a tool — extend/replace `estimate_ball_flight` in `Scripts/coaching_chat.py` (TOOLS
  registry line ~294) so chat can fetch the tracked trajectory + measured launch direction for
  a processed job, with grounding rules distinguishing measured vs simulated values.
- **Confident distance answers (user, 2026-07-17):** when ball_3d quality is `measured`, the
  chat tool returns measured speed/launch/azimuth + carry with its CI band, and the tool's
  display strings/grounding should let the model answer distance questions CONFIDENTLY
  ("your carry was about 245 yards, measured from the video") instead of today's heavy
  "simulated estimate" hedging — reserve the hedge for `partial` (direction measured, speed
  assumed) and `simulated` (no track) tiers. Tier + phrasing guidance must be explicit in the
  tool result so the grounding verifier accepts the confident phrasing.
