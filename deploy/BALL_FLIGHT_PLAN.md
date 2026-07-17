# Ball-Flight Estimation Feature — Implementation Plan

**Source research:** McNally, Lambeth & Brekke, *Combining Physics and Deep Learning Models to
Simulate the Flight of a Golf Ball*, CVPR 2023 CVSports workshop
(https://openaccess.thecvf.com/content/CVPR2023W/CVSports/papers/McNally_Combining_Physics_and_Deep_Learning_Models_To_Simulate_the_Flight_CVPRW_2023_paper.pdf).

**Verdict: FEASIBLE — ready to implement.** Physics core prototyped and validated against
TrackMan tour averages (carry within ~2–5%, apex within ~2 yd, fade/draw curvature realistic,
<1 ms per shot in pure Python). Prototype: scratchpad `proto_ball_flight.py` (this session).

---

## 1. What the paper gives us (and what it doesn't)

- Model: ODE ball-flight simulation (Newton + Magnus lift + drag + spin-decay torque,
  forward-Euler) where a small MLP predicts aerodynamic coefficients CL/CD/CQ per time step
  (CM = pM·S). Inputs: **launch conditions only** — ball speed, launch angle, azimuth,
  backspin, sidespin. Outputs: full 3D trajectory, carry, side, apex, flight time,
  landing velocity/spin. 35 ms/shot CPU.
- **No official code release** (authors at Dunlop Sports). wmcnally's GitHub has none.
  Third-party reimpl `asrospie/golf-flight-model` = minimal physics-only, no weights.
- **Training data (90k TrackMan shots) is proprietary.** The NN variant (Phys-NN+) cannot be
  reproduced as-published. The paper's own baseline (Phys-Q, Ferguson et al. 2022,
  doi 10.5703/1288284317493) uses published polynomial coefficients and is only ~28% worse
  (6.6 vs 4.75 yd mean landing error on their val set) — accurate enough for a coaching demo.
- **This is not monocular depth estimation** — "depth" = shot distance/carry. Feature scoped
  as ball-flight / carry estimation exposed to the coaching LLM as a tool.

Validated Phys-Q coefficients (S = spin ratio = R·|ω|/|v|):
```
CD = 0.1304 + 0.9287·S − 0.8259·S²
CL = 0.0504 + 1.2031·S − 1.1490·S²
CM = 0.01·S
```
⚠ Implementation note: with backspin stored as +x and flight along +y, Magnus lift is
**ω×v**, not the paper's literal v×ω (opposite spin-sign convention). Verified: v×ω makes
the ball dive (82 yd driver); ω×v reproduces tour numbers.

## 2. Launch conditions — where they come from (GAP-2 resolution)

We have no launch monitor. Three-tier sourcing, in priority order:

1. **User-stated** — golfer tells the chat their ball speed / club speed / typical distances;
   tool accepts explicit overrides.
2. **Club-based defaults** — scorecard `meta.club` already exists (e.g. "driver"). Ship a
   lookup table of TrackMan published tour + amateur averages (ball speed, launch angle, spin
   by club and skill tier). Default tier: amateur; LPGA/PGA tier for pro demo clips
   (meta.player is a tour player for GolfDB clips).
3. **(Phase 2) Pose-derived scaling** — hand speed at impact from replay_3d.json (18-joint,
   ~30 fps, meters, impact frame in scorecard events). Too coarse to be a primary source
   (no club/ball visible, 30 fps); use only to scale the default table and mark low-confidence.

## 3. Architecture

```
Scripts/ball_flight.py          NEW — pure-stdlib physics sim (no numpy/torch)
  simulate_flight(ball_speed_mph, launch_deg, azimuth_deg, backspin_rpm,
                  sidespin_rpm, wind=None, dt=0.01) -> dict
  club_launch_defaults(club: str, tier: str) -> dict   # TrackMan averages table
  estimate_for_club(club, tier, overrides: dict) -> dict  # merge + simulate

Scripts/coaching_chat.py        EDIT — one new TOOLS entry + SYSTEM_RULES update
deploy/chat_handler.py          NO CHANGE (imports coaching_chat; Scripts already on path)
Data/coaching qa sets           EDIT — re-label "How far did the ball go?" cases
deploy/web/chat.js              OPTIONAL — trajectory sparkline from tool_log
```

### Tool contract (model-facing)

```python
"estimate_ball_flight": ({
    "description": "SIMULATE the likely ball flight for this swing's club using a physics "
                    "model (McNally et al. 2023 baseline). Returns estimated carry, side, "
                    "apex and flight time. This is an ESTIMATE from typical launch conditions "
                    "for the club — not a measurement of this shot. Use it when the golfer "
                    "asks how far the ball went / would go, or about shot shape. Pass any "
                    "launch numbers the golfer states.",
    "input_schema": {"type": "object", "properties": {
        "ball_speed_mph":  {"type": "number", "description": "override, optional"},
        "launch_angle_deg":{"type": "number"},
        "backspin_rpm":    {"type": "number"},
        "sidespin_rpm":    {"type": "number"},
        "skill_tier": {"type": "string", "enum": ["tour", "amateur"],
                       "description": "default amateur; tour for pro swings"},
    }, "additionalProperties": False},
}, _t_estimate_ball_flight)
```

Tool result shape (all display numbers pre-rounded to ints — the grounding verifier
matches numbers from tool results, so return exactly what the model should quote):
```json
{"estimated": true, "club": "driver", "tier": "tour",
 "assumed_launch": {"ball_speed_mph": 172, "launch_angle_deg": 10.4, "backspin_rpm": 2545},
 "carry_yd": 288, "side_yd": 0, "apex_yd": 33, "flight_time_s": 6.8,
 "note": "Simulated from typical driver launch conditions, not measured from the video."}
```

### SYSTEM_RULES changes (Scripts/coaching_chat.py)

- REMOVE "ball flight/distance, ball direction" from the UNMEASURED refusal list.
- ADD a "SIMULATED ESTIMATES" rule: for how-far / shot-shape questions call
  `estimate_ball_flight`, always say it is a simulation from typical launch conditions
  (not measured), and name the club assumed. Keep clubhead speed etc. in the refusal list.
- Keep FIX/ADVICE refusal unchanged.

### Grounding verifier

No code change expected: `verify_chat_grounding` already collects every number in every
tool result into `grounded_nums`. Return ints (or 1-dp) so the model's quoted numbers match
after the verifier's round(·, 2). Add eval cases to prove it.

## 4. Phases

**Phase 1 — core + tool (demo-ready, ~1 day) — ✅ SHIPPED 2026-07-16**
Live on /chat (Bedrock Haiku 4.5): `estimate_ball_flight` answers distance questions
grounded (279 yd tour driver for clip 0), refusals/metrics unchanged, overrides work
(140 mph → 216 yd). Tests: Scripts/test_ball_flight.py (45), test_coaching_chat.py (59),
coaching_chat_eval.py --self-test (7), deploy/test_chat_handler.py (17).
1. Port prototype → `Scripts/ball_flight.py` (+ unit tests: tour-average table within 6%,
   sidespin sign/curve direction, zero-spin sanity, dt-halving convergence).
2. Add TrackMan defaults table (tour + amateur tiers, all clubs in scorecard vocabulary).
3. Register tool in TOOLS, update SYSTEM_RULES, update tool_specs if club missing → tool
   returns {"estimated": false, "note": "no club recorded"}.
4. Re-label the six "How far did the ball go?" eval cases (qa_set_test/dev) from refusal →
   simulated-estimate; run coaching_chat_eval.py; verify grounding stays green.
5. Redeploy chat Lambda (existing repatch flow: updated coaching_chat.py + new ball_flight.py;
   chat_handler.py unchanged).

**Phase 2 — swing-aware inputs + UI — ✅ SHIPPED 2026-07-16**
- `hand_speed_impact_bs` indicator (peak wrist-midpoint speed into impact, body-scales/s,
  One-Euro-smoothed pose, fps-gated: omitted for slow-motion/unknown-fps clips). Reference
  band rebuilt from the 758 real-time GolfDB clips (median 7.0 [5.9–8.0]); tier "low"
  (chat refuses to state it). Demo scorecards patched (8 real-time clips) + meta.sex added.
- Sim tool scales assumed ball speed by measured-vs-median hand speed, capped ±12%,
  disclosed via `swing_speed_adjustment`; golfer-stated ball speed always wins. LPGA tier
  auto-selected for female pros (meta.sex). Verifier: successful sim legitimizes range
  phrasing (ungrounded_range_claim false-positive fixed); `_ui`-prefixed tool-result keys
  are UI-only (stripped from model view + grounding pool).
- chat.js `flightArc()`: side-view SVG arc from `_ui_trajectory` (24 pts) under sim answers.
  Deployed to S3+CloudFront (browser-verified against live API: 262 yd LPGA-tier arc).
- Uploads: processing_handler probes video fps (cv2) → scorecard_step --fps; image v9 build.

**(original plan below)**
6. Hand-speed-at-impact indicator in scorecard_step.py (wrist joint velocity around impact
   frame from the 3D parquet; mark low-confidence) → requires processing-image rebuild (v9)
   via the documented CodeBuild loop.
7. Scale club defaults by measured hand speed vs tier-typical hand speed; surface the
   assumption in the tool result.
8. chat.js: render a small 2D trajectory arc (side + height vs distance) from a `trajectory`
   field (decimated to ~30 pts) in the tool result.

**Phase 3 — Phys-NN upgrade — ✅ SHIPPED 2026-07-16**
Trained the paper's final config (pM-cQ MLP + Lh loss, dt=0.1 differentiable Euler,
soft-argmax landing) on tim-blackmore/launch-monitor-regression (7,866 clean driver
shots, temporal 90/10 split): **held-out mean landing error 1.84 yd vs 7.11 yd for the
shipped Phys-Q** (the 7.11 independently validates our Phys-Q port — paper reports 6.96).
Shipped as `Data/coaching/ball_flight_nn.json` (weights + 1–99% training envelope) with
a numpy forward pass in ball_flight.py, gated: NN runs only for launch conditions inside
the envelope (ball STATE, club-agnostic), else Phys-Q; missing numpy/weights degrade
gracefully to Phys-Q. Result carries `engine: phys_nn|phys_q`. Trainer:
Scripts/train_phys_nn.py. Live-verified: driver + 7-iron on phys_nn (LPGA 7i sim 140 yd
vs 141 published), tour wedge (9.3k rpm) correctly falls back to phys_q.

**(original plan below)**
9. Train the paper's MLP (2×[256,128] + CQ variant, soft-argmax landing loss + apex loss) on
   public data: tim-blackmore/launch-monitor-regression (~10k TrackMan rows) + Kaggle Garmin
   R50 set. Export weights to JSON; forward pass in pure Python (3 matmuls) keeps the Lambda
   dependency-free. Only ship if val landing error beats Phys-Q; expectation: marginal — the
   paper needed 81k curated shots.

## 5. Risks / accepted limitations

- Estimates, not measurements — mitigated by tool description + system rules + note field;
  the honest framing ("simulated for a typical driver swing") is a feature for the demo.
- Club comes from metadata; if absent the tool declines rather than guesses.
- Phys-Q accuracy (~6.6 yd mean landing error on the paper's val set) is far below the
  uncertainty introduced by assumed launch conditions — the defaults table dominates error;
  don't over-invest in Phys-NN.
- Verifier number-matching: covered by pre-rounding; add regression eval (verifier had prior
  rounding false-positive history, fixed 2026-07-06).

## 6. Exit-condition audit (research loop)

- I1 paper read → GAP-1 no code/weights/data.
- I2 repo hunt → GAP-1 confirmed (no official repo; asrospie reimpl minimal). Plan: implement from equations.
- I3 coefficients found (Ferguson 2022 values) → GAP-1 downgraded to phase-3 stretch.
- I4 codebase: TOOLS registry + chat Lambda imports coaching_chat → GAP-3 (integration) scoped. GAP-2 (no speed metrics) confirmed.
- I5 replay_3d.json (3D pose @30fps, impact frame) + TrackMan averages → GAP-2 resolved (3-tier sourcing).
- I6 scorecard has meta.club; verifier auto-grounds tool numbers → GAP-3 resolved.
- I7 deploy path: Scripts on Lambda path; scorecard built in scorecard_step.py → GAP-5 resolved.
- I8 SYSTEM_RULES currently refuses distance questions; public datasets exist but small → GAP-6 (prompt/evals) identified + resolved in plan; NN = stretch.
- I9 physics prototype validated vs tour averages (2–5%); Magnus sign convention fixed; eval fixtures located.
- I10 final sweep: no new gaps → exit condition 3 (ready to implement).
