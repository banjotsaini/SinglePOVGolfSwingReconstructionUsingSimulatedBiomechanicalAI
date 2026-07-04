# Week 7a Deck — Slide Content & Speaker Notes

Paste-ready. For each slide: **BULLETS** (slide body), **NOTES** (say out loud),
**VISUAL** (file from `Checkpoint 2/slides/`).

---

## Slide 4 — Motion Metrics Layer

**VISUAL:** `slide04_motion_metrics_scorecard.png` (or the hero `slide04b_hero_scorecard_with_llm.png`)

**BULLETS**
- 15 biomechanical indicators computed from the 3D pose at each swing event
- Families: rotation (shoulder/hip turn, X-factor), posture (spine tilt), movement (head sway/lift, weight shift), timing (tempo)
- Each value scored against the **1,400-pro GolfDB distribution** — golfer vs tour, not vs an invented "correct" number
- **Confidence layer**: every metric carries a reliability tier; low-confidence metrics are flagged, never coached on
- Validation: median tempo recovered as **3.4 : 1** — matches the known "Tour tempo" 3:1

**NOTES**
> This is the evidence layer. From the 3D pose we compute 15 swing metrics at each event — rotations, posture, head movement, tempo. The 3D is what lets us measure rotation at all; a flat 2D video can't see a shoulder turning toward or away from the camera. Every number is compared to the distribution across all 1,400 GolfDB pros, so the golfer is measured against tour players rather than an arbitrary target. And critically — each metric has a confidence score. Some joints are reliably located from a single camera, like the hips; others, like the wrists, are not. We tag the unreliable ones and never build feedback on them. One validation we love: across all 1,400 pros our median tempo came out at 3.4-to-1, which is the well-known tour tempo ratio — independent evidence our event timing is right.

---

## Slide 5 — LLM Interpretation Layer

**VISUAL:** `slide05_metrics_to_llm.png` + the explanation text in `slide05_llm_summary_text.txt`

**BULLETS**
- The LLM turns the structured metrics into a **plain-English swing explanation**
- It only *narrates measured numbers* — it never re-analyzes the video or invents observations
- Leads with what's tour-like, then describes differences **neutrally** (no fix-it instructions — MVP scope)
- Skips any low-confidence metric automatically
- Example output → (paste the text from `slide05_llm_summary_text.txt`)

**NOTES**
> This is the layer that makes the data usable for a beginner. We hand the LLM the structured metrics and their confidence, and it writes a short, encouraging explanation in plain language. Two guardrails matter: it only explains numbers we actually measured — it never re-analyzes the video, so it can't hallucinate a swing flaw — and it never tells the golfer how to fix anything, which keeps us in "explain, don't coach" MVP scope. Here's a real example from Charley Hull's swing — notice it leads with what she did well, describes the weight-shift difference neutrally, and silently drops the arm metric because that one was low-confidence. That transparency is our differentiator versus a closed coaching app.

*(Honesty note for Q&A: this uses Codex/GPT-5 — cloud, ~$0.10/call, needs internet. Production would swap to a local or cheaper model.)*

---

## Slide 5b — Monitoring the LLM Layer (eval)

**VISUAL:** `slide05b_llm_eval_monitoring.png`

**BULLETS**
- The LLM is the **only component that can hallucinate** — so we score every explanation against the data it was written from (n = 100 GolfDB clips)
- **Hard guardrails pass 100%**: 0 low-confidence leakage, 0 prescriptive advice (stays in "explain, don't coach" scope)
- Coverage **93%**, readability **75 Flesch** (plain English), consistency **86%** (same swing → same story)
- **The eval caught a real bug**: the LLM was calling a metric "in range" when a sub-metric was out → fixed it → grounding **0.32 → 0.64**
- Deterministic checks on all 100; grounding via LLM-judge on an audited subset (n reported, not hidden)

**NOTES**
> The one place this system could make something up is the LLM that writes the explanation — everything before it is deterministic and benchmarked. So we built an eval that scores every explanation against the exact numbers it was given. The hard safety checks pass completely: it never narrates a metric we tagged low-confidence, and it never slips into giving fix-it advice. The part I'm most proud of: the eval actually caught a real bug. The model was grouping metrics into families and calling the whole family "tour-like" even when one piece was out of range — telling the golfer something flattering that wasn't true. We saw it because grounding scored 0.32, traced it, fixed the prompt, and grounding doubled. That's the whole point of an eval — it's not decoration, it caught a faithfulness bug before a user ever saw it. We report 0.64 honestly rather than tune the wording until the judge is happy, because that would just be overfitting to our own grader.

*(Honesty note for Q&A: grounding is judged by an LLM — a model auditing a model — so we report the audited n=25 and keep every flagged sentence for human spot-check. The residual gap to 1.0 is minor phrasing, not fabricated facts.)*

---

## Slide 11 — Model Selection

**VISUAL:** `slide11_event_detector_results.png` (hero) + `slide11b_event_timeline_accuracy.png` (supporting) + `slide11_model_selection_leaderboard.png` (full board)

**BULLETS**
- Benchmarked **24 pose pipelines** on all 1,400 GolfDB clips, 9 metrics
- Found the bottleneck was the **swing-event detector**, not the pose model (3 cleaner-pose models scored *worse* on events)
- Trained a 1D-CNN event detector on the 3D landmarks → **PCE@5 = 0.865**, beating the published GolfDB SwingNet (~0.76)
- **Generalizes to unseen golfers**: player-disjoint test (49 new golfers) = **0.864** — no overlap inflation
- Production pose stack: **MediaPipe Lite → MotionBERT-Full** (best measured, 6,900 FPS)

**NOTES**
> We didn't pick a model on vibes — we benchmarked 24 pipelines on the full dataset. The surprise: three different "better" pose models — MediaPipe Heavy, the golf-specific GolfPose, and Meta's billion-parameter Sapiens — all produced cleaner pose but *worse* swing-event accuracy. That told us the bottleneck wasn't the pose at all; it was the simple rule we used to detect the 8 swing events. So we trained a small model — a 1D-CNN over the 3D landmark trajectories — and it hits 0.865 on the standard benchmark, which actually beats the published SwingNet paper at 0.76. We win because we feed it clean 3D landmarks instead of raw low-res video. And the number holds up under the strict test: on 49 golfers the model never saw during training, it scored 0.864 — basically identical, so it's not memorizing players. That's the result that turns this from research into a deployable feature.

---

## Slide 12 — Technical Pipeline

**VISUAL:** `slide12_technical_pipeline.png`

**BULLETS**
- One command on a phone video → swing explanation + UE5 files
- Stages: 2D pose (MediaPipe Lite) → 3D lift (MotionBERT-Full) → smoothing → trained event detector → motion metrics → LLM explanation
- Smoothing (One-Euro + bone-lock) cuts jitter ~51% before downstream use
- Outputs per swing: **coaching scorecard**, **UE5 handoff** (CSV/BVH/3D viewer), **LLM summary**
- Working **end-to-end today** — not a future plan

**NOTES**
> Here's the full pipeline, and every box is the measured-best component from the benchmark. A single-POV video goes through MediaPipe Lite for 2D pose, MotionBERT-Full lifts it to 3D, we smooth it, the trained detector finds the 8 swing phases, we compute the metrics, and the LLM explains them. One command — `demo.py` on a video — produces three things: the coaching scorecard for the user, the Unreal handoff files for the 3D replay, and the plain-English summary. The important part for today: this isn't a diagram of what we'll build. It runs end-to-end right now.

---

## Slide 13 — Plan, Risks & Next Steps (UPDATE)

**NEXT STEPS** (replace the old bullets — they're done)
- ✅ Done: trained event detector (0.865, beats SwingNet), full coaching layer, LLM explanation, end-to-end demo
- ➔ Deploy the pipeline to AWS
- ➔ Collect ~20 real phone-video swings to validate beyond GolfDB
- ➔ Scale the LLM-labeled training set (YouTube pipeline) for further gains
- ➔ Wire the 3D output into Theo's Unreal Engine render layer

**KEY RISKS** (refreshed)
- Everything validated on GolfDB (pre-cropped broadcast clips) — **real phone uploads untested**
- LLM explanation currently uses a cloud model (cost/latency) — production needs a cheaper/local option
- Single-camera depth is estimated, not measured — per-clip rotation metrics are noisy (mitigated: reliability gating + interpret relative to pro band)
- Over-smoothing could remove real motion (mitigated: One-Euro preserves impact snap)

**NOTES**
> Quick status correction from last week — "build the first end-to-end pipeline" is now done; it runs locally today. So next steps move up the stack: deploy to AWS, and the big validation gap is real phone video — everything so far is GolfDB's pre-cropped broadcast clips, so we want ~20 real uploads to confirm it generalizes. On risks, the honest ones: the LLM layer uses a cloud model right now, which is a cost and latency question for production; and single-camera depth is always an estimate, which is exactly why we built the confidence gating rather than reporting raw absolute angles.

---

## Slides 8-10 (GolfPose / z-depth) — one-line addition

Add to the z-depth slide:
> Austin's pose-consistency analysis (which joints have reliable depth) now **directly powers the coaching layer's confidence gating** — hips trusted, wrists flagged. The z-depth track and the coaching track are connected.

---

## Closing line (works on any summary slide)
> "We didn't just pick a model — we diagnosed the bottleneck, trained a fix that beats the published baseline and generalizes to new golfers, and shipped it into a working demo that explains a swing in plain English."
