# Motion Caddie — Week 7a Speaker Notes (the deck as presented)

Keyed to the **19-page deck** in `Copy of Motion Caddie GD Week 7a (1).pdf`.
⭐ = a slide where you added a result image — these are the ones you're presenting.
Each block is written to be *said out loud* in ~30–45s, first-person "we" voice.

---

## Slide 1 — Title

> Hi everyone — we're Motion Caddie. The one-liner: we turn an ordinary
> single-camera phone video of a golf swing into interpretable motion data, and
> then explain in plain English what the golfer's body actually did. I'm here with
> Theo, Austin, and Lawrence. Today we'll show how the pieces now connect
> end-to-end — the motion metrics, the 3D depth validation, and the LLM layer that
> explains a swing.

---

## Slide 2 — Updated Approach

> Quick reminder of how our approach evolved from last week's feedback. Coaching is
> still the long-term goal, but the MVP deliberately focuses on *explaining* motion,
> not *prescribing* fixes. That one decision de-risks everything — we don't have to
> be a certified coach to be useful, we just have to describe accurately what the
> body did. So the 3D replay is a visualization layer, the motion metrics are our
> evidence layer, and the LLM explains the swing in beginner-friendly language.

---

## Slide 3 — MVP Scope

> Here's what's in and out, so the boundaries are honest. In-scope is the full spine
> of the product: single-POV upload, 2D pose, 3D pose, motion-metric calculations,
> swing-phase detection, the Unreal visualization, and an LLM summary. Stretch goals
> are the richer experiences — temporal and reference-swing comparison, advanced
> biomechanics. And explicitly out of scope: a full AI coach, correction plans, shot
> prediction, or good-versus-bad classification. We describe motion; we don't judge
> it. That line is what keeps the MVP shippable.

---

## Slide 4 — Motion Metrics Layer (what we calculate)

> This is our evidence layer. From the 3D pose we compute swing metrics at each
> event, grouped into families: movement like head movement and body sway; joint
> angles like shoulder, hip, and spine lean; how those change from address to
> impact; the separation between shoulder and hip turn; and timing like tempo. The
> example on the right is the simplest one — head movement is just the 3D distance
> the head travels between address and impact, normalized by shoulder width so it's
> comparable across body sizes. One thing I want to stress: our Low / Moderate /
> High bins describe motion *magnitude*, not swing quality — we never call a swing
> good or bad.

---

## ⭐ Slide 5 — Motion Metrics Layer (the scorecard)

> This is what all of that produces for a real swing — here, Charley Hull, driver,
> face-on. Across the top is the detected swing timeline: the eight events, located
> by frame. Below are fifteen biomechanical indicators, each plotted as a dot — the
> golfer — against the green band, which is the middle 50% of tour pros from
> GolfDB. So we're measuring her against actual tour players, not an invented
> "correct" number. Two things to notice. First, most indicators sit right in the
> tour band — that's the "what you did well" story. Second — and this is important —
> look at "lead-arm extension at top": it's grayed out and tagged *low-confidence*.
> That's not a bug, it's the system being honest: a single camera can't reliably see
> that joint's depth, so we refuse to score it. The two coaching notes at the bottom
> are the only things that cleared our confidence bar — head sway and weight shift.

---

## ⭐ Slide 6 — 2D → 3D Golfswing Optimization (the numbers)

> Everything we just measured depends on the depth axis being right — and depth from
> one camera is the hard problem. This is how we validated and fixed it. We took the
> GolfPose lab dataset, which has true Vicon 3D ground truth, and measured our
> depth error in millimeters. Raw MediaPipe depth is off by about 106 millimeters
> per joint on average — basically unusable for rotation. We then trained a learned
> residual model that corrects the depth, and the combined approach brings that down
> to about 33 millimeters — roughly a 3× improvement. And the per-joint pattern is
> the tell: the hips are always our most accurate joint, single-digit millimeters,
> while the wrists and elbows are the worst. Hold that thought — it lines up exactly
> with which metrics we trust on the scorecard.

---

## ⭐ Slide 7 — 2D → 3D Golfswing Optimization (the trajectories)

> Here's the same result frame-by-frame, so it's not just one summary number. Each
> plot is the depth trajectory of a wrist through a full swing. Black is the Vicon
> ground truth. Red is raw MediaPipe — you can see it's noisy and biased, drifting
> away from the truth. Blue is our corrected output, and it tracks the black ground
> truth closely through the whole motion. This is the visual proof that the depth
> correction holds across the swing, not just on average — which is what makes the
> downstream rotation metrics believable.

---

## Slide 8 — LLM Interpretation Layer (the concept)

> Now the layer that makes this usable for a beginner. The idea is a simple, strict
> translation: each raw metric on the left becomes a plain-English observation on
> the right. "Shoulder turn = high" becomes "your shoulders showed a clear turn
> during the backswing." The bottom line is the guardrail that matters — the LLM
> explains *measured* movement; it does not diagnose the swing or recommend
> corrections.

---

## ⭐ Slide 9 — LLM Interpretation Layer (real output)

> And here it is end-to-end on a real swing: the scorecard on the left feeds the
> explanation on the right. Two guardrails make it trustworthy. First, it only
> narrates numbers we actually measured — it never re-analyzes the video, so it
> can't hallucinate a swing flaw. Second, it leads with what's tour-like and
> describes the differences neutrally, never as instructions to fix. And notice it
> silently drops the low-confidence arm metric from the previous slide — it's simply
> not in the explanation, because we don't trust that number. That's the
> transparency that separates us from a closed coaching app.

---

## ⭐ Slide 10 — LLM Evals

> This is the slide I most want to land. The LLM is the *only* component in our
> system that can make something up — everything before it is deterministic and
> benchmarked. So we built an eval that scores every explanation against the exact
> data it was written from, across 100 swings. The hard safety checks pass
> completely: zero low-confidence leakage, zero prescriptive advice. Coverage,
> readability, and consistency are all strong. But here's the part I'm proudest of —
> the eval *caught a real bug*. The model was lumping several metrics into a family
> and calling the whole family "in range" even when one piece was out — telling the
> golfer something flattering that wasn't true. We saw it because grounding scored
> 0.32, traced it, fixed the prompt, and grounding doubled to 0.64. That's the whole
> point of an eval: it caught a faithfulness bug before any user saw it. And we
> report 0.64 honestly rather than tune the wording until our own grader is happy —
> that would just be overfitting to the judge.

*(If asked why consistency is n=10: it's the one check that triples Codex calls, so
we ran it as a stability spot-check; the safety checks run on all 100. We can scale
it to n=25+ easily.)*

---

## Slide 11 — Validation & Next Steps

> To close the main thread: our near-term work is all about *proving the pipeline is
> reliable* before we add any coaching. Validate pose quality and smoothing, finalize
> the motion bins through EDA, sanity-check the 3D by projecting it back onto the 2D
> video, keep testing the LLM summaries against the guardrails, and confirm the demo
> runs end-to-end. The bottom line: prove it works reliably first, *then* move toward
> coaching recommendations.

---

## Slide 12 — Thank you / Q&A

> That's Motion Caddie. In one line: we diagnosed the real bottleneck, validated our
> depth against motion-capture ground truth, and built an LLM layer that explains a
> swing in plain English — with an eval that keeps it honest. Happy to take
> questions. *(The slides that follow are our backup/appendix — data, EDA, the full
> model leaderboard, and the plan — ready if you want to dig in.)*

---

# Appendix / backup slides (shown if asked)

## Slide 13 — Data + EDA: GolfDB

> Our primary dataset is GolfDB — 1,400 swing clips from 580 YouTube broadcasts
> across 206 professional players. Every clip is annotated with the 8 swing events
> from address to finish, which is exactly what we train and evaluate event
> detection on. It spans three camera views and a mix of real-time and slow-motion
> footage, and it's driver-heavy at about 68%. Events are stored as absolute frame
> numbers, so we get precise temporal localization.

## Slide 14 — Data + EDA: Attempt Mapping 2D → 3D

> This is the raw 2D→3D mapping work — overlaying the predicted skeleton on the
> video and pulling out per-joint trajectories. It's the exploratory step that led
> us to the validated optimization you saw earlier.

## Slide 15 — Data + EDA: GolfPose dataset

> The dataset behind our 3D validation comes from the GolfPose paper out of NYCU,
> ICPR 2024. Six student golfers hitting 7-irons, about 17,700 frames, captured with
> a 9-camera Vicon motion-capture rig — the same class of tech used for film and
> game character animation. That's what gives us true millimeter-accurate 3D.

## Slide 16 — Data + EDA: GolfPose ground truth

> Why it matters: we get both the 2D a normal camera would see and the real 3D from
> the Vicon rig, simultaneously. That solved the gap we had — no ground truth to
> check our 3D against. Now we feed the 2D into our lift model, predict 3D, and
> compare to Vicon in millimeters. The correspondence check here is the sanity test.

## Slide 17 — Data + EDA: Z-depth trajectory of key joints

> This is the depth-axis story across the whole swing for the key joints. Our current
> GolfDB pipeline estimates depth; the GolfSwing dataset lets us validate that
> estimate against ground truth and optimize. It's also where we learn which joints
> have trustworthy depth — and that map feeds directly into the coaching layer's
> confidence gating.

## ⭐ Slide 18 — Model Selection (the leaderboard)

> If anyone wants the full benchmark: we tested 25 pose pipelines on all 1,400 clips.
> The surprise was that three "better" pose models — MediaPipe Heavy, the
> golf-specific GolfPose, and Meta's billion-parameter Sapiens — all produced cleaner
> pose but *worse* swing-event accuracy. That told us the bottleneck was the event
> detector, not the pose. So we trained a 1D-CNN over the 3D landmarks — that's the
> top bar at 0.865, which beats the published SwingNet at 0.76. And it holds on 49
> golfers it never saw in training, at 0.864 — so it's not memorizing players.

## Slide 19 — Plan, Risks & Next Steps

> On the team: Theo on data and the 3D/Unreal layer, Austin on EDA and model
> evaluation, I'm on model development and evaluation, and Lawrence on project
> management and the app. Next steps: more evals for the LLM coaching, fold the
> GolfSwing data deeper into the z-depth optimization, explore a physics constraint
> on the 3D render, and stand up the first end-to-end pipeline on AWS. The honest
> risks: over-smoothing could remove real motion; GolfDB has no shot-outcome labels;
> variable frame rates affect timing; and fast motion or blocked joints degrade pose
> quality.

---

## Q&A backup facts
- **Event detector:** heuristic 0.170 → trained 1D-CNN **0.865** PCE@5; SwingNet ≈ 0.76; player-disjoint (49 unseen golfers) **0.864**.
- **Depth validation:** raw MediaPipe ≈ **106 mm** Z-MPJPE → corrected ≈ **33 mm** vs Vicon (~3×). Hips most accurate, wrists/elbows worst.
- **Tempo sanity check:** corpus median **3.4 : 1**, matches the tour 3:1 norm.
- **LLM eval (n=100):** leakage 0, prescriptive 0, coverage 93%, readability 75, consistency 86%, grounding 0.32 → **0.64** after the bug-fix.
- **LLM cost honesty:** uses Codex/GPT-5 today — cloud, ~$0.10/call. Production would use a cheaper/local model.
