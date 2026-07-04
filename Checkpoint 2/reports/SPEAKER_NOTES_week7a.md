# Motion Caddie — Week 7a Speaker Notes (all slides)

Copy-paste ready. One block per slide — written to be *said out loud* in ~30–45s.
First-person "we" voice. Numbers match what we actually measured.

---

## Slide 1 — Title

> Hi everyone, we're Motion Caddie. The one-line version: we turn an ordinary
> single-camera phone video of a golf swing into interpretable motion data — and
> then explain, in plain English, what the golfer's body actually did. I'm here
> with Theo, Austin, and Lawrence, and we'll walk through where the project moved
> this week, the data work behind it, and a result we're genuinely excited about.

---

## Slide 2 — Updated Approach

> Quick framing on how our approach evolved, based on last week's feedback.
> Coaching is still the long-term goal — but we deliberately narrowed the MVP to
> *explaining* motion rather than *prescribing* fixes. That one decision de-risks
> the whole project: we don't have to be a certified golf coach to be useful, we
> just have to describe accurately what the body did. So the 3D replay becomes a
> visualization layer, the motion metrics become our evidence layer, and the LLM
> sits on top to explain the swing in beginner-friendly language. The flow is:
> single video → 2D pose → 3D pose → Unreal visualization → LLM interpretation.

---

## Slide 3 — MVP Scope

> Here's what's in and out, so we're honest about the boundaries. In-scope is the
> full spine of the product: upload a single-POV video, detect 2D pose, lift it to
> 3D, compute motion metrics, detect the swing phases, render it in Unreal, and
> generate an LLM summary. Stretch goals are the richer experiences — temporal and
> reference-swing comparison, advanced biomechanics. And we're explicit about
> what's out of scope: we are *not* building a full AI coach, correction plans,
> shot prediction, or a good-versus-bad classifier. We describe motion; we don't
> judge it. That boundary is what keeps the MVP shippable.

---

## Slide 4 — Motion Metrics Layer

> This is our evidence layer. From the 3D pose we compute a set of swing metrics at
> each event — grouped into families: movement like head movement and body sway,
> joint angles like shoulder, hip, and spine lean, how those change from address to
> impact, the separation between shoulder and hip turn, and timing like tempo. The
> example on the right is the simplest one — head movement is just the 3D distance
> the head marker travels between address and impact, normalized by shoulder width
> so it's comparable across body sizes. And critically, every metric carries a
> confidence marker — the 3D depth from a single camera is reliable for some joints
> and not others, so we tag the unreliable ones. One detail worth stressing: our
> Low/Moderate/High bins describe motion *magnitude*, not swing quality — we're not
> calling anything good or bad.

---

## Slide 5 — LLM Interpretation Layer

> This is the layer that makes the data usable for a beginner. We hand the LLM the
> structured metrics and their confidence, and it writes a short, plain-English
> explanation of what the body did. Two guardrails make it trustworthy: it only
> narrates numbers we actually measured — it never re-analyzes the video, so it
> can't invent a swing flaw — and it never tells the golfer how to *fix* anything,
> which keeps us in "explain, don't coach" scope. It also silently drops any
> metric we tagged low-confidence. So the output reads like an encouraging caddie
> describing what happened, not a coach barking corrections.

---

## Slide 5b — Monitoring the LLM Layer  *(optional inserted slide — visual: slide05b_llm_eval_monitoring.png)*

> The LLM is the one component in our whole system that can make something up —
> everything before it is deterministic and benchmarked. So we built an eval that
> scores every explanation against the exact numbers it was given, across 100
> swings. The hard safety checks pass completely: zero low-confidence leakage, zero
> prescriptive advice. Coverage, readability, and consistency are all strong. But
> the part I'm proudest of: the eval *caught a real bug*. The model was grouping
> metrics into families and calling the whole family "tour-like" even when one
> piece was out of range — telling the golfer something flattering that wasn't
> true. We saw it because grounding scored 0.32, traced it, fixed the prompt, and
> grounding doubled to 0.64. That's the entire point of an eval — it caught a
> faithfulness bug before any user saw it. And we report 0.64 honestly rather than
> tune the wording until our own grader is happy, because that would just be
> overfitting to the judge.

---

## Slide 6 — Data + EDA: GolfDB

> Our primary dataset is GolfDB — 1,400 swing clips pulled from 580 YouTube
> broadcasts across hundreds of professional players. Every clip is annotated with
> 8 swing events, from address through to finish, which is exactly what we need to
> train and evaluate event detection. It spans three camera views — down-the-line,
> face-on, and other — and a mix of real-time and slow-motion footage, which is
> great because it stress-tests our timing logic. Worth noting it's driver-heavy,
> about 68%, and the events are stored as absolute frame numbers, so we get precise
> temporal localization within each clip.

---

## Slide 7 — Data + EDA: Attempting 2D → 3D

> The hard technical problem underneath all of this is depth. A phone gives us a
> flat 2D video, but to measure things like how far the shoulders rotate *toward or
> away* from the camera, we need 3D. So this section is about how we go from 2D
> joints to a believable 3D skeleton — and, just as importantly, how we know the 3D
> is right.

---

## Slide 8 — Data + EDA: GolfPose dataset

> To validate our 3D, we brought in a second dataset from the GolfPose paper out of
> NYCU, published at ICPR 2024. It was captured in a controlled lab: six student
> golfers hitting 7-irons, about 17,700 frames total. The key thing is *how* it was
> captured — a Vicon motion-capture system with 9 synchronized cameras. That's the
> same class of technology Hollywood and AAA game studios use for character
> animation, and it gives true millimeter-accurate 3D.

---

## Slide 9 — Data + EDA: GolfPose ground truth

> Why this matters: from this dataset we get both the 2D pixel coordinates a normal
> camera would see *and* the real-world 3D coordinates measured by those 9 cameras
> simultaneously. That directly solves the biggest gap we had last week — we had no
> ground truth to check our 3D reconstruction against. Now we can feed the 2D
> coordinates into our lifting model, predict 3D, and compare against the Vicon
> ground truth in real millimeters. So we can finally *measure* how good our depth
> estimate is instead of eyeballing it.

---

## Slide 10 — Data + EDA: Z-depth trajectory

> This slide makes the validation loop concrete. Our current pipeline takes a
> GolfDB video, runs MediaPipe, estimates depth, and outputs 3D — but that depth is
> a guess. The new loop with GolfSwing lets us take its 2D, run our lift model, and
> compare the predicted 3D against the true 3D in millimeters, then optimize. The
> trajectory plot is the payoff: it shows which joints have trustworthy depth and
> which don't — hips are reliable, the wrists and the finish position are the
> noisiest. And that reliability finding isn't just analysis — it's wired directly
> into the coaching layer's confidence gating, so the joints we can't trust never
> drive an explanation.

---

## Slide 11 — Model Selection  *(visual: slide11_event_detector_results.png + leaderboard)*

> We didn't pick a model on vibes — we benchmarked 24 pose pipelines on all 1,400
> clips across 9 metrics. The surprise result drove everything: three different
> "better" pose models — MediaPipe Heavy, the golf-specific GolfPose, and Meta's
> billion-parameter Sapiens — all produced cleaner pose but *worse* swing-event
> accuracy. That told us the bottleneck wasn't the pose at all; it was the simple
> rule we'd been using to detect the 8 swing events. So we trained a small 1D-CNN
> over the 3D landmark trajectories — and it hits 0.865 on the standard benchmark,
> which actually beats the published SwingNet paper at 0.76. We win because we feed
> it clean 3D landmarks instead of raw low-res video. And it holds up under the
> strict test: on 49 golfers the model never saw in training, it scored 0.864 —
> basically identical, so it's not memorizing players. That's the result that turns
> this from research into a deployable feature.

---

## Slide 12 — Technical Pipeline  *(visual: slide12_technical_pipeline.png)*

> Here's the full pipeline, and every box is the measured-best component from that
> benchmark. A single video goes through MediaPipe Lite for 2D pose, MotionBERT-Full
> lifts it to 3D, we smooth it with a One-Euro filter and a bone-length lock that
> cuts jitter by about half, the trained detector finds the 8 swing phases, we
> compute the metrics, and the LLM explains them. One command produces three
> outputs: the coaching scorecard for the user, the Unreal handoff files for the 3D
> replay, and the plain-English summary. The important part for today — this isn't
> a diagram of what we'll build. It runs end-to-end right now.

---

## Slide 13 — Plan, Risks & Next Steps

> Quick status and honesty. On next steps: bring the LLM into the coaching role over
> time, fold the GolfSwing dataset into our z-depth optimization, explore a physics
> constraint on the 3D render, and stand the first end-to-end pipeline up on AWS.
> On risks, the real ones: everything so far is validated on GolfDB, which is
> pre-cropped broadcast footage — real phone uploads are still untested, and that's
> our biggest gap. Over-smoothing could remove genuine motion, so we tuned the
> filter to preserve the impact snap. Variable frame rates affect timing and
> replay. And fast motion or blocked joints degrade pose quality. On the team:
> Theo's on data and the 3D/Unreal layer, Austin on EDA and model evaluation,
> I'm on model development and evaluation, and Lawrence on project management and
> the app build.

---

## Slide 14 — Thank you / Q&A

> That's Motion Caddie. To sum it up in one line: we didn't just pick a model — we
> diagnosed the real bottleneck, trained a fix that beats the published baseline
> and generalizes to new golfers, and shipped it into a working demo that explains
> a swing in plain English. Happy to take questions.

---

### Q&A backup facts (don't put on a slide — keep in your back pocket)
- **Event detector:** heuristic 0.170 → trained 1D-CNN **0.865** PCE@5; SwingNet paper ≈ 0.76; player-disjoint (49 unseen golfers) **0.864**.
- **Tempo sanity check:** across all 1,400 pros our median tempo came out **3.4 : 1**, matching the known tour 3:1 — independent evidence the event timing is right.
- **LLM eval (n=100):** leakage 0, prescriptive 0, coverage 93%, readability 75 Flesch, consistency 86%, grounding 0.32 → **0.64** after the bug-fix.
- **LLM cost honesty:** the summary currently uses Codex/GPT-5 — cloud, ~$0.10/call, needs internet. Production would swap to a cheaper/local model.
- **Smoothing:** One-Euro + bone-length lock cuts jitter ~51% while preserving the impact snap.
- **Why monocular depth is noisy but usable:** we never report raw absolute angles for unreliable joints — we interpret relative to the pro band and gate low-confidence metrics.
