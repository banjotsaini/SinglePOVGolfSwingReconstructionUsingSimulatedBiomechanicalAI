# Leaderboard + Demo — Speaker Notes

Current as of the trained event detector + coaching demo. Supersedes the
earlier "0.450 projected" framing — that number is now **measured at 0.865**.

---

## 30-second version

> "We benchmarked 24 pose pipelines, then realized the real bottleneck wasn't the pose model — it was the swing-event detector sitting on top of it. So we trained one: a small 1D-CNN on our 3D landmark trajectories. It hits **0.865 PCE@5 on held-out data — above the published SwingNet baseline of 0.76.** And we wired the whole thing end-to-end: a raw phone video now produces a coaching scorecard in one command."

---

## The leaderboard image — how to read it

The bars are PCE@5 (swing-event accuracy). Two things to point at:

1. **Top bar — the trained event detector (0.865).** This is the headline. Everything below it is a *pose* pipeline scored with our simple heuristic detector; the top bar is what happens when you replace that heuristic with a trained model on the same 3D landmarks.
2. **The pose-pipeline cluster (0.05–0.17).** MediaPipe Lite + MotionBERT-Full leads the *pose* tier at 0.170. The spread is narrow — which was the clue that the detector, not the pose, was capping us.

---

## The story arc (the part that lands)

**1. We tested everything.** 24 pipelines — MediaPipe, MoveNet, YOLO, ViTPose, Meta's Sapiens foundation model, MotionBERT 3D lifts, golf-specific GolfPose, even a frontier LLM. Full 1,400-clip GolfDB corpus.

**2. We found the bottleneck — and it wasn't where we expected.** Three different "better" models (MediaPipe Heavy, GolfPose, Sapiens-1B) produced cleaner, smoother, more anatomically perfect pose — yet scored *worse* on event detection. Anatomical quality and event accuracy were nearly uncorrelated. That decoupling told us the event detector was the ceiling.

**3. We built the fix and measured it.** Trained a 0.26M-param 1D-CNN on MotionBERT-Full's 3D trajectories. Held-out test (350 clips, zero shared source videos with training): **PCE@5 = 0.865**, beating the GolfDB SwingNet paper's ~0.76. We win because we feed clean 3D landmarks, not raw 160px video.

**4. We closed the loop into a demo.** `python demo.py swing.mp4` → MediaPipe Lite → MotionBERT-Full → smoothing → trained detector → 15 biomechanical indicators → **coaching scorecard**. Plus the UE5 handoff files (CSV/BVH) and a browser 3D viewer.

---

## The coaching layer (the MVP pivot)

The professors steered us toward "coaching-lite" — interpretive, not prescriptive. That's exactly what the scorecard does:

- 15 indicators computed from the 3D pose at each swing event: shoulder/hip turn, X-factor (coil), spine/posture, head movement, lead-arm extension, knee flex, weight shift, **tempo**.
- Each is scored against the **GolfDB tour-pro distribution** — a golfer is compared to pros, never to an invented "correct" number.
- Feedback is sparse and transparent: it names the metric and the event, and says nothing when you're in range.

**Validation point worth saying out loud:** median tempo across all 1,400 pros came out at **3.4:1** — the golf world's known "Tour tempo" is 3:1. We recovered a real biomechanical constant from our own pipeline. That's independent evidence the event timing is correct.

---

## Honest caveats (say these — they make the wins credible)

1. **The 0.865 uses GolfDB's official split, which has player overlap** (66/95 test golfers also in training, though never the same source video). It matches the SwingNet protocol so the comparison is fair, but it's not yet a strict brand-new-golfer test. Player-disjoint validation is the next step.
2. **The LLM event-detector number (0.258) is on 32 clips, not the full corpus.** Real, but smaller sample.
3. **Per-clip rotation metrics are noisy.** Monocular 3D compresses depth rotation, so shoulder/hip-turn angles read low in absolute terms — fine as a *relative* signal vs the pro band, shaky as an absolute per-swing measurement. The scorecard interprets relative, by design.
4. **Everything is GolfDB** — pre-cropped broadcast clips. Real phone-upload testing hasn't happened yet.

---

## If asked: "is this deployable?"

> "The detector is a 5 MB model that runs in under a second on CPU — yes. The demo already runs the full chain on a raw video. What's left before a real product: validate on a player-disjoint split, test on actual phone uploads (not pre-cropped broadcast clips), and scale the training set with our LLM-labeling pipeline. The architecture is proven and measured; the remaining work is generalization and data volume."

---

## One-line closer

> "We didn't just pick a model — we diagnosed the bottleneck, built the fix, measured it past the published state-of-the-art, and shipped it into a working demo."
