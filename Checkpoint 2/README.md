# Checkpoint 2 — Coaching-Lite MVP

**From "which model?" to a working product.** Checkpoint 1 was the model benchmark.
Checkpoint 2 is the MVP the professors steered us toward: a single phone video →
detected swing phases → motion metrics → a plain-English swing explanation.

---

## The headline results

| Result | Number | Note |
|---|---|---|
| **Swing-event detector (trained 1D-CNN)** | **PCE@5 = 0.865** | beats the published GolfDB SwingNet (~0.76) |
| **Generalization (player-disjoint, 49 unseen golfers)** | **PCE@5 = 0.864** | ≈ standard split → no player-overlap inflation |
| Heuristic detector we started with | 0.170 | the bottleneck we diagnosed + fixed |
| Tempo recovered across 1,400 pros | 3.4:1 | matches the known "Tour tempo" 3:1 norm |

**The story:** we diagnosed that the *event detector*, not the pose model, was the
ceiling (three models — MediaPipe Heavy, GolfPose, Sapiens-1B — had cleaner pose
but worse event accuracy). We trained the fix; it beats the published baseline and
generalizes to golfers it never saw.

---

## The working demo

```bash
python Scripts/demo.py path/to/swing.mp4
```

Chain: **MediaPipe Lite (2D) → GolfPose MixSTE (3D) → 1D-CNN events (raw 3D) +
One-Euro smoothing (measurement branch) → 15 biomechanical indicators → coaching
scorecard → grounded Claude read + browser 3D replay.** Fully working end-to-end today.
(The Unreal Engine handoff has been retired; BVH export is opt-in via `pipeline.py --bvh`.)

`demo/` holds the clip-1292 (Charley Hull) bundle: scorecard PNG+JSON, 3D
landmark CSV, 2D overlay MP4, and the standalone browser 3D viewer.

---

## Slide-ready visuals (`slides/`) — mapped to the Week 7a deck

| Deck slide | File | What it shows |
|---|---|---|
| 4 — Motion Metrics | `slide04_motion_metrics_scorecard.png` | scorecard: 15 indicators vs tour band + confidence markers |
| 4/5 hero | `slide04b_hero_scorecard_with_llm.png` ⭐ | metrics layer → LLM explanation in one card |
| 5 — LLM Interpretation | `slide05_metrics_to_llm.png` + `slide05_llm_summary_text.txt` | 3-stage flow + the real generated explanation |
| 11 — Model Selection | `slide11_event_detector_results.png` ⭐ | the breakthrough: 0.170 → 0.865 + player-disjoint + SwingNet line |
| 11 — supporting | `slide11b_event_timeline_accuracy.png` | predicted vs ground-truth events on 5 real clips |
| 11 — full board | `slide11_model_selection_leaderboard.png` | all 25 pipelines |
| 12 — Technical Pipeline | `slide12_technical_pipeline.png` | 7-stage MVP flow with real components |

---

## What's new since Checkpoint 1

1. **Trained swing-event detector** (`Scripts/train_event_detector.py`, `event_detector.py`) — the 0.865 result + player-disjoint validation.
2. **Coaching layer** — 15 biomechanical indicators (`coaching_indicators.py`), GolfDB pro reference bands, reliability-gated feedback engine, rendered scorecard.
3. **LLM interpretation layer** (`coaching_llm_summary.py`) — the beginner-friendly explanation the MVP is built around.
4. **Austin's findings integrated** — his depth-reliability EDA drives per-indicator confidence; low-confidence metrics never raise a coaching note. See `reports/AUSTIN_FINDINGS_INTEGRATION.md`.
5. **End-to-end demo** (`Scripts/demo.py`) — raw video to scorecard in one command.

---

## Folder contents

| Folder | What |
|---|---|
| `slides/` | Slide-ready PNGs (1.2 MB) + the LLM summary text |
| `reports/` | RESULTS.md, talking points, Austin integration writeup |
| `demo/` | Clip-1292 end-to-end bundle (scorecard, CSV, BVH, MP4, 3D viewer) |
| `interactive/` | Standalone 3D swing viewer (`1292_preview_3d.html`) — opens in any browser |
| `diagrams/` | Editable Excalidraw sources (MVP pipeline, metrics→LLM) |
| `models/` | Trained `event_detector_tcn.pt` + reference bands + confidence + player-disjoint result |

---

## Honest caveats (carry into the meeting)

- **Player-disjoint result (0.864) closes the generalization question** — but everything is still GolfDB (pre-cropped broadcast clips). No real phone-upload testing yet.
- **The LLM summary uses Codex** (cloud, ~$0.10/call, needs internet). Fine for the demo; production would swap to a local/cheaper model.
- **Per-clip rotation metrics are noisy** (monocular depth) — which is exactly why the scorecard interprets relative to the pro band and gates low-confidence metrics rather than reporting absolutes.

---

## What's still open

- ~~**UE5 converter**~~ — dropped; the Unreal Engine handoff is retired (3D replay is browser-native).
- **Dataset scaling** — single-person gate + scale the YouTube LLM-labeler to ~1,000 clips on AWS.
- **Real phone-video validation** — collect ~20 clips in deployment conditions.
