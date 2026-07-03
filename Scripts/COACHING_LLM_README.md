# Coaching LLM interpretation layer

Turns a measured swing **scorecard** into plain-English, **grounded** coaching language.
The LLM never re-analyzes video — it narrates already-computed, reliability-gated metrics,
and every factual claim is mechanically verified against the data.

## Shipped entrypoints

| Use | Command | Gated config (held-out 200) |
|---|---|---|
| Explain one swing | `python coaching_explain.py --scorecard <stem>_scorecard.json` | `f_strict_grounding` — 1.000 |
| Compare two sessions | `python coaching_compare.py --a <earlier>_scorecard.json --b <later>_scorecard.json` | `pg_readable` — 0.9998 |
| Ask a question | `python coaching_ask.py --scorecard <stem>_scorecard.json --question "How was my tempo?"` | `qa_readable` — 0.9910 (decision acc 1.000) |
| Full demo (wired) | `python demo.py --golfdb-clip <id>` | runs `coaching_explain` automatically |

The Q&A layer **refuses** questions it can't answer from the measured metrics
(unmeasured topics like ball flight/grip, low-confidence metrics like arm bend, or
"what should I fix" requests) rather than guessing.

Run with the project venv (`.venv/Scripts/python.exe`) — it has torch + pyarrow. The LLM
calls go through the authed **Codex CLI** by default; both CLIs exit cleanly if Codex is
offline (they leave the scorecard intact).

## How it works
- **Generator** `coaching_llm_summary_v2.py` — builds the prompt (rules + knowledge base +
  scorecard), calls the model with a JSON **output schema**, returns prose + a `claims` list
  (each claim tagged with the indicator key(s) and whether it's in/out of range).
- **Knowledge base** `Data/coaching/indicator_kb.json` — vetted per-indicator explainer cards
  + controlled glossary, so the model uses approved wording and invents no golf terms.
- **Deterministic grounding** `verify_grounding()` — checks every claim against the scorecard
  (no LLM judge): flags claiming in-range when out (or vice versa), low-confidence leakage,
  unknown metrics. Progress adds wrong-direction / invented-trend checks.

## Eval loop (how features are validated)
Fair, player-disjoint **dev(120) / held-out test(200)** split (`build_eval_split.py`).
Composite ∈ [0,1] = 0.45·grounded + 0.25·coverage + 0.15·non-prescriptive + 0.15·readable.
A feature (a prompt/KB/backend config in `Data/coaching/features/`) ships only at ≥0.8 on
the held-out test.

```
python coaching_eval_harness.py   --feature <id> --split test          # single-clip
python coaching_progress_eval.py  --feature <id> --split test          # progress pairs
```
Reports land in `Data/coaching/eval_reports/`. See `memory` note `coaching-llm-eval-loop`
for outcomes (grounding ≠ KB; multimodal declined; production picks).
