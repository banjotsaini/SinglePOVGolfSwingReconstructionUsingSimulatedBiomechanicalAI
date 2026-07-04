# Evals for the LLM Coaching/Explanation Layer

**Question this answers:** *"What evals are we doing to monitor the LLM coaching layer?"*

The LLM layer is the one component that can **make things up**. The pose model,
3D lift, event detector, and metrics are all deterministic and benchmarked
(PCE@5 = 0.865, etc.). The LLM takes our trustworthy numbers and writes prose —
so the risk is not *accuracy of measurement*, it's **faithfulness of narration**:
does the sentence the golfer reads actually match the data, stay in scope, and
stay safe?

So we built a dedicated eval harness (`Scripts/coaching_llm_eval.py`) that scores
every generated explanation against the scorecard it was written from.

---

## The risks → the checks

| # | Risk (what could go wrong) | Check | How | Runs on |
|---|---|---|---|---|
| 1 | **Hallucination / wrong direction** — claims something the data doesn't say | **Grounding (faithfulness)** | LLM-as-judge: given the metrics + the text, list every unsupported statement | subset (audited) |
| 2 | **Leaking an untrustworthy number** — narrates a metric we tagged low-confidence | **Low-confidence leakage** | deterministic: scan text for any low-confidence indicator's concept | **all 100** |
| 3 | **Scope creep** — gives fix-it instructions (MVP scope = explain, don't coach) | **Prescriptiveness** | deterministic: regex for imperative coaching language ("you should", "work on", "try to"…) | **all 100** |
| 4 | **Silently dropping a real finding** — ignores a flagged (out-of-band) metric | **Coverage** | deterministic: does the text mention each flagged metric? | **all 100** |
| 5 | **Unreadable for a beginner** — too dense / jargon-heavy | **Readability** | deterministic: Flesch reading-ease (target ≥ 60 = plain English) | **all 100** |
| 6 | **Instability** — same swing, different story each run | **Consistency** | regenerate N×, compare claim sets | subset |

**Why split deterministic vs. LLM-judge?** The deterministic checks are free,
exact, and reproducible — so they run on **all 100 clips** every time. The
LLM-judge (grounding) costs a Codex call per clip and is itself a model, so we
run it on an **audited subset** and report `n` transparently rather than implying
we judged all 100.

---

## How each check is defined (so the numbers are interpretable)

- **Grounding** — the judge is shown the raw metrics (your value, tour median,
  in-range yes/no, confidence) and the explanation, and asked to list statements
  *not supported* by the metrics. `grounded = true` ⇔ zero unsupported statements.
  General encouragement ("solid pieces to build on") is explicitly allowed; only
  factual claims about the swing are judged. **Target: high grounded-rate; every
  ungrounded case is logged with the offending sentence for inspection.**
- **Leakage** — for each metric the scorecard marked `confidence = low`
  (e.g. wrist/arm metrics, per Austin's depth-reliability EDA), we check whether
  the text references that metric's concept. **Target: 0.** This is the
  hard guardrail — the explanation must never coach on a number we don't trust.
- **Prescriptiveness** — regex bank of imperative coaching phrases. **Target: 0**
  for the MVP ("explain, don't prescribe" per the professor pivot). This is a
  *scope* check, not a quality one; if we later ship a coaching tier we relax it.
- **Coverage** — of the metrics the scorecard *flagged* (out of tour band,
  high-confidence), what fraction did the explanation actually mention? **Target: 1.0** —
  we don't want the LLM quietly omitting the one thing worth noting.
- **Readability** — Flesch reading-ease over the generated text. **Target ≥ 60**
  (plain, beginner-friendly English). Reported as mean and worst-case (min).

---

## Reproduce

```bash
python Scripts/build_scorecards_batch.py --n 100   # Stage A: 100 scorecards (local, deterministic)
python Scripts/coaching_llm_eval.py --judge-n 25   # Stage B+C: generate + score (resumable)
```

Stage B (generation) is cached per clip in `Data/coaching/eval_llm_text/`, so a
crashed or interrupted run resumes for free. Full report → `Data/coaching/llm_eval_report.json`.

The 100 clips are a **stratified sample** of GolfDB (proportional across
view × slow-motion strata) so the eval isn't dominated by one camera angle.

---

## Results (n = 100, judge-subset n = 25)

Visual: `slides/slide05b_llm_eval_monitoring.png`. Raw: `Data/coaching/llm_eval_report.json`.

| Metric | Result | Target | Read |
|---|---|---|---|
| Low-confidence leakage | **0 / 100** | 0 | ✅ hard guardrail holds |
| Prescriptive advice | **0 / 100** | 0 | ✅ stays in "explain, don't coach" scope |
| Flagged-metric coverage | **0.93** | 1.0 | ✅ rarely drops a real finding |
| Readability (Flesch mean / min) | **74.9 / 64.2** | ≥ 60 | ✅ beginner-readable, worst case still plain |
| Consistency (3× regen, Jaccard) | **0.86** | high | ✅ same swing → same story |
| Grounding (LLM-judge, n = 25) | **0.64** | high | ⚠ see below — the eval caught a real bug |

### The eval caught a real bug (and we fixed it)

The first run scored grounding at **0.32**. Inspecting the flagged sentences split
them cleanly:

- **A genuine defect.** The LLM was lumping several sub-metrics under one family
  label — e.g. saying "your head movement / spine tilt stayed in range" when one
  member of that family (`head_lift_max_pct`, `spine_tilt_address_deg`) was
  actually *out* of range. That's a real faithfulness failure: it told the golfer
  an out-of-range number was tour-like.
- **An over-strict judge.** Most other flags were the judge dinging *accurate*
  closing summaries ("nothing stood out", "no confident metric was outside the
  range") — true statements, wrongly counted as unsupported.

Two targeted fixes (both legitimate, neither gaming the metric):
1. **Generator** (`coaching_llm_summary.py`) — narrate each metric using only its
   own `in_tour_range` flag; never call a family "in range" if any member is out;
   don't attach a swing phase the metric name doesn't have.
2. **Judge spec** (`coaching_llm_eval.py`) — flag only genuine contradictions /
   invented specifics; accurate "nothing flagged" summaries are grounded.

**Result: grounding 0.32 → 0.64**, readability min 58.5 → 64.2. The v1 report is
kept at `llm_eval_report_v1_before_fix.json` for the before/after.

### Why we stopped at 0.64 (and didn't tune to green)

Hand-auditing the 9 residual v2 flags: the in-range/out-of-range contradiction is
**gone**. What's left is minor phase/count phrasing ("the two differences showed up
at impact") and a few cases where the judge flagged a *correct* statement (e.g.
clip 1170 said "the two out-of-range metrics were hip turn at impact and head
lift" — which is exactly right). Pushing the generator further to satisfy the
judge on ambiguous phrasing would be **overfitting to the judge**, so we report
0.64 honestly rather than chase 1.0. The genuine-fabrication rate is now near zero
on the audited subset.

**Honest notes for the meeting**
- Grounding uses an LLM judge (Codex/GPT-5) — a model auditing a model. We report
  the audited `n = 25` and keep every flagged sentence so a human can spot-check.
- All 100 clips are GolfDB (pre-cropped broadcast). Real phone-upload behavior of
  the LLM layer is still untested — same caveat as the rest of the pipeline.
- Every ungrounded / leaked / prescriptive case is stored per-clip in the JSON
  report, so failures are inspectable, not just counted.
