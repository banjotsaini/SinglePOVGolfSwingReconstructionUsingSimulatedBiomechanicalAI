# Motion Caddie — Coaching Q&A Chatbot (tool-calling) Plan

> **Goal.** Evolve the shipped one-shot swing Q&A (`coaching_ask.py`) into a **multi-turn
> Q&A chatbot that uses tool calling** to fetch grounded swing data, reason over it, and
> answer (or refuse). Backend runs on **AWS** (Lambda container + Anthropic API) alongside
> the existing demo. A **local version** mirrors the same core so we can iterate offline.

Status: **living doc** — updated each iteration. Last updated: iteration 1.

---

## 1. Why change — gap analysis (current vs target)

| | Current `coaching_ask.py` (`coaching_qa.generate`) | Target chatbot |
|---|---|---|
| Shape | **Single** structured-output call | **Agentic tool-use loop** (model decides what to fetch) |
| Turns | One question, one answer | **Multi-turn** conversation w/ history |
| Grounding | Whole scorecard dumped into the prompt; model self-reports `primary_indicator`; deterministic verifier checks the one claim | Model **only sees data it explicitly fetches via tools**; every number it states came from a tool result; verifier checks the transcript |
| Discovery | Model is told all 15 metrics up front | Model **calls `list_indicators`** to discover what's measurable, `get_indicator` to read one |
| Refusal | One enum `refusal_reason` | Same refusal taxonomy, but driven by tool results (unmeasured → no tool; low-confidence → tool flags it; fix-it → policy) |
| Scope | Single clip | Single clip **+ optional second clip** for progress questions (`compare_indicator`) |

**Net:** tool calling makes grounding *structural* (the model can't cite a number it never fetched)
and unlocks genuine multi-turn reasoning ("what about my hips?" → follow-up "and at impact?").
This is the same direction the eval-loop memory flagged as the real headroom (reasoning tasks,
off-metric hallucination). See `coaching-llm-eval-loop`.

## 2. Architecture (shared core, two runtimes)

```
            ┌─────────────────────────── coaching_chat.py (the core) ───────────────────────────┐
user turn → │  Conversation(messages, tools, system)                                            │
            │     loop:  backend.create(messages, tools, system)                                │
            │       ├─ stop_reason == "tool_use"  → dispatch tool(s) over the SwingContext      │
            │       │       (get_indicator / list_indicators / explain_term / compare_… )       │
            │       │     → append tool_result, continue loop                                   │
            │       └─ stop_reason == "end_turn"   → final assistant text  → verify_grounding    │
            └──────────────────────────────────────────────────────────────────────────────────┘
   Backend interface:  AnthropicBackend (prod / real key)   |   ScriptedBackend (offline tests)
   Data:  SwingContext = scorecard(s) + KB + confidence  → the ONLY thing tools can read
```

- **`SwingContext`** wraps one (or two) scorecards + the KB + `indicator_confidence.json`.
  Pure data accessor; no model. All tools read from it → grounding is enforced at the data layer.
- **Backend abstraction** lets the *entire loop* (tool dispatch, multi-turn, refusal, verifier)
  be tested offline with a `ScriptedBackend` (canned tool_use/text turns), then run for real with
  `AnthropicBackend` (claude-opus-4-8, the AWS-validated backend). No Codex in the loop — tool use
  is native to the Anthropic Messages API; Codex stays the dev helper, not the runtime.

## 3. Tools exposed to the model

All deterministic, all read-only over `SwingContext`. Names/inputs are the contract:

| Tool | Input | Returns | Purpose |
|---|---|---|---|
| `list_indicators` | – | keys + label + plain_name + confidence_tier + measured? | discovery: what *can* be answered |
| `get_indicator` | `key` | value, pro_median, pro_band, percentile, in_range, confidence_tier, KB card | the grounding read; **flags low-confidence** so the model refuses |
| `get_flagged_observations` | – | review-severity feedback list | "what stood out?" |
| `get_swing_summary` | – | n in-range / n flagged, events present, club/view | overview questions |
| `explain_term` | `term` | KB glossary gloss, or `not_in_glossary` | "what is X-factor?" without inventing |
| `compare_indicator` | `key` | session-A vs session-B value + delta + direction (only if 2nd clip loaded) | progress questions |

**Unknown key / unmeasured term → tool returns a structured "not measured" result**, not an error —
the model reads that and refuses gracefully (no hallucinated number).

## 4. Grounding & safety (keep the validated discipline)

1. **System prompt = the QA rules** (reuse `coaching_qa.RULES` taxonomy): answer only from tool
   results; never invent numbers; refuse **unmeasured** topics, **low-confidence** metrics, and
   **fix/advice** asks; gloss terms from the KB only.
2. **Tools are the sole data source.** The scorecard is *not* dumped into the prompt — the model
   must fetch. A number in the answer with no matching `get_indicator`/`compare_indicator` call is a
   grounding violation.
3. **Deterministic transcript verifier** (`verify_chat_grounding`): from the tool-call log + final
   text, check (a) every indicator the model discussed was actually fetched, (b) in/out-of-range
   statements match the fetched `in_range`, (c) no low-confidence metric was used to answer, (d) no
   prescriptive language when it "answered". Mirrors `verify_grounding` but over a transcript.
4. **Tolerant failure** like the existing CLIs: any backend error → graceful message, never a crash.

## 5. AWS backend (extends the existing handoff, not a new stack)

Add a **`/chat` route** to the same Lambda container in `AWS_DEPLOY_HANDOFF.md`:

- **Stateless multi-turn**: the browser holds the transcript and posts
  `{clip_id, messages:[…]}`; Lambda reconstructs `SwingContext` from the cached parquet/scorecard
  (already built for the demo) and runs the loop. No server-side session store → still scale-to-zero.
- **Same Anthropic key** from Secrets Manager, same `claude-opus-4-8`, same reserved-concurrency /
  allow-list cost guards. Tool calls run **inside** Lambda (the tools are local Python over the
  cached scorecard) — no extra network hops, no extra cost surface.
- **Cost note**: a chat turn is a *few* model round-trips (tool calls), so ~Nx a one-shot
  explanation but still sub-cent. Bound it with a **max-tool-iterations** cap (e.g. 6) per turn.
- Reuses the demo's per-clip scorecard build; the `/chat` handler is a thin adapter like `/clip`.

## 6. Local version (for offline iteration)

- `Scripts/coaching_chat.py` — the core + a CLI: `--scorecard <stem>_scorecard.json [--compare <b>]`,
  one-shot `--question` or interactive REPL. Uses `AnthropicBackend` when `ANTHROPIC_API_KEY` is set.
- `Scripts/test_coaching_chat.py` — **no key needed**: drives the full loop with `ScriptedBackend`
  to prove tool dispatch, multi-turn, refusal, and the verifier all work deterministically.
- Eval (later iteration): replay the existing gold QA bank through the chat loop and score
  decision-accuracy + grounding, so we don't regress the shipped `qa_readable` numbers.

## 7. Iteration roadmap (research → gaps → implement → test, ×4–5)

- [x] **It.1** — design + plan; built `SwingContext`, 6 tools, backend abstraction, loop, verifier; **28 offline tests green** (already caught a real bug: confidence-tier loader read the wrong JSON nesting).
- [x] **It.2** — Codex independent gap-review (P0/P1/P2). Real issues confirmed: thinking-block stripping mid-loop, client-transcript trust boundary, weak verifier, cap-fallback message shape, naive `stop_reason`.
- [x] **It.3** — hardening: thinking **off by default** (+ block preservation if on), explicit `stop_reason` branching (`pause_turn`/`max_tokens`/`refusal`), legal cap-fallback (no double user msg), **numeric grounding check** (invented numbers now caught). Tests grew to **35 green** (message-shape, multi-tool-per-turn, numeric-hallucination).
- [x] **It.4** — AWS `/chat` adapter `deploy/chat_handler.py` (stateless, **text-only untrusted history**, cost caps, allow-list) + **15 offline tests green**; handoff doc §Phase G.
- [x] **HTML concept demo** — `coaching_chatbot_demo.html`: interactive chat (metric / progression / refusal), live tool-call trace, architecture diagram + grounding explainer. Real scorecard numbers; verified in-browser (no console errors).
- [x] **It.5** — chatbot eval harness (`coaching_chat_eval.py`) replaying the gold QA bank through the tool loop; grades decision-accuracy + grounding + tool-choice.
- [x] **Live via Codex** — added `CodexBackend` (emulates the tool-use loop via Codex structured output, so it runs with the authed Codex CLI — no Anthropic key). `--backend codex` on the CLI and eval.

### Live eval results (Codex, 12 stratified gold questions, clip 1292)
- **Round 1:** decision **0.50**, grounded 0.83 → the failures were **grader/verifier bugs, not the agent**: (a) the model uses curly apostrophes (`can't`) that the straight-quote regexes missed, so valid refusals scored as answers; (b) naming a low-conf metric *to decline it* was mis-flagged as a leak.
- **Round 2:** decision **1.00**, grounded 0.92 → one more verifier miss: a model that declined to *prescribe* then gave a fully-grounded description was flagged prescriptive ("practice" in the refusal clause; "prescribe" wasn't a refusal cue).
- **Round 3 (final):** **decision 1.000 · grounded 1.000 · tool-choice 1.000** — passes the gate (decision ≥0.95, grounded ≥0.98).
- Takeaway: the agent refuses/grounds correctly live; the eval loop's value was hardening the *verifier* against real model phrasing (smart quotes, decline-to-prescribe, name-to-refuse).

### Review-driven design decisions (from It.2)
- **Trust boundary:** client posts prior turns as **plain text only**; server drops any tool blocks and re-grounds every new answer. A forged history can add fake *conversational memory* but never a fake *number*.
- **Thinking off** for chat: short grounded Q&A doesn't need it, and it sidesteps the resend-thinking-blocks requirement + max_tokens truncation.
- **Grounding = numbers must trace to tools:** the verifier now rejects any measurement-looking number in the answer that no tool returned (integers <13 exempted as phrasing).
- **Cost:** `/chat` can't use the once-per-clip cache, so spend is bounded by `MAX_TOOL_ITERS`, history/question/body caps, reserved concurrency, and the Anthropic console cap.

## 8. Non-goals / guardrails

- Don't break the validated grounding (same KB, same refusal taxonomy, same verifier philosophy).
- No new always-on infra; reuse the scale-to-zero stack.
- No prescriptive coaching (MVP describes, doesn't fix) — unchanged.
</content>
</invoke>
