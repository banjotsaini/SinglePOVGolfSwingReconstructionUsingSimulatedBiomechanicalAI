# Motion Caddie — AWS Deployment Handoff (for the engineer + their Claude)

> **Read this first.** You are picking up the AWS deployment of "Motion Caddie", a golf-swing
> coaching capstone. The original author **set the architecture but cannot build the AWS infra
> themselves** — that's your job. This doc is written so a fresh Claude Code agent with **zero prior
> context** can execute the build end-to-end. Every path and code fact below was verified against
> the repo on 2026-06-30; still re-verify before you rely on it (`git log` may have moved on).
>
> **Companion doc:** [`AWS_HOSTING_PLAN.md`](AWS_HOSTING_PLAN.md) holds the *why* (architecture
> rationale, cost posture) and the full **team-access / IAM design (§5)**. This doc is the *how*.

---

## 0. TL;DR — what you are shipping

A **scale-to-zero serverless demo** of the coaching pipeline:

```
Browser → CloudFront → S3 (static clip-picker UI)
                  └→ Lambda Function URL → Lambda (container image, CPU-only)
                          ├ loads a pre-cached 3D pose parquet for the chosen clip
                          ├ runs the trained event detector (tiny CNN, CPU)
                          ├ builds + renders a coaching scorecard (JSON + PNG)
                          ├ calls Claude (Anthropic API) for a grounded explanation
                          └ returns { scorecard JSON, scorecard PNG URL, explanation }
```

No GPU. No always-on compute. AWS cost ≈ cents/mo (covered by the account's $250 promo credits).
Claude runs on the **direct Anthropic API** (separate cents/mo bill) — **not Bedrock** (the AWS
credits don't cover Bedrock).

**Definition of done** is in §8. Do the human prereqs (§5 / Phase 0) and the build phases (§6) in
order; §7/§7b/§7c are reference (fallback, security, ops). Read the §11 landmines before you start.

---

## 1. Rules of engagement (read before touching anything)

- **You drive the code + IaC; the human runs console-only / credential steps.** You cannot create
  the human's AWS root MFA, Identity Center users, or paste an API key into Secrets Manager from
  here — produce the exact commands/clicks and have the human run them. Everything else (handler,
  Dockerfile, SAM template, UI, smoke tests) you build and test.
- **Never commit secrets.** The `ANTHROPIC_API_KEY` lives only in AWS Secrets Manager. No keys in
  the image, in `template.yaml`, in env-var literals, or in git. Grep your diff before committing.
- **This repo has many untracked/modified files** (it's an active capstone). When you commit, stage
  **only the files you create/change for the deploy** — never `git add -A`.
- **The model weight and 3D cache are gitignored** (`Models/`, `Data/eval_runs/`). They are *not*
  in git — you bake them into the container at build time from the local working tree. Confirm they
  exist locally before building (§3).
- **Don't break the science.** The coaching explanation's grounding was validated on a held-out set
  with the `f_strict_grounding` feature. Switching the LLM backend from Codex→Anthropic is a
  **config flip**, not a rewrite — see §4. Keep the same KB, prompt, and verifier.
- **Ask the human to confirm the §9 open decisions** before deploying (region, who can deploy,
  Function URL auth).

---

## 2. What Motion Caddie does (just enough domain context)

Input: a golf swing. Output: a "coaching scorecard" — biomechanical indicators (tempo, X-factor,
posture, weight shift, etc.) compared to tour ranges, plus a beginner-friendly **grounded** text
explanation written by an LLM that is only allowed to describe measured numbers (no invented advice).

The **full** pipeline (raw video → 2D MediaPipe → 3D GolfPose MixSTE → events on raw 3D + One-Euro smoothing on the measurement branch → scorecard)
needs a GPU and the heavy ML stack. **You are NOT deploying that.** You are deploying the
**cached-clip demo path**, which starts from a *pre-computed* 3D pose file and is pure CPU. This is
the entire reason the demo is cheap.

---

## 3. Verified repo facts (the demo path)

**Entry point (run this locally to see the whole flow):**
```bash
python Scripts/demo.py --golfdb-clip 1292
# → writes Data/demo/clip1292/{clip1292_scorecard.json, clip1292_scorecard.png}
```

**What `demo.py --golfdb-clip <id>` actually does** (`Scripts/demo.py:42`):
1. Reads cached 3D parquet: `Data/eval_runs/golfpose3d_from_mediapipe_lite/<id>.parquet`
2. Reads clip metadata (player/club/view) from `golfdb/golfDB.pkl`
3. `scorecard_step.py` → event detector + `build_scorecard` + `render_scorecard` → JSON + PNG
4. `coaching_explain.py` → LLM explanation written back into the scorecard JSON

**Files/data the container must contain** (all verified to exist locally):

| Artifact | Local path | Size | Notes |
|---|---|---|---|
| Event-detector weight | `Models/event_detector_tcn.pt` | ~1.1 MB | `event_detector.py:22` `DEFAULT_WEIGHTS` — **MixSTE-retrained** (2026-06-30, PCE@5 0.865); regenerate via `train_event_detector.py --lifter golfpose3d_from_mediapipe_lite` |
| 3D pose cache | `Data/eval_runs/golfpose3d_from_mediapipe_lite/<id>.parquet` | 131 MB all clips | **bake only the curated subset** (§6) |
| Clip metadata | `golfdb/golfDB.pkl` | small | player/club/view header |
| Scripts | `Scripts/*.py` | small | see import graph below |
| Feature config | `Data/coaching/features/f_strict_grounding.json` | tiny | **flip backend → anthropic** (§4) |

**Python import graph for the demo (the only modules you need):**
`demo.py` → `scorecard_step.py` → `event_detector.py` (→ `train_event_detector.py`, `eval_utils.py`),
`coaching_indicators.py`, `coaching_scorecard.py`, `render_scorecard.py`; and
`coaching_explain.py` → `coaching_eval_harness.py` → `coaching_llm_summary_v2.py`.

**⚠️ Transitive `cv2` import:** `event_detector.py` does `from eval_utils import SWING_EVENTS`, and
`eval_utils.py:29` imports `cv2` at module load. So the container **needs OpenCV** even though the
demo never decodes video. Use **`opencv-python-headless`** (no GUI/X11 libs). (Alternative: lift
`SWING_EVENTS` out of `eval_utils` into a tiny constants module so `cv2` isn't pulled — optional
refactor, only if you want to drop the dep.)

**Slim runtime dependencies** (NOT the repo's full `requirements_eval.txt`, which includes
tensorflow/mediapipe/ultralytics — none needed here):
```
torch            # CPU build — install from https://download.pytorch.org/whl/cpu
numpy==1.26.4    # pin <2 for torch 2.5.x compatibility; pandas 3.0 is fine with it
pandas
pyarrow          # parquet read
matplotlib       # scorecard PNG render
opencv-python-headless   # transitive via eval_utils (see warning above)
anthropic        # Claude SDK
```
(No scipy/sklearn needed by this path — verified from imports.)

---

## 4. The one critical code change: LLM backend flip (Codex → Anthropic)

The explanation backend is selected **by the feature config**, not by code. `coaching_explain.py`
loads `f_strict_grounding` and `coaching_eval_harness.generate()` branches on `cfg["backend"]`
(`coaching_eval_harness.py:126`): `"anthropic"` → uses the `anthropic` SDK
(`anthropic.Anthropic().messages.create(...)`, model `claude-opus-4-8`, with prompt caching),
reading `ANTHROPIC_API_KEY` from the environment. `"codex"` (current default) shells out to the
local Codex CLI — **which will not exist in Lambda.**

**Change:** in `Data/coaching/features/f_strict_grounding.json`, set:
```json
"backend": "anthropic",
"model": "claude-opus-4-8",
```
Leave `use_kb`, `rules_append`, etc. untouched (that's what was validated). No other code change is
required to switch backends.

**Grounding-tolerant by design:** `coaching_explain.py` catches any LLM failure and exits 0, leaving
the scorecard intact (`coaching_explain.py:48`). Good for resilience — but in the Lambda you want a
*real* explanation, so surface failures in logs and fail the request if the explanation is missing
(don't silently return a scorecard with no text).

---

### 4b. Anthropic API integration details (verified against the Claude API reference)
- **Model:** `claude-opus-4-8` is current and correct — **$5 / $25 per 1M tokens** (input/output),
  1M context, 128K max output. Adaptive thinking only (`thinking: {type: "adaptive"}`); the old
  `budget_tokens` form 400s on this model. The repo's existing `_run_anthropic` backend already
  targets this model — don't "modernize" it without checking; just confirm it doesn't send
  `budget_tokens`/`temperature` (both rejected on 4.8).
- **Prompt caching reality check.** The KB is sent as a stable prefix with `cache_control` for cost
  savings, BUT: the default cache TTL is **5 minutes** and the minimum cacheable prefix on Opus 4.8
  is **4096 tokens**. So (a) the KB cache helps only within a burst of calls, **not between demos
  minutes/hours apart**, and (b) if the KB is under ~4096 tokens it silently won't cache at all.
  **Conclusion: don't rely on prompt caching to make repeat demos free — the S3 per-clip cache
  (Phase F) is what guarantees $0 repeat cost.** Prompt caching is a nice-to-have on top.
- **SDK timeout/retry.** The `anthropic` SDK defaults to a **10-minute timeout** and **2 automatic
  retries** (429/5xx/connection errors, exponential backoff). Worst-case wall-clock is
  `timeout × (retries+1)`. For Lambda: set a **short explicit client timeout** (e.g.
  `Anthropic(timeout=30, max_retries=2)`) so a hung call can't blow the Lambda budget, and make sure
  the **Lambda function timeout exceeds** the client's worst case. Short grounded explanations
  return fast; no streaming needed at this size.
- **Cost is trivial.** Explanations are a few hundred tokens in/out → well under a cent each, capped
  by the Anthropic console limit. The Anthropic bill is separate from the AWS credits.

## 5. Prerequisites the human must complete (Phase 0)

These are console/credential steps you cannot do from the CLI session. Generate the exact
instructions for the human; do not block your local work (§6 Phase A/B) waiting on them.

1. **Secure AWS root + stand up team access** per [`AWS_HOSTING_PLAN.md`](AWS_HOSTING_PLAN.md) **§5**:
   enable root MFA; enable **IAM Identity Center** (org instance); create the **Admins / Developers /
   Billing** groups and the least-privilege **`MotionCaddieDeployer`** permission set; create the 4
   users (owner + 3 teammates). The full permission-set JSON is in §5.3 of that doc.
2. **Local CLI auth (no long-lived keys):** `aws configure sso` → `aws sso login --profile motion-caddie`.
   Install **AWS SAM CLI** and **Docker** (needed for container builds / `sam local`).
3. **Anthropic API key:** create at console.anthropic.com, set a **usage cap**. The human stores it
   in Secrets Manager (you create the empty secret in the SAM template; they put the value in, or
   you guide `aws secretsmanager put-secret-value`). Secret name suggestion: `motion-caddie/anthropic-api-key`.
4. **Confirm region** (default `us-east-1`) and the §9 decisions.

You can fully build and locally test Phases A–B with just an Anthropic key in your own env
(`export ANTHROPIC_API_KEY=...`) before any AWS access exists.

---

## 6. Build phases (you do these)

### Phase A — Lambda handler (`deploy/app.py`)
A thin adapter: event with a `clip_id` → run the demo flow → return JSON. Reuse the existing scripts;
don't reimplement the pipeline. Skeleton (verify against current script signatures before trusting):

```python
import base64, json, os, sys
from pathlib import Path

# scripts + project root are baked into the image (see Dockerfile)
ROOT = Path(os.environ.get("APP_ROOT", "/var/task"))
sys.path.insert(0, str(ROOT / "Scripts"))

CACHE = ROOT / "Data" / "eval_runs" / "golfpose3d_from_mediapipe_lite"
ALLOWED = set(json.loads(os.environ.get("ALLOWED_CLIPS", "[]")))  # curated allow-list

def handler(event, _ctx):
    clip_id = int((event.get("queryStringParameters") or {}).get("clip_id")
                  or json.loads(event.get("body") or "{}").get("clip_id"))
    if ALLOWED and clip_id not in ALLOWED:
        return _resp(400, {"error": f"clip {clip_id} not in demo set"})

    parquet = CACHE / f"{clip_id}.parquet"
    if not parquet.exists():
        return _resp(404, {"error": f"no cached clip {clip_id}"})

    # --- run the pipeline in-process (import the step modules, don't subprocess) ---
    # NB: demo.py uses subprocesses for the GPU mediapipe+torch co-import problem,
    # which does NOT apply on the cached path. In Lambda, call the functions directly:
    #   load parquet → event_detector.EventDetector().predict(xyz)
    #   → coaching_scorecard.build_scorecard → render_scorecard.render (PNG to /tmp)
    #   → coaching_explain.explain(scorecard_json)   # backend=anthropic via feature
    # Read back the JSON + base64 the PNG.
    ...
    return _resp(200, {
        "clip_id": clip_id,
        "scorecard": scorecard_dict,
        "explanation": scorecard_dict.get("llm_explanation"),
        "grounding": scorecard_dict.get("llm_grounding"),
        # PREFER returning a URL, not the bytes — see the 6 MB limit note below.
        "png_url": png_s3_or_cloudfront_url,        # recommended
        # "png_base64": base64.b64encode(png_bytes).decode(),  # only if PNG is small
    })

def _resp(code, body):
    return {"statusCode": code,
            "headers": {"content-type": "application/json",
                        "access-control-allow-origin": "*"},
            "body": json.dumps(body)}
```

Implementation notes:
- **⚠️ 6 MB response cap.** A Lambda (incl. Function URL) response payload is capped at **~6 MB**
  (6,291,556 bytes), and the platform adds ~1 MB of overhead on top of your body. A base64'd
  scorecard PNG (~33% inflation) is *probably* under that, but it's fragile. **Recommended pattern:**
  write the PNG to the `motion-caddie-demo-assets` S3 bucket and return a **presigned GET URL** (or a
  CloudFront URL if the bucket is fronted). This also dovetails with the per-clip caching in Phase F —
  cache the PNG + explanation in S3 and the response is always tiny. Keep `png_base64` only as a
  fallback for known-small images.
- **Write to `/tmp` only** — Lambda's filesystem is read-only except `/tmp` (512 MB default; bump via
  `EphemeralStorage` if a render needs more). The scripts write the scorecard JSON/PNG next to the
  parquet by default; point them at `/tmp` (pass an out-dir, or copy logic from `scorecard_step.py`).
- **Lazy / global model load.** Load the torch weight **once at module scope** (outside `handler`) so
  warm invocations reuse it (~150 ms warm vs 10–25 s cold). Don't reload per request.
- **Set the Anthropic key into the env before the import that needs it.** Fetch the secret **once at
  cold start (module scope), not per invocation** — either via the **AWS Parameters & Secrets Lambda
  Extension** (a layer that caches the secret locally so you avoid a Secrets Manager API call on
  every request) or a single `boto3` `get_secret_value` cached in a module global. Then set
  `os.environ["ANTHROPIC_API_KEY"]` before `coaching_explain.explain` runs (the `anthropic` SDK
  reads that env var). Don't bake the key into an env-var literal in the template.
- **Reuse `coaching_explain.explain()`** (it's importable, not just a CLI) — it does the
  feature-load → generate → grounding-verify → write-back in one call.
- Add a tiny `GET /clips` route (or a static JSON) so the UI can list the curated clips.

### Phase B — Container image (`deploy/Dockerfile`)
Base on the **AWS Lambda Python base image**, install CPU torch, copy scripts + weights + curated
clips. Skeleton:

```dockerfile
FROM public.ecr.aws/lambda/python:3.11

# CPU-only torch (huge saving vs the cuda wheels in requirements_eval.txt)
COPY deploy/requirements-lambda.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements-lambda.txt

# code + assets (weights/clips are gitignored — copied from the local working tree)
COPY Scripts/                         ${LAMBDA_TASK_ROOT}/Scripts/
COPY Models/event_detector_tcn.pt     ${LAMBDA_TASK_ROOT}/Models/
COPY golfdb/golfDB.pkl                ${LAMBDA_TASK_ROOT}/golfdb/
COPY Data/coaching/features/          ${LAMBDA_TASK_ROOT}/Data/coaching/features/
COPY deploy/clips/                    ${LAMBDA_TASK_ROOT}/Data/eval_runs/golfpose3d_from_mediapipe_lite/
COPY deploy/app.py                    ${LAMBDA_TASK_ROOT}/

CMD ["app.handler"]
```
- `requirements-lambda.txt` = the slim list from §3 (minus torch, installed separately from the CPU index).
- Keep the curated parquets in `deploy/clips/` (copied/symlinked from the cache) so the build context
  is small. Expect the final image ≈ **1–1.5 GB** (well under Lambda's 10 GB image limit).
- **Test locally before deploying:** `sam local invoke` or the Lambda Runtime Interface Emulator,
  with `ANTHROPIC_API_KEY` passed in. Confirm you get a scorecard + a non-empty grounded explanation.

### Phase C — Curate the demo clips
Pick **~10–25** representative GolfDB clip ids (vary player / club / view: down-the-line vs face-on).
For each, confirm `python Scripts/demo.py --golfdb-clip <id>` produces a good scorecard + a sensible
explanation. Copy just those `<id>.parquet` files into `deploy/clips/`. Record the id list in
`ALLOWED_CLIPS` (handler env) and in the UI.

### Phase D — SAM template (`template.yaml`)
One stack, `sam delete`-able. Resources:
- `AWS::Serverless::Function` — `PackageType: Image` (+ `Metadata.DockerContext`/`Dockerfile`
  pointing at the Dockerfile above), `FunctionUrlConfig` (see CORS note below), memory ≈
  2048–3008 MB (Lambda scales CPU with memory — up to 6 vCPU at 10 GB — so more memory ≈ faster
  torch + faster cold start), timeout ≈ 60–120 s (Function URLs do **not** have API Gateway's 29 s
  cap; Lambda max is 15 min, but keep it tight), ephemeral `/tmp` storage bumped via
  `EphemeralStorage` if a render needs >512 MB.
  - **`FunctionUrlConfig`** fields: `AuthType` (`NONE` + shared-token check, or `AWS_IAM` — §9),
    `InvokeMode: BUFFERED` (default; fine), and a **`Cors`** block. ⚠️ **The static UI on CloudFront
    calls the Function URL cross-origin**, so without `Cors` the browser blocks every request. Set
    `Cors.AllowOrigins` to the CloudFront domain (or `*` for the demo) and `AllowMethods`/`AllowHeaders`.
  - **Cold start:** a CPU-torch container cold-starts in ~10–25 s; warm ~150 ms. For a live
    presentation, either **pre-warm** (hit the URL once before demoing) or set **provisioned
    concurrency = 1** for the demo window (small cost, kills the cold-start risk on stage), then
    remove it after. Don't leave provisioned concurrency on — it defeats scale-to-zero.
- `AWS::SecretsManager::Secret` — `motion-caddie/anthropic-api-key` (value set out-of-band by human).
- **Execution role** — least privilege: read **only** that one secret, write CloudWatch Logs,
  read/write the explanation-cache S3 bucket. Nothing else. (SAM generates a basic role; add the
  secret + bucket policies via the function's `Policies`.)
- `AWS::S3::Bucket` ×2 — `motion-caddie-demo-web` (static UI) and `motion-caddie-demo-assets`
  (per-clip explanation cache + optional pre-rendered fallback).
- `AWS::CloudFront::Distribution` — in front of the web bucket via **Origin Access Control (OAC)**;
  keep the bucket **private** (no public-read), grant CloudFront read via the bucket policy.
- Inject the secret into the function (env var from Secrets Manager, or the AWS Parameters/Secrets
  Lambda extension — see Phase A). `ALLOWED_CLIPS` env = the curated id list.
- **ECR repo: don't hand-author it.** The SAM CLI creates and manages the ECR repository for an
  `Image` function via a **companion stack** — run `sam deploy --guided` (it prompts to create the
  repo) or `sam deploy --resolve-image-repos` for non-guided. Deleting the function later
  auto-deletes the repo. (This is why §5.3's deployer policy grants `ecr:CreateRepository`.)
- Build/deploy: `sam build && sam deploy --guided` (then non-guided with the saved `samconfig.toml`).

### Phase E — Deploy + smoke test
```bash
sam build
sam deploy --guided          # first time; pick region, stack name "motion-caddie"
curl "$(FUNCTION_URL)?clip_id=1292"   # expect JSON with scorecard + explanation + png_url
```
Verify: 200, non-empty `explanation`, `grounding.grounded == true`, `png_url` fetches a valid PNG.

### Phase F — Frontend + polish
- Minimal static UI (`deploy/web/index.html` + a little JS): clip-picker dropdown (the curated ids)
  → fetch the Function URL → show the PNG and the explanation text. Upload to the web bucket; serve
  via CloudFront.
- **Cache explanations per clip in S3** (`motion-caddie-demo-assets`): on a cache hit, skip the
  Anthropic call entirely → repeat demos cost $0 in tokens and are instant.
- **AWS Budgets** alert (~$5–10/mo) to the team; short **runbook** + **teardown** steps (§10).

---

### Phase G — `/chat` route (coaching Q&A chatbot)  ← added for the tool-calling chatbot
The demo also ships a **multi-turn Q&A chatbot** (`Scripts/coaching_chat.py`) that uses **Anthropic
tool calling**: the model fetches each swing number via read-only tools instead of being handed the
scorecard, so every claim is grounded and auditable. The Lambda adapter is already written and
offline-tested — **`deploy/chat_handler.py`** (+ `deploy/test_chat_handler.py`, 15 tests, no key).

- **Route:** add a second path (or a `POST` branch in `app.py`) that calls
  `chat_handler.handler(event)`. Request body: `{clip_id, question, history:[{role,content}], compare_clip_id?}`.
  Response: `{answer, grounded, violations, tools_used, iterations, stop}`.
- **Stateless & trust-bounded (P0 from the review):** the browser holds the transcript and posts it
  back each turn. The handler treats that history as **untrusted plain text** — `_sanitize_history`
  drops any client-supplied `tool_use`/`tool_result` blocks, so a forged history can't inject a fake
  measurement. Every number in the new answer is re-fetched server-side this turn and re-verified by
  `verify_chat_grounding`. Do **not** change this to trust client tool history.
- **Cost caps (chat ≠ the cached clip path — it can't rely on the once-per-clip cache):** a turn is
  several model round-trips, so spend is bounded by explicit caps, not caching:
  `MAX_TOOL_ITERS=6` (in `coaching_chat.py`), `CHAT_MAX_HISTORY_MSGS`, `CHAT_MAX_QUESTION_CHARS`,
  `CHAT_MAX_BODY_BYTES` (env-overridable in `chat_handler.py`), the `ALLOWED_CLIPS` allow-list, plus
  the function's **reserved concurrency** (compute cap) and the **Anthropic console hard cap** (token
  cap). Keep all of these; unlike `/clip`, there is no S3 cache backstop for arbitrary chat turns.
- **Backend:** same `AnthropicBackend` (`claude-opus-4-8`), same `ANTHROPIC_API_KEY` from Secrets
  Manager. Thinking is **off** by default for chat (short task; avoids the thinking-block-preservation
  requirement and reduces truncation risk). No Codex in the runtime — tool use is native to the
  Messages API.
- **Scorecards:** the handler reads `Data/demo/<clip>/<clip>_scorecard.json` (built by the same
  CPU-only demo path). Bake the curated clips' scorecards into the image, or build-on-first-hit to
  `/tmp` and cache in the assets bucket like the PNGs.
- **Local test before deploy:** `./.venv/Scripts/python.exe deploy/test_chat_handler.py` (offline),
  then a live smoke with a key set: `POST` a couple of questions incl. a progression one
  (`compare_clip_id`) and an unmeasured one; confirm `grounded:true` and a sensible refusal.

## 7. Cheaper fallback (presentation safety net)
If the live Lambda path is flaky during a demo: run the pipeline locally for the curated clips, push
the resulting JSON + PNG + explanation to S3 once, and serve them statically via CloudFront. Zero
live compute (~$0.50/mo). Not "live in the cloud," but it never fails on stage. Keep this in your
back pocket; the assets bucket already holds the cached outputs.

---

## 7b. Security & cost-abuse exposure (read before choosing Function URL auth)
A Function URL that triggers paid Claude calls is, in principle, a cost-DoS surface. For this app it
is **largely self-limiting by design** — but only if you keep two controls:
- **Spend is architecturally bounded.** The handler serves only `ALLOWED_CLIPS` and caches each
  clip's explanation in S3 (Phase F). So the *total* number of Anthropic calls this endpoint can ever
  make is **N (curated clips), once each** — every subsequent hit is a cache read at $0. A flood of
  requests can't run up a token bill. **This only holds if both the allow-list and the S3 cache are
  in place** — if you skip the cache, every request re-calls Claude. Don't skip it.
- **Cap compute too.** Set **reserved concurrency** (e.g. 2–5) on the function — it's free and puts a
  hard ceiling on simultaneous executions, so a burst can't run up Lambda/compute cost or exhaust the
  account's 1000-concurrency pool. Excess requests are throttled (429), which is the desired behavior
  for a demo.
- **Auth choice (ties to §9):** for an internal capstone demo, `AuthType: NONE` + the allow-list +
  cache + reserved concurrency is acceptable. If you want more: `AuthType: AWS_IAM` (teammates sign
  requests — most secure, but the static UI then needs SigV4), or a shared secret header the handler
  checks, or AWS WAF rate-based rules in front. Don't add a login system for a fixed-clip demo.
- **Always on:** Budgets alert (§Phase F), Anthropic console hard cap, CloudTrail. The credits cover
  AWS; the Anthropic cap is the backstop on token spend.

## 7c. Local testing, observability & demo-day runbook

**Local container test (do this before every deploy).** The AWS Lambda base image bundles the RIE, so
no extra setup — just Docker running:
```bash
sam build
# event.json mirrors a Function URL request:
#   {"rawPath":"/","queryStringParameters":{"clip_id":"1292"},"requestContext":{"http":{"method":"GET"}}}
sam local invoke MotionCaddieFunction -e event.json --env-vars env.json
# env.json: {"MotionCaddieFunction": {"ANTHROPIC_API_KEY": "sk-ant-..."}}  (gitignore this)
```
Confirm: 200, non-empty `explanation`, `grounding.grounded == true`, a decodable PNG/URL. Test a
clip **not** in `ALLOWED_CLIPS` → expect 400. (Alternatively run the image directly and `curl` the
RIE at `localhost:9000/2015-03-31/functions/function/invocations`.)

**Observability.**
- CloudWatch Logs are automatic. On any Anthropic failure, log the SDK's `response._request_id` so
  issues are traceable; **fail the request if the explanation is empty** (don't silently return a
  textless scorecard — see §4).
- Log one line per request with `clip_id` + `cache_hit` (S3 hit vs Anthropic call) so you can see the
  spend pattern.
- A CloudWatch **alarm on the function's `Errors` metric** (≥1 in 5 min) emails the team. X-Ray is
  optional and overkill here.

**Demo-day runbook (paste into the repo).**
1. Confirm the Anthropic key isn't expired and the console cap isn't already hit.
2. **Warm the function** ~1 min before presenting: `curl "$URL?clip_id=<known-good>"` once (or set
   provisioned concurrency=1 for the session) so the first live click isn't a 10–25 s cold start.
3. Click through 2–3 curated clips; verify PNG + grounded explanation render in the UI.
4. If the live path misbehaves, switch to the **pre-rendered static fallback** (§7) — same clips,
   zero compute.
5. After the demo: remove provisioned concurrency if you set it.

## 8. Definition of done
- [ ] `f_strict_grounding.json` backend flipped to `anthropic`; local `demo.py --golfdb-clip` produces a grounded explanation with **no** Codex CLI present.
- [ ] Container builds, runs under `sam local invoke`, returns scorecard + PNG + non-empty grounded explanation.
- [ ] SAM stack deploys; Function URL returns 200 for every curated clip; secret read only from Secrets Manager (no key in image/repo).
- [ ] Static UI on CloudFront lets you pick a clip and see PNG + explanation.
- [ ] Per-clip explanation caching works (2nd call to same clip makes no Anthropic call).
- [ ] Execution role is least-privilege (one secret, logs, one bucket); Budgets alert set.
- [ ] `FunctionUrlConfig.Cors` set (UI fetches the URL from the browser, not just `curl`); reserved concurrency set as the compute cap (§7b).
- [ ] Runbook + `sam delete` teardown documented; no secrets in git history.

## 9. Open decisions — confirm with the human before deploying
- **Region** — default `us-east-1` (cheapest CloudFront). Keep Identity Center region consistent.
- **Function URL auth** — `NONE` vs `AWS_IAM` vs shared-token. See **§7b** for the full tradeoff;
  for an internal demo, `NONE` + allow-list + S3 cache + reserved concurrency is acceptable. Default
  to that unless the team wants `AWS_IAM`.
- **Who can deploy** — all 3 teammates (Developer/`MotionCaddieDeployer`) or only the owner? (See
  `AWS_HOSTING_PLAN.md` §5.2.)
- **Lambda memory/timeout** — start 2048 MB / 60 s; tune from cold-start + p95 latency.
- **Reserved concurrency** — recommend 2–5 (§7b) as the compute spend cap; confirm it won't starve
  any other function sharing the account's concurrency pool.

## 10. Cost & teardown
- AWS side ≈ cents/mo (S3 + CloudFront + Lambda), covered by the shared **$250 credits**.
- Anthropic ≈ cents/mo (short, cached, capped). Separate bill.
- **Teardown:** `sam delete` removes the stack **and** the SAM-managed ECR companion stack/repo →
  empty + delete the two S3 buckets (non-empty buckets block deletion) → rotate/revoke the Anthropic
  key → (end of capstone) remove Identity Center users.

---

## 11. Landmines (things that will bite you)
1. **No Codex CLI in Lambda** — you MUST flip the feature backend to `anthropic` (§4) or every
   explanation silently no-ops (exit 0, empty text).
2. **`cv2` is pulled transitively** via `eval_utils` — include `opencv-python-headless` or you get
   `ImportError: libGL` / `No module named cv2` at cold start (§3).
3. **CUDA torch is the default in `requirements_eval.txt`** (`torch==2.5.1+cu124`, multi-GB). Install
   the **CPU** wheel from the PyTorch CPU index, or the image balloons and may exceed limits.
4. **numpy 2.x vs torch 2.5.x** — pin `numpy==1.26.4` to avoid ABI breakage.
5. **Read-only filesystem** — anything the scripts write must go to `/tmp`. The default out-dir is
   next to the parquet (read-only in the image) — redirect it.
6. **Gitignored assets** — `Models/` and `Data/eval_runs/` are NOT in git; they're copied from the
   local tree at build time. A clean `git clone` on another machine won't have them — build where the
   files exist, or store them in S3 and fetch at build.
7. **`golfDB.pkl` path is `golfdb/golfDB.pkl`** (lowercase dir at repo root), loaded by `demo.py:52`.
8. **Don't `git add -A`** in this repo — it has dozens of unrelated untracked files.
9. **6 MB response limit** — don't return large base64 payloads; use S3 + presigned/CloudFront URLs
   for the PNG (§Phase A). The platform adds ~1 MB overhead, so your usable body is smaller than 6 MB.
10. **Cold start vs timeout** — set the function timeout *above* the worst cold start (10–25 s) or the
    first request of the day 500s. Pre-warm or use provisioned concurrency for live demos.
11. **CORS** — the CloudFront-hosted UI → Function URL is cross-origin. Set `FunctionUrlConfig.Cors`
    or the browser silently blocks every fetch (works in `curl`, fails in the page). Easy to miss
    because the backend smoke test (`curl`) passes.
12. **ECR companion stack** — the SAM CLI manages the image repo in a *second* CloudFormation stack
    (e.g. `<stack>-...`/`aws-sam-cli-managed-...`). The deployer's `cloudformation:*` is scoped to
    `motion-caddie*`; if the companion stack name falls outside that prefix, widen the policy or run
    the first `--guided` deploy as Admin, then narrow.
