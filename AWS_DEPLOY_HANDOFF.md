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
                          └ returns { scorecard JSON, scorecard PNG (base64), explanation }
```

No GPU. No always-on compute. AWS cost ≈ cents/mo (covered by the account's $250 promo credits).
Claude runs on the **direct Anthropic API** (separate cents/mo bill) — **not Bedrock** (the AWS
credits don't cover Bedrock).

**Definition of done** is in §8. Work the phases in §5–§7 in order.

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

The **full** pipeline (raw video → 2D MediaPipe → 3D MotionBERT → smoothing → events → scorecard)
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
1. Reads cached 3D parquet: `Data/eval_runs/motionbert_full_from_mediapipe_lite/<id>.parquet`
2. Reads clip metadata (player/club/view) from `golfdb/golfDB.pkl`
3. `scorecard_step.py` → event detector + `build_scorecard` + `render_scorecard` → JSON + PNG
4. `coaching_explain.py` → LLM explanation written back into the scorecard JSON

**Files/data the container must contain** (all verified to exist locally):

| Artifact | Local path | Size | Notes |
|---|---|---|---|
| Event-detector weight | `Models/event_detector_tcn.pt` | ~1.1 MB | `event_detector.py:22` `DEFAULT_WEIGHTS` |
| 3D pose cache | `Data/eval_runs/motionbert_full_from_mediapipe_lite/<id>.parquet` | 131 MB all clips | **bake only the curated subset** (§6) |
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

CACHE = ROOT / "Data" / "eval_runs" / "motionbert_full_from_mediapipe_lite"
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
        "png_base64": base64.b64encode(png_bytes).decode(),
    })

def _resp(code, body):
    return {"statusCode": code,
            "headers": {"content-type": "application/json",
                        "access-control-allow-origin": "*"},
            "body": json.dumps(body)}
```

Implementation notes:
- **Write to `/tmp` only** — Lambda's filesystem is read-only except `/tmp` (512 MB default; bump
  if needed). The scripts write the scorecard JSON/PNG next to the parquet by default; point them at
  `/tmp` (pass an out-dir, or copy logic from `scorecard_step.py`).
- **Set the Anthropic key into the env before the import that needs it.** The handler reads the
  secret (via the SAM-injected env var or the Secrets Manager extension) and ensures
  `os.environ["ANTHROPIC_API_KEY"]` is set before `coaching_explain.explain` runs.
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
COPY deploy/clips/                    ${LAMBDA_TASK_ROOT}/Data/eval_runs/motionbert_full_from_mediapipe_lite/
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
- `AWS::Serverless::Function` — `PackageType: Image`, the Dockerfile above, **Function URL** enabled
  (`AuthType` per §9 decision), memory ≈ 2048–3008 MB (torch needs headroom; more memory = more CPU =
  faster cold start), timeout ≈ 60–120 s, ephemeral `/tmp` storage bumped if needed.
- `AWS::SecretsManager::Secret` — `motion-caddie/anthropic-api-key` (value set out-of-band by human).
- **Execution role** — least privilege: read **only** that one secret, write CloudWatch Logs,
  read/write the explanation-cache S3 bucket. Nothing else.
- `AWS::S3::Bucket` ×2 — `motion-caddie-demo-web` (static UI) and `motion-caddie-demo-assets`
  (per-clip explanation cache + optional pre-rendered fallback).
- `AWS::CloudFront::Distribution` — in front of the web bucket (OAC, not public bucket).
- Inject the secret into the function (env var from Secrets Manager, or the AWS Parameters/Secrets
  Lambda extension). `ALLOWED_CLIPS` env = the curated id list.
- Build/deploy: `sam build && sam deploy --guided` (then non-guided with the saved `samconfig.toml`).

### Phase E — Deploy + smoke test
```bash
sam build
sam deploy --guided          # first time; pick region, stack name "motion-caddie"
curl "$(FUNCTION_URL)?clip_id=1292"   # expect JSON with scorecard + explanation + png_base64
```
Verify: 200, non-empty `explanation`, `grounding.grounded == true`, decode `png_base64` → valid PNG.

### Phase F — Frontend + polish
- Minimal static UI (`deploy/web/index.html` + a little JS): clip-picker dropdown (the curated ids)
  → fetch the Function URL → show the PNG and the explanation text. Upload to the web bucket; serve
  via CloudFront.
- **Cache explanations per clip in S3** (`motion-caddie-demo-assets`): on a cache hit, skip the
  Anthropic call entirely → repeat demos cost $0 in tokens and are instant.
- **AWS Budgets** alert (~$5–10/mo) to the team; short **runbook** + **teardown** steps (§10).

---

## 7. Cheaper fallback (presentation safety net)
If the live Lambda path is flaky during a demo: run the pipeline locally for the curated clips, push
the resulting JSON + PNG + explanation to S3 once, and serve them statically via CloudFront. Zero
live compute (~$0.50/mo). Not "live in the cloud," but it never fails on stage. Keep this in your
back pocket; the assets bucket already holds the cached outputs.

---

## 8. Definition of done
- [ ] `f_strict_grounding.json` backend flipped to `anthropic`; local `demo.py --golfdb-clip` produces a grounded explanation with **no** Codex CLI present.
- [ ] Container builds, runs under `sam local invoke`, returns scorecard + PNG + non-empty grounded explanation.
- [ ] SAM stack deploys; Function URL returns 200 for every curated clip; secret read only from Secrets Manager (no key in image/repo).
- [ ] Static UI on CloudFront lets you pick a clip and see PNG + explanation.
- [ ] Per-clip explanation caching works (2nd call to same clip makes no Anthropic call).
- [ ] Execution role is least-privilege (one secret, logs, one bucket); Budgets alert set.
- [ ] Runbook + `sam delete` teardown documented; no secrets in git history.

## 9. Open decisions — confirm with the human before deploying
- **Region** — default `us-east-1` (cheapest CloudFront). Keep Identity Center region consistent.
- **Function URL auth** — `AuthType: NONE` + a shared token check (simplest for an internal demo) vs
  `AWS_IAM` (teammates sign requests). Default to a shared token; it's internal-only.
- **Who can deploy** — all 3 teammates (Developer/`MotionCaddieDeployer`) or only the owner? (See
  `AWS_HOSTING_PLAN.md` §5.2.)
- **Lambda memory/timeout** — start 2048 MB / 60 s; tune from cold-start + p95 latency.

## 10. Cost & teardown
- AWS side ≈ cents/mo (S3 + CloudFront + Lambda), covered by the shared **$250 credits**.
- Anthropic ≈ cents/mo (short, cached, capped). Separate bill.
- **Teardown:** `sam delete` the stack → empty + delete the two S3 buckets → delete the ECR repo →
  rotate/revoke the Anthropic key → (end of capstone) remove Identity Center users.

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
