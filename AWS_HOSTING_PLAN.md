# Motion Caddie — AWS Hosting Plan (Capstone Demo)

**Scope decided with stakeholder (2026-06-26):**
- **Goal:** shareable *demo* for the capstone — fixed set of clips, not arbitrary public uploads.
- **Audience:** just me / the team (internal, intermittent use).
- **LLM backend:** **Claude via the direct Anthropic API** (`claude-opus-4-8`), switching off the
  local Codex CLI. Billed separately to Anthropic (cents/mo, cached per clip); keeps the
  validated grounding eval valid.
- **Bedrock is ruled out:** the account's **$250 AWS promotional credits exclude Bedrock**
  (third-party model usage isn't credit-eligible). Credits fund the hosting instead.
- **Cost posture:** minimize cost — CPU/serverless, scale-to-zero, **no always-on GPU**. The
  $250 AWS credits cover Lambda/S3/CloudFront/ECR (nearly free at this scale → lasts ~forever).
- **IaC:** AWS SAM. **AWS account exists; CLI/credentials not yet configured locally** (Phase 0).
- **Team (2026-06-30):** **4 people total — you (account owner) + 3 teammates** get AWS access for
  the capstone. Access is granted via **IAM Identity Center** (short-lived SSO creds, no shared
  keys), with least-privilege permission sets. Full design in **§5 (Team access)**.

---

## 1. Key finding — the demo path is lightweight

The demo runs the **cached-clip path** (`demo.py --golfdb-clip <id>`), which skips the GPU
stages entirely (2D MediaPipe → 3D MotionBERT are pre-computed). What it actually needs:

| Need | Footprint | Notes |
|---|---|---|
| `event_detector_tcn.pt` | **1.1 MB** | the only model weight required |
| Cached 3D parquets | **131 MB** (all 1400) / a few MB curated | `Data/eval_runs/motionbert_full_from_mediapipe_lite/` |
| `golfDB.pkl` | small | clip metadata (player/club/view) |
| Python deps | torch-CPU, numpy, pandas, pyarrow, matplotlib, anthropic | **no** mediapipe / tensorflow / ultralytics / golfpose .bin |

So the runtime container is modest (~1–1.5 GB with CPU-only torch) and needs **no GPU**.

**Runtime flow per request:** `clip_id` → load cached parquet → `event_detector` (CNN, CPU) →
`build_scorecard` → `render_scorecard` (PNG) → `coaching_explain` (Anthropic API) → return
scorecard JSON + PNG + grounded explanation.

---

## 2. Recommended architecture — serverless, scale-to-zero

```
 Browser ──HTTPS──> CloudFront ──> S3 (static demo UI: index.html + JS)
    │
    └── pick clip id ──> Lambda Function URL ──> Lambda (container image)
                                                   ├─ torch-CPU + scripts + weights + clips (baked in)
                                                   ├─ Claude via Anthropic API  (key from Secrets Manager)
                                                   └─ returns { scorecard.json, scorecard.png(base64), explanation }
```

**Why this shape (for demo / team / minimize-cost):**
- **Lambda container image** — torch-CPU + curated clips fit well under the 10 GB image limit
  (the 250 MB zip limit rules out a plain zip Lambda, so we use a container image).
- **Lambda Function URL** instead of API Gateway — one fewer service to manage and pay for;
  fine for a low-traffic internal demo. (Swap in API Gateway later if we need auth/throttling.)
- **Scale-to-zero** — $0 when idle; pay only per demo click. No GPU, no idle EC2/Fargate.
- **Claude via Anthropic API** — `ANTHROPIC_API_KEY` in **Secrets Manager**, read by the Lambda
  execution role. Billed to Anthropic (separate from AWS), not the credits — but the bill is
  cents/mo for short, cached explanations. Bedrock was the AWS-native route but is **credit-
  ineligible**, so the direct API is the pragmatic choice.

### Cheaper fallback — fully pre-rendered static (≈$0.50/mo, zero compute)
Run the pipeline locally for the curated clip set, produce JSON + PNG + explanation once,
upload to S3, serve via CloudFront. No Lambda, no live compute — but not "live in the cloud."
Good fallback / safety net if the live path has issues during a presentation.

### Explicitly out of scope (future)
Raw-video upload (the full GPU pipeline: MediaPipe → MotionBERT → golfpose). That needs a
GPU and would run as an async **Fargate/Batch** job triggered from S3 upload — deferred until
there's a reason to demo arbitrary videos.

---

## 3. Moving the project to AWS — data & code

- **Code:** the repo is light (weights + `Data/eval_runs/` are gitignored). The container build
  pulls the curated clips + `event_detector_tcn.pt` in at build time.
- **Curate clips:** pick ~10–25 representative GolfDB clip ids for the demo (e.g. varied
  player/club/view). Bake just those parquets into the image to keep it small, or store all
  131 MB in S3 and fetch on cold start.
- **S3 buckets:** `motion-caddie-demo-web` (static UI) + optional `motion-caddie-demo-assets`
  (clips, pre-rendered fallback outputs).
- **LLM switch (minimal — backend already exists):** `coaching_llm_summary_v2.py` already has an
  `anthropic` backend (`_run_anthropic`, model `claude-opus-4-8`, with prompt caching). Point
  `coaching_explain` at it (set the `f_strict_grounding` feature's backend to `anthropic`) and
  read the key from the env var the Lambda injects from Secrets Manager. No new client code —
  same model family, so the validated grounding holds.

---

## 4. Security & cost controls

- `ANTHROPIC_API_KEY` in **Secrets Manager**; Lambda execution role scoped to read only that
  secret. Never bake the key into the image or commit it.
- Function URL: restrict with a simple shared token or IAM auth (it's internal-only).
- **Anthropic token spend** (separate from AWS credits) is the main variable cost — explanations
  are short and grounded, so per-call cost is small. Cache explanations per clip in S3 so repeat
  demos cost $0 in tokens. Set a usage limit in the Anthropic console as a hard cap.
- **AWS Budgets** alert at a low threshold (e.g. $5/mo) as a backstop on the AWS side; the $250
  credits should comfortably cover all hosting for the capstone.

**Rough monthly cost:** AWS side ≈ a few cents (S3 + CloudFront + Lambda), **covered by the $250
credits**. Separate Anthropic token bill = cents/mo (cap + cache it). Realistically **< $5/mo**
total, and effectively ~$0 out of pocket on AWS while credits last.

---

## 5. Team access — roles, permissions & onboarding (you + 3 teammates)

**Decision:** use **AWS IAM Identity Center** (formerly AWS SSO) — not bare IAM users — for all 4
people. It is the current AWS-recommended way for human users to access an account, it's **free**
(no extra AWS charge), and it issues **short-lived credentials** instead of long-lived access
keys. When the capstone ends, you delete one user and their access is gone everywhere — nothing to
hunt down.

> Why not plain IAM users? IAM users carry long-term access keys that must be rotated and manually
> revoked per person. AWS best practice is to require human users to assume roles for temporary
> credentials; Identity Center does exactly that. (Fallback in §5.6 if Identity Center setup stalls.)

### 5.1 One-time setup (account owner)
1. **Secure the root user first** (§5.4) — MFA + lock it away before anything else.
2. **Enable IAM Identity Center** in the console. On a standalone account this creates an
   **organization instance** (your account becomes the management account). Pick an **account
   instance only if** you never want cross-account access — but the org instance is required for
   **permission sets / AWS-account access**, so choose the org instance. Set the Identity Center
   region (keep it consistent with the deploy region, default `us-east-1`).
3. Use the **built-in Identity Center directory** as the identity source (no external IdP needed
   for a 4-person team). Create **4 users** (you + 3 teammates) — each gets an email invite to set
   a password and register MFA.
4. Create the **groups** and **permission sets** in §5.2, then assign groups → account → permission set.

### 5.2 Roles (groups → permission sets)

| Group | Members | Permission set | What it can do | MFA |
|---|---|---|---|---|
| **Admins** | you only | `AdministratorAccess` (AWS-managed) | Everything — IAM, Identity Center, root-adjacent tasks. **Break-glass: use sparingly.** | Required |
| **Developers** | all 4 (incl. you) | `MotionCaddieDeployer` (custom, §5.3) **+** `ReadOnlyAccess` | Deploy/update/delete the SAM stack; browse all resources read-only | Required |
| **Billing** | you (+ optional) | `Billing` (AWS-managed) | View costs, budgets, credit balance | Required |

For day-to-day work everyone signs in as **Developer** (deploy + read-only). You keep the separate
**Admin** set for the rare IAM/account chore, so a fat-fingered command during a demo can't nuke
account config. If you'd rather some teammates *not* be able to deploy, drop them to
`ReadOnlyAccess` + `AWSLambda_ReadOnlyAccess` (a "Presenter" tier) — they can still run the demo
via the Function URL, which doesn't need any AWS permission.

### 5.3 `MotionCaddieDeployer` — least-privilege deploy permission set

The secure pattern is **two roles**: the *deployer* (the human) can drive CloudFormation and
`PassRole` a single **CloudFormation execution role**, and that execution role (assumed only by the
CloudFormation service) holds the actual resource-create permissions. This stops a developer from
granting themselves escalated IAM permissions. For a capstone this is optional polish — a single
scoped policy is acceptable — but it's the right shape and not much extra work in SAM.

**Deployer policy (attach to the permission set)** — replace `<ACCT>` / `<REGION>`:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    { "Sid": "Cfn", "Effect": "Allow",
      "Action": ["cloudformation:*"],
      "Resource": ["arn:aws:cloudformation:<REGION>:<ACCT>:stack/motion-caddie*/*"] },
    { "Sid": "SamArtifacts", "Effect": "Allow",
      "Action": ["s3:CreateBucket","s3:PutObject","s3:GetObject","s3:ListBucket","s3:DeleteObject","s3:PutBucketPolicy","s3:GetBucketLocation"],
      "Resource": ["arn:aws:s3:::motion-caddie-*","arn:aws:s3:::motion-caddie-*/*","arn:aws:s3:::aws-sam-cli-*","arn:aws:s3:::aws-sam-cli-*/*"] },
    { "Sid": "Ecr", "Effect": "Allow",
      "Action": ["ecr:GetAuthorizationToken","ecr:BatchCheckLayerAvailability","ecr:PutImage","ecr:InitiateLayerUpload","ecr:UploadLayerPart","ecr:CompleteLayerUpload","ecr:CreateRepository","ecr:DescribeRepositories","ecr:SetRepositoryPolicy","ecr:GetRepositoryPolicy","ecr:BatchGetImage"],
      "Resource": "*" },
    { "Sid": "LambdaCfFront", "Effect": "Allow",
      "Action": ["lambda:*","cloudfront:*","logs:*"],
      "Resource": "*" },
    { "Sid": "SecretsRead", "Effect": "Allow",
      "Action": ["secretsmanager:GetSecretValue","secretsmanager:DescribeSecret","secretsmanager:CreateSecret","secretsmanager:PutSecretValue","secretsmanager:TagResource"],
      "Resource": "arn:aws:secretsmanager:<REGION>:<ACCT>:secret:motion-caddie/*" },
    { "Sid": "PassExecRoleOnly", "Effect": "Allow",
      "Action": "iam:PassRole",
      "Resource": "arn:aws:iam::<ACCT>:role/motion-caddie-*",
      "Condition": { "StringEquals": { "iam:PassedToService": ["lambda.amazonaws.com","cloudformation.amazonaws.com"] } } },
    { "Sid": "RoleMgmtScoped", "Effect": "Allow",
      "Action": ["iam:CreateRole","iam:DeleteRole","iam:AttachRolePolicy","iam:DetachRolePolicy","iam:PutRolePolicy","iam:DeleteRolePolicy","iam:GetRole","iam:TagRole"],
      "Resource": "arn:aws:iam::<ACCT>:role/motion-caddie-*" }
  ]
}
```

Notes: scoped to the `motion-caddie*` stack/bucket/role naming prefix so it can't touch unrelated
resources; `iam:PassRole` is limited to the project's own roles; ECR auth token requires `*`
(AWS limitation). Whenever you grant a `Create*`, the matching `Update*`/`Delete*` are included —
otherwise the **second** deploy or a `sam delete` fails. The `RoleMgmtScoped` block is the
single-policy shortcut; the cleaner two-role version moves resource perms into the execution role
and drops most of `lambda:*`/`cloudfront:*` here.

> **Simpler fallback:** attach the AWS-managed **`PowerUserAccess`** (full access *except* IAM/Org
> management) plus a tiny add-on granting `iam:PassRole`/`CreateRole` **only** on `motion-caddie-*`
> with a permissions boundary. Faster to stand up; less tightly scoped. Fine for a short-lived capstone.

### 5.4 Security baseline (apply on day one)
- **Root user:** enable **MFA** (phishing-resistant passkey/security key preferred; TOTP app is
  fine), remove any root access keys, store the password + recovery in a shared password manager
  the team trusts. **Never use root for daily work** and never share it casually.
- **MFA for everyone:** require MFA on all 4 Identity Center users (enforced in Identity Center settings).
- **No long-lived keys:** teammates authenticate with `aws sso login` (temporary creds). Don't
  create IAM-user access keys unless a tool truly can't do SSO.
- **Least privilege:** start from the scoped sets above; use **IAM Access Analyzer** to tighten
  later if needed.
- **Turn on CloudTrail** (a single-region trail is free-tier-friendly) so every teammate's actions
  are attributable in the shared account.
- **Review at project end:** delete the Identity Center users, `sam delete` the stack, revoke the
  Anthropic key (§5.5).

### 5.5 Cost & secret controls with 4 people in one account
- **Shared bill, no native per-user split.** Tag every stack resource `Project=motion-caddie`
  (cost-allocation tag) and rely on **CloudTrail** for who-did-what. **AWS Budgets** alert at a low
  threshold (e.g. $5–10/mo) emails the whole team if spend spikes.
- **The `$250 credits are shared** across all 4 — one budget, one balance. Hosting is cents/mo, so
  this is comfortable, but a misconfigured always-on resource by any teammate burns the shared pool.
  The Budgets alert is the backstop.
- **Anthropic key is shared and sensitive.** Keep it **only** in Secrets Manager, read at runtime
  by the **Lambda execution role** — teammates do **not** need it on their laptops to deploy. Scope
  the secret's read access to the Lambda role + Admin, not to the Developer set. Set a **hard usage
  cap in the Anthropic console** and **cache explanations per clip in S3** so repeat demos cost $0
  in tokens. The Anthropic bill is separate from AWS credits.

### 5.6 Teammate CLI onboarding (replaces single-user `aws configure`)
Each teammate, once invited:
```bash
aws configure sso          # paste the SSO start URL + region from their invite
# pick the account + the "MotionCaddieDeployer" / Developer role, name the profile e.g. motion-caddie
aws sso login --profile motion-caddie
sam build && sam deploy --profile motion-caddie    # deploy with temporary SSO creds
```
No access keys are ever stored on disk; `aws sso login` refreshes the short-lived session. Everyone
uses the **same SAM stack name** so they're updating one shared deployment, not 4 copies.

### 5.7 Onboarding / offboarding checklist
- **Onboard a teammate:** create Identity Center user → add to **Developers** (and Billing if
  needed) → they accept the email, set password, register MFA → run `aws configure sso`.
- **Offboard / end of capstone:** remove the user from groups (instant revoke of AWS access) →
  delete the user → `sam delete` the stack → delete the ECR repo + S3 buckets → revoke/rotate the
  Anthropic key → disable Identity Center if fully done.

---

## 6. Phased task breakdown

**Phase 0 — AWS + local prep**
1. **Secure root + stand up team access (§5):** enable MFA on root; enable **IAM Identity Center**
   (org instance); create the **Admins / Developers / Billing** groups + the `MotionCaddieDeployer`
   permission set; create the 4 users and invite teammates. Then **install AWS CLI** and run
   `aws configure sso` (no long-lived keys) — each teammate does the same per §5.6.
2. **Get an Anthropic API key** (console.anthropic.com) and set a usage limit; install AWS **SAM CLI**.
3. Point `coaching_explain` at the existing `anthropic` backend; verify the cached-clip demo
   works end-to-end locally with the key; re-confirm `f_strict_grounding` grounds 1.000 on a few clips.
4. Curate the demo clip-id list; confirm each renders a good scorecard + explanation.
5. Write a thin Lambda handler (`app.py`) wrapping the demo flow → returns JSON + base64 PNG.

**Phase 1 — containerize**
6. `Dockerfile` (AWS Lambda Python base image): CPU torch, deps, scripts, weights, clips.
7. Test the container locally with `sam local invoke` / the Lambda Runtime Interface Emulator.

**Phase 2 — deploy backend (SAM)**
8. `template.yaml`: Lambda (container image from ECR) + Function URL + Secrets Manager secret
   (`ANTHROPIC_API_KEY`) + execution role scoped to read that secret. `sam build && sam deploy --guided`.
9. Smoke-test: `curl` the Function URL with a clip id.

**Phase 3 — frontend + polish**
10. Minimal static UI (clip picker → shows PNG + explanation) to S3 + CloudFront (in the SAM stack).
11. Cache explanations per clip in S3; add a Budgets alert; write a short runbook + teardown steps.

**IaC:** AWS SAM throughout — one `template.yaml` for the whole stack, reproducible and
`sam delete`-able (keeps idle cost honest).

---

## 7. Resolved decisions & remaining question
**Resolved:** demo scope · internal/team audience · minimize-cost serverless · **Claude via
direct Anthropic API** (Bedrock excluded — not credit-eligible) · **AWS SAM** · account exists
but **CLI not yet configured** (Phase 0 task 1) · **Anthropic key needed** (Phase 0 task 2) ·
**team access = IAM Identity Center, 4 users, least-privilege permission sets** (§5).

**Remaining:**
- **Region?** Default `us-east-1` (cheapest CloudFront). No Bedrock dependency now, so region
  is just a cost/latency choice — `us-east-1` is fine. Keep the Identity Center region consistent.
- **Teammate deploy rights?** Default: all 4 are **Developers** (can deploy). Downgrade any to a
  read-only **Presenter** tier (§5.2) if you'd rather only you push deploys.
- **Root MFA type?** Passkey/security key is best; a TOTP authenticator app is an acceptable
  capstone-grade choice.
