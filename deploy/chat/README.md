# Chat Lambda container — coaching chatbot `/chat` backend

Packages [`deploy/chat_handler.py`](../chat_handler.py) (grounded coaching chatbot,
one HTTP request = one chat turn) as a **Lambda container image**. CPU-only, no
GPU, no ML stack — the sole third-party dep is the Anthropic SDK.

Validated offline against the exact image layout (`APP_ROOT=/var/task`): the import
graph resolves, a scripted grounded turn reads a real cached indicator, the HTTP
handler returns 200, and the `ALLOWED_CLIPS` guardrail rejects out-of-set clips.

## Build & push (from repo root)
```bash
ACCT=084375574654 REGION=us-east-1
REPO=$ACCT.dkr.ecr.$REGION.amazonaws.com/motion-caddie/chat   # reuse the ECR repo, chat tag

aws ecr get-login-password --region $REGION --profile capstone \
  | docker login --username AWS --password-stdin $ACCT.dkr.ecr.$REGION.amazonaws.com

docker build -f deploy/chat/Dockerfile -t $REPO:latest .      # context = repo root
docker push $REPO:latest
```
(No local Docker on the build box yet — build via Docker Desktop or an ECR/CodeBuild
job. The `motion-caddie/processing` ECR repo already exists; this reuses it with a
`chat` image name.)

## Deploy as a Lambda Function URL
The chat backend is a **separate** Lambda from the SQS `ProcessingFunction` in
`full_app.yaml` (not yet a resource there — see gap note below). Simplest path for
the demo:
```bash
aws lambda create-function --function-name motion-caddie-chat \
  --package-type Image --code ImageUri=$REPO:latest \
  --role <lambda-exec-role-arn> --timeout 60 --memory-size 1024 \
  --environment "Variables={ALLOWED_CLIPS=[0,1292],CHAT_MAX_QUESTION_CHARS=500}" \
  --profile capstone --region $REGION
# ANTHROPIC_API_KEY -> set from Secrets Manager, never in the image or in git.
aws lambda create-function-url-config --function-name motion-caddie-chat \
  --auth-type NONE --cors '{"AllowOrigins":["*"],"AllowMethods":["POST"]}' --profile capstone
```
Point the front end's `loadClipBundle()` / chat call at the returned Function URL.

## Runtime env vars (all optional, safe defaults)
| Var | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | **required in prod**; injected from Secrets Manager |
| `ALLOWED_CLIPS` | `[]` (all) | whitelist of demo clip ids |
| `CHAT_MAX_QUESTION_CHARS` | `500` | per-question cap |
| `CHAT_MAX_HISTORY_MSGS` | `20` | ~10 prior turns kept |
| `CHAT_MAX_BODY_BYTES` | `32768` | request-size cap |

## Open gap
The chat Lambda + its Function URL + exec role are **not yet in `full_app.yaml`**.
For a repeatable deploy they should be added as a `AWS::Lambda::Function`
(`PackageType: Image`) + `AWS::Lambda::Url` + a least-priv role that can read the
`ANTHROPIC_API_KEY` secret. Standalone `create-function` above is fine for the demo.
