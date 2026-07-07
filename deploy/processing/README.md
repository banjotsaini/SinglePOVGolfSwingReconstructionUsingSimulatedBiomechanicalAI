# Processing worker — SQS-triggered pipeline container

The backend that turns an uploaded video into results. `S3 upload →
EventBridge → SQS → this Lambda`. Runs the same 3-step pipeline as the local demo
(2D→3D→overlay/replay, then event detection + scorecard, then grounded Claude eval),
uploads artifacts to `03_outputs/<job_id>/`, and writes queryable rows via the RDS
Data API.

## Why a container (not a zip)
torch + mediapipe + the ~470 MB model bundle blow past Lambda's 250 MB zip limit.
This is the **only** deploy piece that genuinely needs the container build — the chat
and upload Lambdas ship as zips.

## Status
- **Handler** (`processing_handler.py`): written; the testable logic (SQS/S3 event
  parsing, job-id extraction, idempotent uuid5 ids, DB-write column mapping against
  the live schema) is unit-verified. The pipeline invocation + artifact upload run
  only inside the built container.
- **Container** (`Dockerfile`, `requirements.txt`, `buildspec.yml`): authored, **not
  yet built** — no Docker/CodeBuild available locally. First real build will likely
  need iteration on mediapipe/opencv native libs and torch/mediapipe version pins.

## Build & wire (once CodeBuild perms land — see PENDING_PERMISSIONS.md)
```bash
# 1. build+push via CodeBuild (models staged from S3 by buildspec)
#    project env: ECR_REPO=<acct>.dkr.ecr.us-east-1.amazonaws.com/motion-caddie/processing
#                 IMAGE_TAG=v1  MODELS_S3=s3://motioncaddie-capstone-data-lj-2026/01_inputs/Models
# 2. flip the gated Lambda on by setting the image (single param update, no template change):
aws cloudformation deploy --template-file deploy/infra/full_app.yaml \
  --stack-name motion-caddie-app --capabilities CAPABILITY_IAM \
  --parameter-overrides ProcessingImageUri=<acct>.dkr.ecr.us-east-1.amazonaws.com/motion-caddie/processing:v1 \
  --profile capstone
```
That last step un-gates `ProcessingFunction` + its SQS event source (the
`HasProcessingImage` condition in `full_app.yaml`) and the upload→results path goes live.

## Env (set by full_app.yaml on the function)
`ARTIFACTS_BUCKET`, `DB_CLUSTER_ARN`, `DB_SECRET_ARN`, `APP_ROOT=/var/task`,
`ANTHROPIC_API_KEY` (for the eval step), `POSE_BACKBONE`, `LIFTER`.
