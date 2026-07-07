# Pending permissions — consolidated ask for Lawrence

Living list of IAM actions the `MotionCaddie-Deploy` group is still missing, found
while building. **Batch these into one `put-group-policy` update** rather than
running the command repeatedly. Most are already folded into
`deploy/infra/deploy-policy.json`; this file is the human-readable "why".

To apply everything at once (from repo root, as account owner):
```bash
git pull
aws iam put-group-policy --group-name MotionCaddie-Deploy \
  --policy-name motion-caddie-deploy --policy-document file://deploy/infra/deploy-policy.json
```

## ✅ CHAT IS LIVE — Bedrock (bedrock-runtime) + Claude Haiku 4.5, no API key

`motion-caddie-chat` runs with `CHAT_BACKEND_PROVIDER=bedrock-runtime` +
`BEDROCK_MODEL=us.anthropic.claude-haiku-4-5-20251001-v1:0`. Verified end-to-end in
the hosted UI: live grounded answers with tool-call traces. Entitlement findings:
- **Classic bedrock-runtime surface: Haiku 4.5 enabled** (Lawrence was right).
- **Mantle (Messages-API) endpoint: separate entitlement, NOT enabled** — all models
  return "not available for this account" there.
- Exec role has both surfaces' IAM (`bedrock:InvokeModel*` on models/profiles,
  `bedrock-mantle:CreateInference`) — switching later is env-vars only.

### Optional upgrades (not blockers)
| Item | Why | Who |
|---|---|---|
| Enable more Claude models on bedrock-runtime (Sonnet/Opus) | better coaching answers than Haiku | Lawrence (Bedrock console → Model access) |
| `ANTHROPIC_API_KEY` (direct API, `CHAT_BACKEND_PROVIDER=anthropic`) | claude-opus-4-8 quality, no Bedrock dependency | Banjot |
| Lambda quota raises: function memory → 10240 MB; account concurrency → 1000 | processing runs ~3× faster; restores the reserved-concurrency spend cap | Lawrence (Service Quotas → Lambda) |

## Resolved this pass ✅
- **SCP blocking public Function URLs** → worked around, no org action needed:
  both Lambdas now sit behind API Gateway HTTP API `motion-caddie-api`
  (`https://bk7s56lvq3.execute-api.us-east-1.amazonaws.com` — POST /chat, /upload-url).
  Function URLs deleted.
- **`codebuild:*`** → granted; both build projects created
  (`motion-caddie-chat-image` SUCCEEDED → `motion-caddie/chat:v1` in ECR;
  `motion-caddie-processing-image` iterating).
- **CloudFront + web bucket** → granted; `motion-caddie-web` stack deployed
  (teammate's `web_hosting.yaml`), site synced and serving at
  **https://d3oak5k3h8fdvi.cloudfront.net**.

## Already granted (for reference)
CloudFormation, EC2/VPC, IAM (roles), RDS + rds-data, SQS, SNS, Lambda,
EventBridge, Cognito, CloudWatch, Logs, Budgets, App-AutoScaling, SageMaker,
ECR, S3, Secrets Manager.
