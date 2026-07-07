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

## Still needed (as of this build pass)

### `ANTHROPIC_API_KEY` — a secret, not a permission (THE one remaining item)
Live chat answers need the key (the chatbot calls Claude via the direct Anthropic
API — Bedrock is excluded, credits don't cover it). The deployed Lambda reads env
`ANTHROPIC_API_KEY`; its exec role can already read a Secrets Manager secret named
`motion-caddie/anthropic-*`. Provide the key, or set it on `motion-caddie-chat`
(Console → Configuration → Environment variables).

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
