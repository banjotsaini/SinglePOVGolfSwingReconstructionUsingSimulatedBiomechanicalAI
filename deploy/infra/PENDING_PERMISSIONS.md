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

### 1. Account guardrail (SCP) — blocks public Lambda Function URLs  ⚠️ NOT an IAM group perm
The chat Lambda (`motion-caddie-chat`) is deployed and **works** (verified by direct
`aws lambda invoke` — guardrails return 400, code path runs). But its public Function
URL returns **403 Forbidden** even with a correct resource policy
(`Principal:*`, `lambda:InvokeFunctionUrl`, condition `FunctionUrlAuthType=NONE`).
That signature = an **Organizations SCP** denying unauthenticated Function URLs.
- **This can't be fixed with `put-group-policy`** — it's an org-level control.
- **Options for Lawrence / org admin:** (a) relax the SCP to allow `AuthType=NONE`
  Function URLs for `motion-caddie-*`; or (b) we front the Lambdas with API Gateway
  or CloudFront (may face the same guardrail); or (c) switch the URL to
  `AuthType=AWS_IAM` + a signing proxy (breaks simple browser calls — worst for a demo).
- **Recommended:** confirm whether public Function URLs are org-blocked; if so,
  decide (a) vs (b) before wiring the front end.

### 2. `ANTHROPIC_API_KEY` — a secret, not a permission
Live chat answers need the key. Deployed Lambda reads env `ANTHROPIC_API_KEY`; its
exec role can already read a Secrets Manager secret named `motion-caddie/anthropic-*`.
Provide the key (I'll store it in Secrets Manager + wire it), or set it as a Lambda env var.

### IAM group actions still missing
| Action(s) | Needed for | In deploy-policy.json? |
|---|---|---|
| `codebuild:*` (motion-caddie-*) | build the heavy processing container | ✅ (prior pass) |
| _(none new this pass — the light Lambdas deploy as zips with existing perms)_ | | |

## Already granted (for reference)
CloudFormation, EC2/VPC, IAM (roles), RDS + rds-data, SQS, SNS, Lambda,
EventBridge, Cognito, CloudWatch, Logs, Budgets, App-AutoScaling, SageMaker,
ECR, S3, Secrets Manager.
