# Deploy permissions — least-privilege, not AdministratorAccess

`full_app.yaml` needs to create IAM roles + a VPC + Aurora, so the deployer needs
more than the data-plane `banjot-saini` user has. **You do not need to grant
`AdministratorAccess`.** Two scoped options, cleanest first.

## Exact services the stack touches
| Service | Why | Needed for first deploy? |
|---|---|---|
| CloudFormation | runs the stack | yes |
| EC2 (VPC/Subnet/SecurityGroup) | minimal private VPC for Aurora — **no NAT, no IGW** | yes |
| RDS | Aurora Serverless v2 (scale-to-zero) | yes |
| SQS, SNS, EventBridge | async ingest + alerts | yes |
| Lambda | processing worker | yes |
| CloudWatch, Logs, Budgets | alarms + $40/mo budget guard | yes |
| S3 | `motion-caddie-uploads-*`, `motion-caddie-artifacts-*` (scoped by name) | yes |
| IAM | 2 stack roles, name-scoped to `motion-caddie-app-*` | yes |
| **Cognito** | user login | **no — deferrable, see below** |
| **SageMaker** | optional GPU worker, `EnableGpuWorker=false` by default | **no — off by default** |

Every resource is region-locked to `us-east-1` and name-prefixed `motion-caddie`.

## Option A (recommended) — CloudFormation service role
You (as account owner) create **one role** that only CloudFormation can assume,
carrying `deploy-policy.json`. Then grant the deployer just
`cloudformation:*` on the `motion-caddie-app` stack + `iam:PassRole` on that one
role. The powerful permissions never live on a user's long-term keys.

```bash
# you run once (as owner):
aws iam create-role --role-name motion-caddie-cfn-deploy \
  --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"cloudformation.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
aws iam put-role-policy --role-name motion-caddie-cfn-deploy \
  --policy-name motion-caddie-deploy --policy-document file://deploy/infra/deploy-policy.json

# deployer then runs, passing the role so CFN — not the user — holds the power:
aws cloudformation deploy --template-file deploy/infra/full_app.yaml \
  --stack-name motion-caddie-app --capabilities CAPABILITY_IAM \
  --role-arn arn:aws:iam::<acct>:role/motion-caddie-cfn-deploy \
  --parameter-overrides AlertEmail=you@example.com ProcessingImageUri=<ecr-uri>
```

## Option B — attach the scoped policy to a deploy user
Attach `deploy-policy.json` directly to a `capstone-deploy` user (or to
`banjot-saini`). Simpler, but the permissions sit on a user's keys. Same policy
file either way.

## Deferring Cognito to week 12
The **demo path (pick a pre-rendered clip) needs no login at all** — Cognito only
gates the real user-upload feature. To ship the deploy now without auth, delete the
`UserPool` + `UserPoolClient` resources (and their outputs) from `full_app.yaml`,
or leave them — they cost ~$0 idle and add no attack surface for a demo. Either is
fine; dropping them removes the `cognito-idp:*` block from the policy above.
