# Upload-URL issuer Lambda

Issues short-lived **presigned S3 POST** URLs so the browser uploads swing videos
straight to S3 (never through compute). Presigned POST enforces content-type + a
hard size cap (`MAX_UPLOAD_MB`, default 200) as S3 conditions the client can't override.

## Deployed
- Function: `motion-caddie-upload-url` (python3.12, zip — boto3 is in the runtime, no deps)
- Role: `motion-caddie-app-upload-exec` (logs + `s3:PutObject` on the uploads prefix only)
- Env: `UPLOADS_BUCKET`, `UPLOAD_PREFIX=01_inputs/uploads`, `MAX_UPLOAD_MB=200`, `URL_TTL_SECONDS=300`

**Verified end-to-end**: issuer → presigned POST → browser-style multipart upload to
S3 (HTTP 204) → object lands at `01_inputs/uploads/<job_id>/<file>`. That PUT is what
triggers the EventBridge→SQS→processing path in `full_app.yaml`.

## Contract
`POST {filename, content_type}` →
`{ job_id, object_key, url, fields, max_mb, expires_in, result_prefix }`
Client does a multipart POST of `fields` + the file to `url`; then polls
`result_prefix` (`03_outputs/<job_id>/`) for pipeline output.

## Redeploy
```bash
cd deploy/upload && zip -r ../../build/upload-lambda.zip upload_handler.py
aws lambda update-function-code --function-name motion-caddie-upload-url \
  --zip-file fileb://build/upload-lambda.zip --profile capstone
```

## Note
The issuer's public Function URL is subject to the same account SCP blocking
unauthenticated Function URLs (see `deploy/infra/PENDING_PERMISSIONS.md`). The S3
upload itself is direct-to-bucket and unaffected.
