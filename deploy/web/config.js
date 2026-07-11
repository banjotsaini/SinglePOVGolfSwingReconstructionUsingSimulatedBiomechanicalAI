/* MotionCaddie front-end backend config — deployed endpoint wiring.
 *
 * Load this BEFORE chat.js / app.js (e.g. <script src="config.js"></script> first).
 * Leave the URLs blank to run the offline/mock preview; set them to go live.
 *
 * Both backends sit behind ONE API Gateway HTTP API (motion-caddie-api,
 * bk7s56lvq3) — single origin, real paths, CORS handled at the gateway:
 *   POST /chat        -> motion-caddie-chat        (grounded coaching chatbot)
 *   POST /upload-url  -> motion-caddie-upload-url  (presigned S3 POST issuer)
 * chat.js appends `/chat` to API_BASE, which matches this routing exactly.
 *
 * (History: Lambda Function URLs were tried first but an account SCP blocks
 * unauthenticated Function URLs — API Gateway is the sanctioned front door.
 * Verified publicly: /chat guardrail 400s, /upload-url issues a real presign,
 * OPTIONS preflight 204 with ACAO *. Chat answers go live once
 * ANTHROPIC_API_KEY is set on the chat Lambda.)
 */
(function () {
  window.API_BASE = "https://bk7s56lvq3.execute-api.us-east-1.amazonaws.com";

  // upload-URL issuer — POST {filename, content_type} -> {url, fields, job_id, ...}
  window.UPLOAD_URL = window.API_BASE + "/upload-url";

  // demo clip whitelist the chat backend accepts (ALLOWED_CLIPS on the Lambda)
  window.DEMO_CLIPS = [417, 886, 830, 269, 0];

  // Where processed upload results are served from (the pipeline writes
  // 03_outputs/<job_id>/{metrics,explanation,replay_3d}.json + videos in the
  // same shape as assets/<clip>/). Leave "" until that prefix is exposed
  // read-only (CloudFront behavior or presigned GETs) — uploads then land in
  // S3 and the pipeline runs, but the UI falls back to the illustrative
  // result with the honest banner instead of polling.
  window.RESULTS_BASE = "";
})();
