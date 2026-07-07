/* MotionCaddie front-end backend config — deployed endpoint wiring.
 *
 * Load this BEFORE chat.js / app.js (e.g. <script src="config.js"></script> first).
 * Leave the URLs blank to run the offline/mock preview; set them to go live.
 *
 * NOTE (routing): our backends are separate Lambda **Function URLs**, one per
 * function — not one API_BASE with path routing. chat.js appends `/chat` to
 * API_BASE; a Function URL catches all paths and ignores them, so pointing
 * API_BASE at the chat function works for chat. Uploads use their own URL.
 * If we later want a single origin (one domain, real paths), front both Lambdas
 * with API Gateway or CloudFront and set API_BASE to that instead.
 *
 * ⚠️ BLOCKER: these public Function URLs currently return 403 — an account SCP
 * blocks unauthenticated Function URLs. Until that's resolved (org admin) or the
 * Lambdas are fronted by API Gateway/CloudFront, live mode won't reach them.
 * See deploy/infra/PENDING_PERMISSIONS.md. The Lambdas themselves are verified
 * working via direct invoke.
 */
(function () {
  // chat backend (motion-caddie-chat). Trailing slash trimmed so `${API_BASE}/chat` is clean.
  window.API_BASE = "https://h4uwq7gcuyg3ayfo45q7zsbwny0pagdu.lambda-url.us-east-1.on.aws".replace(/\/$/, "");

  // upload-URL issuer (motion-caddie-upload-url) — POST {filename, content_type}
  window.UPLOAD_URL = "https://ksodla6bakjuis5hfl7wia7lby0xzupx.lambda-url.us-east-1.on.aws/";

  // demo clip whitelist the chat backend accepts (ALLOWED_CLIPS on the Lambda)
  window.DEMO_CLIPS = [0, 2, 4, 6, 8, 10, 1292];
})();
