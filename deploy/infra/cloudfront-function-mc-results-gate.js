/* mc-results-gate — CloudFront Function (viewer-request, cloudfront-js-2.0).
 *
 * DEPLOYED (2026-07-17) on distribution E1OPRZIXL2IZX, associated with the
 * `03_outputs/*` and `audio/*` cache behaviors. Uploaded swings are of real
 * people: their artifacts (metrics/explanation/replay JSON, overlay +
 * pose-debug videos) and job-id narration audio are private during the pilot.
 *
 * Access = the dev access code, presented either as the `mc_dev` cookie
 * (set by dev sign-in on portal.html — auth.js) or as a `?t=` query param
 * (curl / API use). Demo-clip assets (assets/*) and demo narrations
 * (audio/* without a 32-hex job id) stay public.
 *
 * THE REAL TOKEN IS NOT IN GIT. This copy is redacted; the deployed source of
 * truth is the function itself:
 *   aws cloudfront get-function --name mc-results-gate --stage LIVE \
 *     --profile capstone out.js
 * To rotate: edit TOKEN, update-function + publish-function, then hand the
 * new code to the dev account holder(s).
 *
 * `/03_outputs/__access_check__` returns 204 (code valid) / 403 — used by
 * portal.js to validate a sign-in immediately instead of failing per-fetch.
 */
function handler(event) {
  var req = event.request;
  var TOKEN = "<REDACTED — see deployed function>";
  var ok = false;
  var c = req.cookies && req.cookies["mc_dev"];
  if (c && c.value === TOKEN) ok = true;
  var q = req.querystring && req.querystring["t"];
  if (!ok && q && q.value === TOKEN) ok = true;
  var uri = req.uri;
  if (uri.endsWith("__access_check__")) {
    return ok ? { statusCode: 204, statusDescription: "NoContent" }
              : { statusCode: 403, statusDescription: "Forbidden" };
  }
  var isPrivate = uri.indexOf("/03_outputs/") === 0 || /[0-9a-f]{32}/.test(uri);
  if (!isPrivate || ok) return req;
  return { statusCode: 403, statusDescription: "Forbidden",
           headers: { "cache-control": { value: "no-store" } } };
}
