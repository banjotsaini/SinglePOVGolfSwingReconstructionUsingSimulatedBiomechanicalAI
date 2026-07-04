# MotionCaddie — Front-End Deployment Plan & Handoff

**Owner today:** Austin · **Proposed next owner:** Theo
**Status:** static demo built + working locally; deployment + upload feature not started
**Last updated:** 2026-07-03

---

## TL;DR for whoever picks this up
There's a working static front end at `deploy/web/` (vanilla HTML/CSS/JS, no
framework). It runs locally today off pre-rendered placeholder data for 3 fixed
clips. This doc explains what it is, how it connects to our AWS setup, and the
plan to take it from "local demo" → "hosted app with real uploads." Read the
"Current state" section first so you don't rebuild what exists.

---

## Current state (what already exists — don't redo)
- `deploy/web/index.html · styles.css · app.js` — editable vanilla front end,
  no build step. Three screens: **Pick → Analyze (7-step loader) → Results**
  (Plain English + The Numbers tabs, 9-metric table, 2D overlay, rotatable 3D
  replay).
- `deploy/web/assets/<clip_id>/` — per-clip data: `metrics.json`,
  `explanation.json`, `replay_3d.json`, `overlay.mp4`, `raw.mp4`.
- Data is currently **placeholder** (clip 0 uses real MotionBERT 3D output;
  830/269 are labeled variants). All files carry `"placeholder": true`.
- The single backend swap point is `loadClipBundle(clipId)` in `app.js`, marked
  `TODO(live-backend)`. Everything routes through that one function.
- Run locally: `cd deploy/web && python -m http.server 8000` → localhost:8000.
- ⚠️ Video encoding gotcha: assets must be **H.264/yuv420p** or Chrome won't play
  them. Any new renders have to match.

## Our AWS reality (follow Lawrence's setup — this is decided)
- We use **Lawrence's shared account + IAM users** so the team can collaborate.
  Do NOT follow the Identity Center / `motion-caddie-demo-*` naming in the
  `deploy/infra` handoff docs — that's a different, superseded plan.
- Data bucket: `motioncaddie-capstone-data-lj-2026` (region **us-east-1**),
  folders: `01_inputs/ 02_working/ 03_outputs/ 04_docs/ 05_archive/`.
- LLM eval: **Claude API directly (claude-opus-4-8)** for the demo (best results);
  SageMaker + Hugging Face is the eventual production path.
- **Bedrock is blocked** on course credits — don't use it.

---

## The plan — from local demo to hosted app

### Phase 1 — Host the current static demo (no upload yet)
Goal: get the working static site live on the web so anyone can open it.
1. Create a **separate S3 bucket for the web app** (e.g. `motioncaddie-web-<...>`)
   — this is NOT the data bucket; it holds only the site files (html/css/js/assets).
   Confirm naming with Lawrence since it's his account.
2. `aws s3 sync deploy/web/ s3://<web-bucket>/` to upload the site.
3. Put **CloudFront** in front of the web bucket:
   - CloudFront is a CDN — it serves the site fast, over **HTTPS**, from edge
     locations, and lets us keep the bucket private (access via Origin Access
     Control). Raw S3 static hosting is HTTP-only, so CloudFront is what gives us
     a real `https://` URL.
4. Result: a public URL that serves the static demo. Still fixed clips, still
   pre-rendered — but now hosted, not just localhost.

### Phase 2 — Re-render the demo with REAL pipeline data
Goal: replace placeholder assets with real pipeline output.
- Blocked on the ~135 MB teammate model/data bundle (MixSTE checkpoint,
  `event_detector_tcn.pt`, cached 3D parquets, `Data/coaching/*.json`,
  `golfDB.pkl`). Once it lands (locally or in `02_working/`):
  1. Run the pipeline per clip (`Scripts/demo.py --golfdb-clip <id>`).
  2. Drop real `metrics.json / explanation.json / replay_3d.json / overlay.mp4`
     into `deploy/web/assets/<clip_id>/` (re-encode video to H.264).
  3. Upload the real renders to the data bucket `03_outputs/<clip_id>/`.
  4. No UI changes needed — same asset shape.
- The Claude eval path already exists: `coaching_explain.py --backend anthropic`
  (claude-opus-4-8, grounded). Set `ANTHROPIC_API_KEY` in env; never commit it.

### Phase 3 — Add the upload feature (the real product)
Goal: user uploads their own swing video instead of picking a fixed clip.
This is the big new build. Flow:

```
User uploads video (front end on CloudFront)
      ↓ upload straight to S3 01_inputs/  (presigned URL — see note)
Backend runs the pipeline on the new video
      ↓ writes 2D overlay + 3D reconstruction + metrics.json + eval
        to S3 03_outputs/<id>/
Front end polls / is notified, then displays the results
```

Front-end work for this:
- An upload screen (replace/augment the "Use this sample swing" button) that
  sends the file to S3 `01_inputs/`.
- ⚠️ **Uploads should go directly to S3 via a presigned URL**, not through the
  app server — big video files shouldn't route through compute. The backend
  hands the browser a short-lived presigned PUT URL; the browser uploads straight
  to the bucket.
- A "processing…" state (reuse the existing 7-step Analyze screen) while the
  backend works, since real processing isn't instant.
- A results view that loads the real outputs from `03_outputs/<id>/`.

### Phase 4 — The backend / Lambda question (align before building)
The front end needs *something* to run the pipeline when a video is picked/uploaded.
Per the `deploy/infra` handoff docs, the intended design is:
- **Lambda (CPU) for the cached/light path** — takes a pre-computed 3D pose,
  runs event detection + scorecard + Claude eval, returns results. The front end
  calls a **Lambda Function URL**; Lambda IS the backend compute, not a
  "hand-off" layer.
- **The heavy GPU path (raw video → MediaPipe → 3D lifting) is NOT Lambda** —
  Lambda is CPU-only. That live-upload processing would need an async
  **Fargate/Batch** job triggered by the S3 upload. This is the harder, later
  piece.
- ❗ **Open item:** whether Lambda deploys cleanly under Lawrence's IAM setup is
  unconfirmed. Confirm with Lawrence before committing to the Lambda path. If
  Lambda is awkward, the fallback is running the pipeline in our SageMaker
  environment and having the front end call that.

---

## How the front end connects to Lawrence's S3 (summary)
- **App code** (`deploy/web/`) → its own **web bucket** + CloudFront (Phase 1).
  Code lives in git; the web bucket is just a hosting copy.
- **User-uploaded videos** → data bucket `01_inputs/` (Phase 3, via presigned URL).
- **Pipeline outputs** (overlay, 3D, metrics, eval) → data bucket `03_outputs/`.
- The front end reads results from `03_outputs/` (or from whatever the backend
  returns). Two buckets, two jobs: **web bucket = the site**, **data bucket =
  videos + results**.

## Things not to miss
- Keep `loadClipBundle()` as the single swap point — demo (local JSON) vs live
  (backend/S3) should be a one-function change, no UI rewrite.
- Never commit secrets (`ANTHROPIC_API_KEY` stays in env / Secrets Manager).
- Don't `git add -A` — repo has many unrelated untracked files; stage only
  `deploy/web/` and deploy outputs.
- All video assets must be H.264/yuv420p for browser playback.
- Everything region **us-east-1**.
- Confirm the web-bucket name + Lambda permissions with Lawrence before Phase 1/4.
- CORS: once the front end (CloudFront domain) calls a backend URL cross-origin,
  the backend must send CORS headers or the browser silently blocks the calls.

## Open decisions (need team/Lawrence sign-off)
1. Web-app S3 bucket name (Phase 1).
2. Lambda under Lawrence's account — works, or use SageMaker-hosted backend?
3. Who owns Phase 3 upload vs Phase 4 backend (front-end vs infra split)?
4. Demo scope for next presentation: hosted static (Phase 1) is the safe target;
   live upload (Phase 3) is stretch.
