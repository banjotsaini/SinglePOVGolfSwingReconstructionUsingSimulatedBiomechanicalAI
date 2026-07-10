/* MotionCaddie demo — static front end over pre-rendered pipeline results.
 * Three screens: pick → analyze (animated checklist) → results.
 * No backend: every clip's data is baked under assets/<clip_id>/.
 */
"use strict";

/* =========================================================================
 * Data access — the ONE place that knows where results come from.
 *
 * TODO(live-backend): to switch from pre-rendered files to a live backend,
 * replace the fetches below with a single call like
 *   const r = await fetch(`${API_BASE}/analyze/${clipId}`);
 * returning the same shape: { metrics, explanation, replay, overlayUrl, rawUrl }.
 * Nothing else in this file needs to change.
 * ========================================================================= */
async function loadClipBundle(clipId) {
  const base = `assets/${clipId}`;
  const [metrics, explanation, replay] = await Promise.all([
    fetch(`${base}/metrics.json`).then(r => r.json()),
    fetch(`${base}/explanation.json`).then(r => r.json()),
    fetch(`${base}/replay_3d.json`).then(r => r.json()),
  ]);
  return { metrics, explanation, replay, overlayUrl: `${base}/overlay.mp4`, rawUrl: `${base}/raw.mp4` };
}

async function loadManifest() {
  return (await fetch("assets/clips.json").then(r => r.json())).clips;
}

/* ============================== app state =============================== */
const SAMPLE_ID = "830";   // demo swing used to illustrate a prototype upload result

const state = {
  clips: [],
  selectedId: null,
  mode: "demo",       // "demo" (real pre-rendered clip) | "upload" (prototype)
  upload: null,       // { name, url } for a prototype uploaded file
  uploads: [],        // this session's prototype uploads (for the dashboard)
  bundle: null,       // loaded clip bundle for the results screen
  bundlePromise: null,
  viewer: null,       // Replay3D instance
};

const $ = (sel) => document.querySelector(sel);

/* Replace an element with a fresh clone (keeps id/attrs). Used for the replay
 * canvas so a new viewer never inherits a stale WebGL/2D drawing context. */
function resetCanvas(sel) {
  const old = $(sel);
  const fresh = old.cloneNode(false);
  old.replaceWith(fresh);
  return fresh;
}

/* ============================ screen router ============================= */
const SCREENS = { pick: "#screen-pick", analyze: "#screen-analyze", results: "#screen-results" };
const NAV_ORDER = ["pick", "analyze", "results"];

function goto(screen) {
  for (const [name, sel] of Object.entries(SCREENS)) {
    $(sel).hidden = name !== screen;
  }
  const idx = NAV_ORDER.indexOf(screen);
  document.querySelectorAll(".pstep").forEach((el, i) => {
    el.classList.toggle("active", i === idx);
    el.classList.toggle("done", i < idx);
  });
  window.scrollTo({ top: 0 });
}

/* ============================ 1 · landing screen ======================= */
function renderGallery() {
  const gal = $("#clip-gallery");
  gal.innerHTML = "";
  for (const clip of state.clips) {
    const card = document.createElement("button");
    card.className = "clip-card";
    card.dataset.id = clip.id;
    card.innerHTML = `
      <video src="assets/${clip.id}/raw.mp4" preload="metadata" muted playsinline></video>
      <div class="clip-title">${clip.title}</div>
      <div class="clip-meta">${clip.view} · ${clip.club}</div>
      <div class="clip-cta">Open result</div>`;
    card.addEventListener("click", () => startDemo(clip.id));
    gal.appendChild(card);
  }
}

/* start a real pre-rendered demo swing */
function startDemo(id) {
  state.mode = "demo";
  state.selectedId = id;
  state.upload = null;
  runAnalyze();
}

/* ---- prototype upload: pick a local file, then run the SAME loader ---- */
function onFileChosen(file) {
  if (!file) return;
  if (state.upload && state.upload.url) URL.revokeObjectURL(state.upload.url);
  state.upload = { name: file.name, url: URL.createObjectURL(file) };
  const box = $("#upload-status");
  box.hidden = false;
  box.innerHTML = `
    <div class="upload-file">
      <span class="upload-file-name" title="${file.name}">${file.name}</span>
      <span class="upload-badge">Prototype — no real analysis</span>
    </div>
    <button id="btn-analyze-upload" class="btn-primary" type="button">Analyze swing</button>`;
  $("#btn-analyze-upload").addEventListener("click", startUpload);
}

function startUpload() {
  if (!state.upload) return;
  state.mode = "upload";
  state.selectedId = SAMPLE_ID;   // illustrative results bundle; overlay uses the real file
  if (!state.uploads.some(u => u.name === state.upload.name)) {
    state.uploads.push({ name: state.upload.name });
    renderDashboard();
  }
  runAnalyze();
}

/* ========================== 2 · analyze screen ========================== */
const STEP_MS = 430;  // pre-rendered demo: tick quickly, then reveal results

function runAnalyze() {
  // kick off the data load in parallel with the animation
  state.bundlePromise = loadClipBundle(state.selectedId);
  goto("analyze");

  const items = [...document.querySelectorAll("#analyze-steps li")];
  items.forEach(li => li.classList.remove("doing", "done"));

  let i = 0;
  const tick = () => {
    if (i > 0) items[i - 1].classList.replace("doing", "done");
    if (i < items.length) {
      items[i].classList.add("doing");
      i += 1;
      setTimeout(tick, STEP_MS);
    } else {
      state.bundlePromise.then(bundle => {
        state.bundle = bundle;
        setTimeout(() => { renderResults(); goto("results"); }, 350);
      }).catch(err => {
        alert(`Could not load results for clip ${state.selectedId}: ${err}`);
        goto("pick");
      });
    }
  };
  tick();
}

/* ========================== 3 · results screen ========================= */
function metricByKey(key) {
  return state.bundle.metrics.metrics.find(m => m.key === key);
}

function renderResults() {
  const { metrics, explanation, overlayUrl, replay } = state.bundle;
  const clip = state.clips.find(c => c.id === state.selectedId);
  const isUpload = state.mode === "upload";

  // honesty banner + labelling for prototype uploads
  $("#results-banner").hidden = !isUpload;
  if (isUpload) {
    $("#results-clip-label").textContent = `Your upload · ${state.upload.name}`;
    $("#results-headline").textContent = "Illustrative result (prototype)";
  } else {
    $("#results-clip-label").textContent = `${clip.title} · ${clip.view} · ${clip.club}`;
    $("#results-headline").textContent = explanation.headline;
  }

  renderPlain(explanation);
  renderNumbers(metrics.metrics);
  Chat.activate(state.selectedId);

  const vid = $("#overlay-video");
  // upload mode plays back the user's ACTUAL file (truthful — their raw clip, no overlay claimed)
  vid.src = isUpload ? state.upload.url : overlayUrl;
  vid.load();

  if (state.viewer) state.viewer.destroy();
  // fresh canvas each time so a WebGL/2D context is never reused across viewers
  const canvas = resetCanvas("#replay-canvas");
  const ui3d = { scrub: $("#replay-scrub"), label: $("#replay-frame"), playBtn: $("#replay-play") };
  const Cap = window.CapsuleViewer3D;   // WebGL capsule viewer (module); falls back to canvas
  state.viewer = (Cap && Cap.supported())
    ? new Cap(canvas, replay, ui3d)
    : new Replay3D(canvas, replay, ui3d);

  loadRenderView(state.selectedId, isUpload);
  showTab("plain");
}

/* ---- optional "3D Swing View": precomputed headless-Blender render stills ----
 * Presence-driven off the clips.json manifest: the tab appears only for clips whose
 * entry carries a `render.phases` list (no per-clip fetch/404). Uploads never show it.
 *
 * TODO(mixste-swap): clip 0's render is currently generated from the MotionBERT-full
 * lift (the mocap JSON that exists today), NOT the golfpose3d/MixSTE production lifter.
 * Once the model bundle lands, regenerate Data/handoff/0/0_mocap.json through the
 * golfpose3d (MixSTE) path and re-run Scripts/blender_mocap.py (unisex mode) to
 * overwrite the PNGs in assets/0/blender/ in place — no front-end change needed (same
 * filenames = same manifest). The caption is deliberately lifter-neutral so it stays
 * true after the swap. To add a NEW clip's render: drop its PNGs under
 * assets/<id>/blender/ and add a `render.phases` block to that clip in clips.json.
 */
function loadRenderView(clipId, isUpload) {
  const tab = $("#tab-render"), panel = $("#render-phases");
  tab.hidden = true;                       // default: no render for this clip
  if (isUpload) return;                    // uploads use a sample clip; never claim a render
  const clip = state.clips.find(c => c.id === clipId);
  const render = clip && clip.render;
  if (!render || !Array.isArray(render.phases) || !render.phases.length) return;
  // optional avatar animation above the stills (presence-driven, like the phases)
  const vid = render.video
    ? `<video class="render-video" src="assets/${clipId}/blender/${render.video}"
         controls muted loop playsinline preload="metadata"></video>`
    : "";
  panel.innerHTML = vid + render.phases.map(p =>
    `<figure class="render-phase">
       <img src="assets/${clipId}/blender/${p.src}" alt="${p.label} — rendered 3D pose" loading="lazy">
       <figcaption>${p.label}</figcaption>
     </figure>`).join("");
  tab.hidden = false;
}

function renderPlain(explanation) {
  const el = $("#view-plain");
  el.innerHTML = "";
  for (const sec of explanation.sections) {
    const tagLabel = { good: "Good", watch: "Watch", low: "Low confidence" }[sec.tag] || sec.tag;
    const chips = (sec.chips || []).map(key => {
      const m = metricByKey(key);
      return m ? `<span class="chip">${m.label} · <b>${m.you_display}</b></span>` : "";
    }).join("");
    const div = document.createElement("div");
    div.className = "eval-section";
    div.innerHTML = `
      <div class="eval-tagrow">
        <span class="eval-tag ${sec.tag}">${tagLabel}</span>
        <h3>${sec.title}</h3>
      </div>
      <p>${sec.body}</p>
      <div class="chips">${chips}</div>`;
    el.appendChild(div);
  }
}

function renderNumbers(metrics) {
  const el = $("#metrics-table");
  el.innerHTML = "";
  for (const m of metrics) {
    const [lo, hi] = m.axis;
    const pct = v => Math.min(100, Math.max(0, (v - lo) / (hi - lo) * 100));
    const row = document.createElement("div");
    row.className = "metric-row";
    row.innerHTML = `
      <div class="metric-name">
        <i class="dot dot-${m.status}"></i><strong>${m.label}</strong>
      </div>
      <div class="axis">
        <div class="axis-track"></div>
        <div class="axis-band" style="left:${pct(m.band[0])}%; width:${pct(m.band[1]) - pct(m.band[0])}%"></div>
        <div class="axis-tour" style="left:${pct(m.tour)}%" title="tour median ${m.tour_display}"></div>
        <div class="axis-you ${m.status}" style="left:${pct(m.you)}%" title="you: ${m.you_display}"></div>
      </div>
      <div class="metric-vals">
        <div class="you-val">${m.you_display}</div>
        <div class="tour-val">tour ${m.tour_display}</div>
      </div>
      <p class="metric-blurb">${m.blurb}</p>`;
    el.appendChild(row);
  }
}

function showTab(which) {
  $("#tab-plain").classList.toggle("active", which === "plain");
  $("#tab-numbers").classList.toggle("active", which === "numbers");
  $("#tab-chat").classList.toggle("active", which === "chat");
  $("#tab-render").classList.toggle("active", which === "render");
  $("#view-plain").hidden = which !== "plain";
  $("#view-numbers").hidden = which !== "numbers";
  $("#view-chat").hidden = which !== "chat";
  $("#view-render").hidden = which !== "render";
}

/* =========================================================================
 * Replay3D — tiny dependency-free 3D skeleton viewer (canvas 2D).
 * Data: { fps, joint_names, bones:[{a,b,color}], frames:[T][J][3] } in
 * h36m-camera axes (y is DOWN) — flipped to y-up for display.
 * Drag to orbit (yaw/pitch); play/pause + frame scrub.
 * ========================================================================= */
class Replay3D {
  constructor(canvas, data, ui) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.data = data;
    this.ui = ui;
    this.frame = 0;
    this.playing = true;
    this.yaw = -0.6;
    this.pitch = 0.12;
    this._destroyed = false;
    this._lastT = 0;
    this._acc = 0;

    // fit: center on the mid-hips averaged over the clip, y flipped
    const pts = data.frames.flat();
    let maxR = 1e-6;
    const c = [0, 0, 0];
    const hipIdx = data.joint_names.indexOf("hip_center");
    for (const fr of data.frames) {
      const h = fr[hipIdx >= 0 ? hipIdx : 0];
      c[0] += h[0]; c[1] += h[1]; c[2] += h[2];
    }
    c[0] /= data.frames.length; c[1] /= data.frames.length; c[2] /= data.frames.length;
    this.center = c;
    for (const p of pts) {
      const r = Math.hypot(p[0] - c[0], p[1] - c[1], p[2] - c[2]);
      if (r > maxR) maxR = r;
    }
    this.radius = maxR;

    // hi-dpi
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth || canvas.width, h = canvas.clientHeight || canvas.height;
    canvas.width = w * dpr; canvas.height = h * dpr;
    this.w = w; this.h = h;
    this.ctx.scale(dpr, dpr);

    // interactions
    this._onDown = e => { this.drag = { x: e.clientX, y: e.clientY }; };
    this._onMove = e => {
      if (!this.drag) return;
      this.yaw += (e.clientX - this.drag.x) * 0.01;
      this.pitch = Math.max(-1.2, Math.min(1.2, this.pitch + (e.clientY - this.drag.y) * 0.01));
      this.drag = { x: e.clientX, y: e.clientY };
      if (!this.playing) this.draw();
    };
    this._onUp = () => { this.drag = null; };
    canvas.addEventListener("pointerdown", this._onDown);
    window.addEventListener("pointermove", this._onMove);
    window.addEventListener("pointerup", this._onUp);

    ui.scrub.max = data.frames.length - 1;
    ui.scrub.value = 0;
    this._onScrub = () => { this.playing = false; this.frame = +ui.scrub.value; this.draw(); };
    this._onPlay = () => { this.playing = !this.playing; };
    ui.scrub.addEventListener("input", this._onScrub);
    ui.playBtn.addEventListener("click", this._onPlay);

    requestAnimationFrame(t => this._loop(t));
  }

  destroy() {
    this._destroyed = true;
    this.canvas.removeEventListener("pointerdown", this._onDown);
    window.removeEventListener("pointermove", this._onMove);
    window.removeEventListener("pointerup", this._onUp);
    this.ui.scrub.removeEventListener("input", this._onScrub);
    this.ui.playBtn.removeEventListener("click", this._onPlay);
  }

  _loop(t) {
    if (this._destroyed) return;
    if (this.playing) {
      if (this._lastT) {
        this._acc += (t - this._lastT) / 1000;
        const spf = 1 / (this.data.fps || 30);
        while (this._acc >= spf) {
          this._acc -= spf;
          this.frame = (this.frame + 1) % this.data.frames.length;
        }
      }
      this.ui.scrub.value = this.frame;
      this.draw();
    }
    this._lastT = t;
    requestAnimationFrame(tt => this._loop(tt));
  }

  _project(p) {
    // recenter, flip y (h36m y is down), orbit, mild perspective
    let x = p[0] - this.center[0];
    let y = -(p[1] - this.center[1]);
    let z = p[2] - this.center[2];
    const cy = Math.cos(this.yaw), sy = Math.sin(this.yaw);
    [x, z] = [x * cy + z * sy, -x * sy + z * cy];
    const cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
    [y, z] = [y * cp - z * sp, y * sp + z * cp];
    const persp = 1 / (1 + (z / this.radius) * 0.18);
    const s = (Math.min(this.w, this.h) * 0.40) / this.radius;
    return [this.w / 2 + x * s * persp, this.h / 2 - y * s * persp, persp];
  }

  draw() {
    const { ctx } = this;
    ctx.clearRect(0, 0, this.w, this.h);
    ctx.fillStyle = "#10231a";
    ctx.fillRect(0, 0, this.w, this.h);

    // ground grid (square at the lowest point of frame 0)
    const groundY = this.center[1] + this.radius * 0.02 +
      Math.max(...this.data.frames[0].map(p => p[1] - this.center[1]));
    ctx.strokeStyle = "rgba(127,227,172,.14)";
    ctx.lineWidth = 1;
    const G = this.radius * 0.9, N = 6;
    for (let i = 0; i <= N; i++) {
      const u = -G + (2 * G * i) / N;
      let a = this._project([this.center[0] + u, groundY, this.center[2] - G]);
      let b = this._project([this.center[0] + u, groundY, this.center[2] + G]);
      ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
      a = this._project([this.center[0] - G, groundY, this.center[2] + u]);
      b = this._project([this.center[0] + G, groundY, this.center[2] + u]);
      ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
    }

    const fr = this.data.frames[this.frame];
    const proj = fr.map(p => this._project(p));

    // bones (rainbow map from the design spec), far bones first
    const bones = [...this.data.bones].sort((p, q) =>
      Math.min(proj[p.a][2], proj[p.b][2]) - Math.min(proj[q.a][2], proj[q.b][2]));
    for (const bone of bones) {
      const a = proj[bone.a], b = proj[bone.b];
      ctx.strokeStyle = bone.color;
      ctx.lineWidth = 3.5 * Math.min(a[2], b[2]);
      ctx.lineCap = "round";
      ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
    }
    // joints
    ctx.fillStyle = "#faf9f5";
    for (const p of proj) {
      ctx.beginPath(); ctx.arc(p[0], p[1], 2.4 * p[2], 0, Math.PI * 2); ctx.fill();
    }

    this.ui.label.textContent = `${this.frame + 1} / ${this.data.frames.length}`;
  }
}

/* ================================ wiring ================================ */
async function init() {
  state.clips = await loadManifest();
  renderGallery();

  // Prefetch every clip's metrics.json (small) so the coach chat can talk about
  // any swing and compare across them — same source the "numbers" tab renders.
  const metricsByClip = {};
  await Promise.all(state.clips.map(async (c) => {
    try { metricsByClip[c.id] = await fetch(`assets/${c.id}/metrics.json`).then(r => r.json()); }
    catch (e) { /* a clip without metrics just won't be chat-enabled */ }
  }));
  Chat.setLibrary(state.clips, metricsByClip);
  Chat.init({
    stream: $("#chat-stream"), input: $("#chat-q"), send: $("#chat-send"),
    compare: $("#chat-compare"), active: $("#chat-active"), mode: $("#chat-mode"),
    chips: [...document.querySelectorAll("#view-chat .chip-btn")],
  });

  // upload (prototype): primary CTA opens the file picker
  $("#btn-upload").addEventListener("click", () => $("#file-input").click());
  $("#file-input").addEventListener("change", (e) => onFileChosen(e.target.files[0]));

  $("#btn-restart").addEventListener("click", () => {
    if (state.viewer) { state.viewer.destroy(); state.viewer = null; }
    $("#overlay-video").pause();
    goto("pick");
  });
  $("#tab-plain").addEventListener("click", () => showTab("plain"));
  $("#tab-numbers").addEventListener("click", () => showTab("numbers"));
  $("#tab-chat").addEventListener("click", () => showTab("chat"));
  $("#tab-render").addEventListener("click", () => showTab("render"));

  // prototype account: render the signed-in dashboard on any auth change
  if (window.Auth) Auth.init();
  document.addEventListener("mc-auth-change", renderDashboard);
  renderDashboard();

  goto("pick");
}

/* ---- signed-in "My swings" dashboard (prototype) ---- */
function renderDashboard() {
  const dash = $("#dashboard");
  if (!dash) return;
  const signedIn = window.Auth && Auth.isSignedIn();
  dash.hidden = !signedIn;
  if (!signedIn) return;
  const user = Auth.currentUser();
  $("#dash-name").textContent = user.displayName || user.username;
  const wrap = $("#dash-swings");
  if (!state.uploads.length) {
    wrap.innerHTML = `<p class="muted small">No swings yet. Upload a swing video above to get started — analysis is prototype-only in this demo.</p>`;
    return;
  }
  wrap.innerHTML = state.uploads.map(u =>
    `<div class="dash-swing"><span class="dash-swing-name" title="${u.name}">${u.name}</span>` +
    `<span class="upload-badge">Prototype</span></div>`).join("");
}

init();
