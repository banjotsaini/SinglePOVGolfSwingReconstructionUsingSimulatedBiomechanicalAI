/* MotionCaddie — coaching Q&A chat (folded in from Banjot's coaching_chatbot_demo.html).
 *
 * The grounded-answer ENGINE (indicator copy, tool mocks, keyword router, refusal
 * taxonomy, grounding verifier) is ported verbatim in spirit from that page so the
 * offline demo behaves like the real agent. What changed for this integration:
 *   - per-clip VALUES / BAND / STATUS are read from each clip's assets/<id>/metrics.json
 *     (the SAME source the "The numbers" tab renders) so chat and table never disagree.
 *   - META below is copy ONLY (plain names, in/low/high phrasing, glossary).
 *
 * TODO(live-backend): flip to the real grounded agent by setting `window.API_BASE`
 * (e.g. a Lambda Function URL, or serve_demo.py's origin). askChat() then POSTs
 *   {clip_id, question, history[], compare_clip_id?}  ->  {answer, grounded, violations, tool_log}
 * and nothing else changes. Unset => the in-page mock below answers. This is the
 * single chat swap seam, mirroring loadClipBundle() for data.
 */
"use strict";

/* ---- indicator COPY (bands/values come from metrics.json, not here) ---- */
const CHAT_META = {
  tempo_ratio: { plain: "the rhythm of the swing", unit: "",
    in: "the swing rhythm is close to the tour-typical pace",
    lo: "the backswing-to-downswing rhythm is quicker than the roughly 3-to-1 tour norm",
    hi: "the backswing is longer relative to the downswing than the roughly 3-to-1 tour norm" },
  shoulder_turn_top_deg: { plain: "how far the shoulders rotate at the top", unit: "°",
    in: "the shoulders rotate about as far as a typical tour swing at the top",
    lo: "the shoulders rotate less than most tour swings at the top",
    hi: "the shoulders rotate further than most tour swings at the top" },
  hip_turn_top_deg: { plain: "how far the hips rotate at the top", unit: "°",
    in: "the hips rotate about as far as a typical tour swing at the top",
    lo: "the hips rotate less than most tour swings at the top",
    hi: "the hips rotate further than most tour swings at the top" },
  x_factor_top_deg: { plain: "the gap between shoulder turn and hip turn at the top", unit: "°",
    in: "the separation between shoulders and hips is about typical for a tour swing",
    lo: "there is less separation between the shoulders and hips than in most tour swings",
    hi: "there is more separation between the shoulders and hips than in most tour swings" },
  hip_turn_impact_deg: { plain: "how open the hips are at impact", unit: "°",
    in: "the hips are about as open at impact as a typical tour swing",
    lo: "the hips are less open at impact than most tour swings",
    hi: "the hips are more open at impact than most tour swings" },
  spine_tilt_address_deg: { plain: "how much the upper body leans at setup", unit: "°",
    in: "the upper body leans about as much as a typical tour swing at setup",
    lo: "the upper body leans less than most tour swings at setup",
    hi: "the upper body leans more than most tour swings at setup" },
  posture_loss_deg: { plain: "how much the spine angle changes from setup to impact", unit: "°",
    in: "the posture holds about as steadily from setup to impact as a typical tour swing",
    lo: "",
    hi: "the spine angle changes more from setup to impact than in most tour swings, meaning the posture shifts during the swing" },
  head_sway_max_pct: { plain: "how much the head drifts sideways", unit: "%",
    in: "the head stays about as steady side-to-side as a typical tour swing",
    lo: "the head moves less side-to-side than most tour swings",
    hi: "the head drifts further side-to-side than most tour swings" },
  hip_lateral_shift_pct: { plain: "how much the hips slide toward the target", unit: "%",
    in: "the hips slide toward the target about as much as a typical tour swing",
    lo: "the hips slide toward the target less than most tour swings",
    hi: "the hips slide sideways more than most tour swings" },
  left_arm_bend_top_deg: { plain: "how straight the lead arm is at the top", unit: "°",
    in: "the lead arm is about as extended at the top as a typical tour swing",
    lo: "the lead arm is more bent at the top than most tour swings", hi: "" },
};
const GLOSSARY = {
  "tour range": "the typical spread of values seen across professional tour swings",
  "tour median": "the middle value among professional tour swings",
  "address": "the setup position, just before the swing starts",
  "top": "the top of the backswing, where the club changes direction",
  "impact": "the moment the club meets the ball",
  "x-factor": "the difference between how far the shoulders and the hips have turned at the top — a measure of coil",
};

/* ============================== chat module ============================== */
const Chat = (() => {
  let ACTIVE = null, COMPARE = null;
  const CLIPDATA = {};       // id -> { name, club, view, ind:{key:{label,value,unit,band,tour,status,tier,plain,in,lo,hi}} }
  const HIST = {};           // per-clip [{role,content}] for the live backend
  let dom = {};

  /* ---- build a clip's indicator record from its metrics.json rows ---- */
  function buildFromMetrics(clip, rows) {
    const ind = {};
    for (const r of rows) {
      const m = CHAT_META[r.key] || {};
      ind[r.key] = {
        label: r.label, value: r.you, unit: m.unit || "",
        band: r.band, tour: r.tour, status: r.status,      // "in" | "watch" | "low"
        tier: r.status === "low" ? "low" : "med",
        plain: m.plain || r.label, in: m.in || "", lo: m.lo || "", hi: m.hi || "",
      };
    }
    CLIPDATA[clip.id] = { name: clip.title, club: clip.club, view: clip.view, ind };
  }

  /* ---- accessors ---- */
  const IND = (c) => CLIPDATA[c].ind;
  const keys = (c) => Object.keys(IND(c));
  const val = (k, c) => IND(c)[k].value;
  const unit = (k, c) => IND(c)[k].unit;
  const band = (k, c) => IND(c)[k].band;
  const tour = (k, c) => IND(c)[k].tour;
  const tier = (k, c) => IND(c)[k].tier;
  const label = (k, c) => IND(c)[k].label;
  const inRange = (k, c) => IND(c)[k].status === "in";
  const reliableKeys = (c) => keys(c).filter(k => tier(k, c) !== "low");
  const clipName = (c) => (CLIPDATA[c] ? CLIPDATA[c].name : "swing " + c);

  const cap = s => s.charAt(0).toUpperCase() + s.slice(1);
  const low = s => s.charAt(0).toLowerCase() + s.slice(1);
  const listEng = a => a.length <= 1 ? (a[0] || "") : a.slice(0, -1).join(", ") + " and " + a[a.length - 1];

  /* ---- tools (mirror coaching_chat.py; read ACTIVE, COMPARE) ---- */
  function t_list() { return { indicators: keys(ACTIVE).map(k => ({ key: k, label: label(k, ACTIVE), reliable: tier(k, ACTIVE) !== "low" })), count: keys(ACTIVE).length }; }
  function t_get(k) {
    if (!IND(ACTIVE)[k]) return { measured: false, key: k, note: "not measured for this swing" };
    const o = { measured: true, key: k, label: label(k, ACTIVE), value: val(k, ACTIVE), unit: unit(k, ACTIVE),
      tour_median: tour(k, ACTIVE), pro_band: band(k, ACTIVE), in_tour_range: inRange(k, ACTIVE),
      confidence_tier: tier(k, ACTIVE), reliable: tier(k, ACTIVE) !== "low" };
    if (!o.reliable) o.note = "LOW CONFIDENCE from a single camera — do not state or judge the value";
    return o;
  }
  function t_compare(k) {
    if (!IND(ACTIVE)[k]) return { comparable: false, key: k };
    if (COMPARE == null) return { comparable: false, key: k, note: "no comparison swing selected" };
    if (tier(k, ACTIVE) === "low") return { comparable: false, key: k, confidence_tier: "low", note: "low-confidence; not reliable enough to compare" };
    const a = val(k, ACTIVE), b = (IND(COMPARE)[k] ? val(k, COMPARE) : null);
    if (b == null) return { comparable: false, key: k, note: "not measured on the comparison swing" };
    const d = Math.round((a - b) * 100) / 100;
    return { comparable: true, key: k, earlier_value: b, current_value: a, delta: d,
      direction: d > 0 ? "increased" : d < 0 ? "decreased" : "unchanged",
      current_in_tour_range: inRange(k, ACTIVE), earlier_in_tour_range: inRange(k, COMPARE) };
  }
  function t_flagged() { const f = reliableKeys(ACTIVE).filter(k => !inRange(k, ACTIVE)); return { flagged: f.map(k => ({ key: k, label: label(k, ACTIVE) })), count: f.length }; }
  function t_summary() { const rel = reliableKeys(ACTIVE); return { n_indicators: keys(ACTIVE).length, n_reliable: rel.length, n_in_tour_range: rel.filter(k => inRange(k, ACTIVE)).length, n_flagged: rel.filter(k => !inRange(k, ACTIVE)).length }; }
  function t_term(term) { const t = term.toLowerCase().trim(); return GLOSSARY[t] ? { term: t, in_glossary: true, gloss: GLOSSARY[t] } : { term: t, in_glossary: false, note: "not in glossary" }; }

  /* ---- keyword routing ---- */
  const UNMEASURED = ["ball", "distance", "how far", "carry", "yard", "slice", "hook", "grip", "club face", "clubface", "face open", "face closed", "swing plane", "plane", "hinge", "clubhead", "club head", "speed", "mph", "which club", "what club", "launch", "spin", "draw", "fade"];
  const LOWCONF = ["lead arm", "left arm", "trail arm", "right arm", "arm straight", "arm bend", "arm extension", "elbow"];
  const FIX = ["what should i", "how do i fix", "how can i fix", "what do i fix", "fix my", "should i work on", "drill", "how to improve my", "what to improve", "help me fix", "what can i do to"];
  const PROG = ["improve", "improved", "progress", "better than", "since last", "last time", "before", "earlier", "used to", "weeks ago", "week ago", "changed", "change since", "compared", "compare", "comparison", "versus", "differ", "against", "getting better", "any better"];
  const SUMMARY = /takeaway|summar|overall|rundown|overview|biggest|key (point|thing)|main (thing|point)|how did i do|break it down|how (was|were) my swing|tell me about my swing|in general|the gist|highlight|how('?s| is) my swing|report/;
  const REF = /all of the above|all of them|all of it|all of my|all the metrics|all metrics|everything|every metric|full (breakdown|rundown|report)|list them all|show me all|each of (them|those)|go through (them|all)/;
  const has = (arr, ql) => arr.some(w => ql.includes(w));

  function matchMetrics(ql) {
    const present = new Set(keys(ACTIVE));
    const f = [], add = k => { if (present.has(k) && !f.includes(k)) f.push(k); };
    if (/\btempo\b|rhythm|\bpace\b/.test(ql)) add("tempo_ratio");
    if (/x[-\s]?factor|separation|shoulder[-\s]hip/.test(ql)) add("x_factor_top_deg");
    if (/\bshoulder/.test(ql) && !/shoulder[-\s]hip/.test(ql)) add("shoulder_turn_top_deg");
    if (/\bhip/.test(ql)) { if (/impact/.test(ql)) add("hip_turn_impact_deg"); else if (!/weight|lateral|shift|transfer/.test(ql)) add("hip_turn_top_deg"); }
    if (/weight shift|weight transfer|\btransfer\b|lateral|slide toward|\bweight\b/.test(ql)) add("hip_lateral_shift_pct");
    if (/posture|hold my angle|spine angle change|stay down|keep.*posture|hold.*posture/.test(ql)) add("posture_loss_deg");
    if (/\bspine\b|setup lean|posture at setup|at address/.test(ql) && !/change/.test(ql)) add("spine_tilt_address_deg");
    if (/head sway|head move|side to side|head still|\bhead\b/.test(ql)) add("head_sway_max_pct");
    return f;
  }
  const findKey = ql => matchMetrics(ql)[0] || null;

  function rundown(tc, call, ks, cmp) {
    const lines = [cmp ? `Here's this swing next to ${clipName(COMPARE)}'s:` : "Here's a quick rundown:"];
    ks.forEach(k => {
      const L = cap(low(label(k, ACTIVE)));
      if (tier(k, ACTIVE) === "low") { call("get_indicator", { key: k }, t_get(k)); lines.push(`• ${L}: measured from a single camera, so it's low-confidence — not reliable enough to call.`); return; }
      if (cmp) { const r = call("compare_indicator", { key: k }, t_compare(k)); if (!r.comparable) { lines.push(`• ${L}: no comparison available.`); return; } lines.push(`• ${L}: ${r.current_value}${unit(k, ACTIVE)} here vs ${r.earlier_value}${unit(k, ACTIVE)} for ${clipName(COMPARE)} (${r.current_in_tour_range ? "in range" : "outside range"}).`); }
      else { const r = call("get_indicator", { key: k }, t_get(k)); lines.push(`• ${L}: ${r.value}${unit(k, ACTIVE)} — ${r.in_tour_range ? "inside the tour range" : "outside the tour range"}.`); }
    });
    return { toolCalls: tc, answer: lines.join("\n") };
  }
  function summarize(tc, call, cmp) {
    const sum = call("get_swing_summary", {}, t_summary());
    const fl = call("get_flagged_observations", {}, t_flagged());
    const inR = reliableKeys(ACTIVE).filter(k => inRange(k, ACTIVE));
    const notable = ["tempo_ratio", "shoulder_turn_top_deg", "x_factor_top_deg", "hip_lateral_shift_pct"].filter(k => inR.includes(k)).slice(0, 3);
    const strong = (notable.length ? notable : inR.slice(0, 3));
    strong.forEach(k => call("get_indicator", { key: k }, t_get(k)));
    const lines = [`Here are the biggest takeaways from ${clipName(ACTIVE)}'s swing:`,
      `• Of ${sum.n_reliable} reliably-measured things, ${sum.n_in_tour_range} are tour-like.`,
      `• Working well: ${listEng(strong.map(k => `${low(label(k, ACTIVE))} (${val(k, ACTIVE)}${unit(k, ACTIVE)})`))}.`];
    lines.push(fl.count ? `• Outside the tour range: ${listEng(fl.flagged.map(f => low(f.label)))}.` : "• Nothing fell outside the tour range on this swing.");
    if (cmp) { const diff = reliableKeys(ACTIVE).filter(k => IND(COMPARE)[k] && inRange(k, ACTIVE) && !inRange(k, COMPARE)); diff.forEach(k => call("compare_indicator", { key: k }, t_compare(k))); if (diff.length) lines.push(`• In range here but not in ${clipName(COMPARE)}'s: ${listEng(diff.map(k => low(label(k, ACTIVE))))}.`); }
    return { toolCalls: tc, answer: lines.join("\n") };
  }

  /* ---- the offline "brain" ---- */
  function respond(q) {
    const ql = q.toLowerCase();
    const tc = []; const call = (name, input, result) => { tc.push({ name, input, result }); return result; };
    const cmp = (COMPARE != null);
    if (/what('| i)?s |what does |what is |explain |meaning of |define /.test(ql) && !has(PROG, ql)) {
      const m = ql.match(/(?:what(?:'s| is| does)?|explain|define|meaning of)\s+(?:an?\s+|the\s+|my\s+)?([a-z\- ]+?)(?:\s+mean)?[\?\.]?$/);
      let term = m ? m[1].trim() : "";
      if (term === "x factor") term = "x-factor";
      if (GLOSSARY[term]) { const r = call("explain_term", { term }, t_term(term)); if (r.in_glossary) return { toolCalls: tc, answer: `${cap(term)} is ${r.gloss}.` }; }
    }
    if (has(UNMEASURED, ql) && !findKey(ql)) { call("list_indicators", {}, { count: t_list().count, matched: false }); return { toolCalls: tc, refuse: true, answer: "That isn't something this swing analysis measures — it works from body motion, not ball flight, club, or contact. So I can't tell you that from what I have." }; }
    if (has(LOWCONF, ql) && matchMetrics(ql).length === 0) { call("get_indicator", { key: "left_arm_bend_top_deg" }, t_get("left_arm_bend_top_deg")); return { toolCalls: tc, refuse: true, answer: "That one comes from a single camera, so the measurement isn't reliable enough to judge — I'd rather not call it either way than guess." }; }
    if (has(FIX, ql) && !has(PROG, ql) && !SUMMARY.test(ql)) { return { toolCalls: tc, refuse: true, answer: "I can describe what your swing did, but I don't give fixes or drills — that's outside what this tool is meant to do. Want me to walk through what stood out instead?" }; }
    if (REF.test(ql)) return rundown(tc, call, reliableKeys(ACTIVE), has(PROG, ql) && cmp);
    if (SUMMARY.test(ql) && matchMetrics(ql).length < 2) return summarize(tc, call, cmp);
    { const ms = matchMetrics(ql); if (ms.length >= 2 && (/[,&]|\band\b|\bplus\b/.test(ql) || ms.length >= 3)) return rundown(tc, call, ms, has(PROG, ql) && cmp); }
    if (has(PROG, ql)) {
      if (!cmp) return { toolCalls: tc, refuse: true, answer: "Pick a swing to compare against — use the “Compare with” selector above the chat — and I'll line the two up." };
      const k = findKey(ql);
      if (k) {
        if (tier(k, ACTIVE) === "low") { call("compare_indicator", { key: k }, t_compare(k)); return { toolCalls: tc, refuse: true, answer: `${label(k, ACTIVE)} is a low-confidence measurement, so I can't reliably compare it across swings.` }; }
        const r = call("compare_indicator", { key: k }, t_compare(k));
        if (!r.comparable) return { toolCalls: tc, answer: `I don't have ${low(label(k, ACTIVE))} on ${clipName(COMPARE)}'s swing to compare against.` };
        let s = `For ${low(label(k, ACTIVE))}, this swing is ${r.current_value}${unit(k, ACTIVE)} (${r.current_in_tour_range ? "in the tour range" : "outside the tour range"}) versus ${r.earlier_value}${unit(k, ACTIVE)} for ${clipName(COMPARE)} (${r.earlier_in_tour_range ? "in range" : "outside range"}).`;
        if (Math.abs(r.delta) > 0) s += ` A ${Math.abs(r.delta)}${unit(k, ACTIVE)} difference.`;
        return { toolCalls: tc, answer: s };
      }
      call("get_swing_summary", {}, t_summary());
      const diff = reliableKeys(ACTIVE).filter(k => IND(COMPARE)[k] && inRange(k, ACTIVE) && !inRange(k, COMPARE));
      diff.forEach(k => call("compare_indicator", { key: k }, t_compare(k)));
      if (diff.length) return { toolCalls: tc, answer: `Compared with ${clipName(COMPARE)}, this swing is inside the tour range on ${listEng(diff.map(k => low(label(k, ACTIVE))))} where ${clipName(COMPARE)}'s isn't. Ask about any one for the numbers.` };
      return { toolCalls: tc, answer: `This swing and ${clipName(COMPARE)}'s land inside the tour range on the same metrics — no in/out-of-range differences to call out.` };
    }
    if (/stood out|stand out|flag|issue|problem|wrong|off|notable|notice/.test(ql)) {
      const r = call("get_flagged_observations", {}, t_flagged());
      if (!r.count) return { toolCalls: tc, answer: "Nothing stood out — every reliably-measured metric is inside the tour range on this swing." };
      const names = r.flagged.map(f => low(f.label));
      return { toolCalls: tc, answer: `On this swing, ${listEng(names)} ${r.count > 1 ? "were" : "was"} outside the typical tour range. Everything else measured sat inside it.` };
    }
    const k = findKey(ql);
    if (k) {
      if (tier(k, ACTIVE) === "low") { call("get_indicator", { key: k }, t_get(k)); return { toolCalls: tc, refuse: true, answer: "That measurement isn't reliable enough from a single camera, so I can't assess it." }; }
      const r = call("get_indicator", { key: k }, t_get(k));
      const phr = r.in_tour_range ? IND(ACTIVE)[k].in : (r.value < r.pro_band[0] ? IND(ACTIVE)[k].lo : IND(ACTIVE)[k].hi);
      let s = `Your ${low(label(k, ACTIVE))} is ${r.value}${unit(k, ACTIVE)}`;
      s += r.in_tour_range ? `, inside the tour range (${r.pro_band[0]}–${r.pro_band[1]})${phr ? " — " + phr : ""}.` : `${phr ? " — " + phr : ""} (tour range ${r.pro_band[0]}–${r.pro_band[1]}).`;
      return { toolCalls: tc, answer: s };
    }
    call("list_indicators", {}, { count: t_list().count });
    return { toolCalls: tc, answer: "I can break down the measured parts of your swing — tempo, shoulder and hip turn, X-factor, posture, spine angle, head sway, and weight shift. Ask about any of those, list a few, or say “give me the biggest takeaways” for a summary." };
  }

  /* ---- grounding verifier (mirrors verify_chat_grounding), for the mock ---- */
  function verify(res) {
    const a = (res.answer || "").toLowerCase(); const v = [];
    const lc = new Set();
    (res.toolCalls || []).forEach(e => { const r = e.result; if (!r) return; if ((e.name === "get_indicator" || e.name === "compare_indicator") && (r.reliable === false || r.confidence_tier === "low")) lc.add(r.key); });
    lc.forEach(k => { if (IND(ACTIVE)[k] && a.includes(String(val(k, ACTIVE)))) v.push("low_confidence_leak"); });
    if (!res.refuse && /\byou should\b|\btry to\b|\bwork on\b|\bdrill\b|\bpractice\b/.test(a)) v.push("prescriptive");
    return { grounded: v.length === 0, violations: v };
  }

  /* ======= askChat seam: live backend if window.API_BASE set, else mock ======= */
  async function askChat(q) {
    if (window.API_BASE) {
      const hist = HIST[ACTIVE] || (HIST[ACTIVE] = []);
      const body = { clip_id: Number(ACTIVE), question: q, history: hist.slice() };
      if (COMPARE != null) body.compare_clip_id = Number(COMPARE);
      const r = await fetch(`${window.API_BASE}/chat`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
      if (!r.ok) throw new Error("http " + r.status);
      const d = await r.json();
      hist.push({ role: "user", content: q }, { role: "assistant", content: d.answer || "" });
      const tcs = (d.tool_log && d.tool_log.length) ? d.tool_log.map(e => ({ name: e.name, input: e.input, result: e.result })) : (d.tools_used || []).map(n => ({ name: n }));
      return { answer: d.answer, toolCalls: tcs, grounded: d.grounded, violations: d.violations, refuse: false };
    }
    const res = respond(q);
    await new Promise(r => setTimeout(r, 380 + Math.random() * 260));  // feel of "thinking"
    return res;
  }

  /* ================================ UI ================================ */
  const el = (t, c, txt) => { const e = document.createElement(t); if (c) e.className = c; if (txt != null) e.textContent = txt; return e; };
  const scroll = () => { dom.stream.scrollTop = dom.stream.scrollHeight; };
  function addUser(q) { const w = el("div", "chat-turn"); w.append(el("div", "who", "You"), el("div", "msg user", q)); dom.stream.append(w); scroll(); }
  function addTyping() { const t = el("div", "typing"); t.innerHTML = (window.API_BASE ? "Running the coach" : "Checking the measurements") + ' <span class="dot"></span><span class="dot"></span><span class="dot"></span>'; dom.stream.append(t); scroll(); return t; }
  const grade = res => (res.grounded !== undefined) ? { grounded: res.grounded, violations: res.violations || [] } : verify(res);
  function addBot(res) {
    const tcs = res.toolCalls || [];
    if (tcs.length) {
      const d = el("details", "trace"); const sum = el("summary", null, `${tcs.length} tool call${tcs.length > 1 ? "s" : ""} — click to inspect`);
      const bodyd = el("div", "trace-body");
      tcs.forEach(e => {
        const r = el("div", "tc");
        const argStr = e.input ? JSON.stringify(e.input).replace(/^{}$/, "") : "";
        let rs = e.result ? JSON.stringify(e.result) : "";
        if (rs.length > 340) rs = rs.slice(0, 340) + " … (" + rs.length + " chars)";
        const esc = s => s.replace(/[<>&]/g, c => ({ "<": "&lt;", ">": "&gt;", "&": "&amp;" }[c]));
        r.innerHTML = `<span class="call">${esc(e.name)}</span>(<span class="arg">${esc(argStr)}</span>)` + (rs ? `<span class="res">-&gt; ${esc(rs)}</span>` : "");
        bodyd.append(r);
      });
      d.append(sum, bodyd); dom.stream.append(d);
    }
    const w = el("div", "chat-turn"); w.append(el("div", "who", "MotionCaddie"));
    const m = el("div", "msg bot" + (res.refuse ? " refuse" : ""), res.answer);
    const g = grade(res); const gd = el("span", "grade" + (g.grounded ? "" : " warn"));
    gd.textContent = g.grounded ? "Grounded in fetched data" : "Grounding check: " + (g.violations || []).join(", ");
    m.append(document.createElement("br"), gd); w.append(m); dom.stream.append(w); scroll();
  }
  async function ask(q) {
    addUser(q); const typing = addTyping();
    let res;
    try { res = await askChat(q); }
    catch (e) { res = respond(q); res.answer += "  (couldn't reach the live coach — showing the offline read)"; }
    typing.remove(); addBot(res);
  }
  function greet() {
    const c = CLIPDATA[ACTIVE];
    const w = el("div", "chat-turn"); w.append(el("div", "who", "MotionCaddie"));
    w.append(el("div", "msg bot", `You're looking at ${c.name}'s swing (${c.club}, ${c.view}). Ask me anything about it — a metric like tempo or weight shift, the biggest takeaways, or pick a second swing to compare.`));
    dom.stream.append(w); scroll();
  }
  function populateCompare() {
    if (!dom.compare) return;
    dom.compare.innerHTML = '<option value="">— none —</option>';
    Object.keys(CLIPDATA).filter(c => c !== ACTIVE).forEach(c => { const o = document.createElement("option"); o.value = c; o.textContent = CLIPDATA[c].name; dom.compare.append(o); });
    dom.compare.value = COMPARE || "";
  }

  /* ---- public API ---- */
  return {
    init(nodes) {
      dom = nodes;
      dom.send.addEventListener("click", () => { const v = dom.input.value.trim(); if (v) { ask(v); dom.input.value = ""; } });
      dom.input.addEventListener("keydown", e => { if (e.key === "Enter") { const v = dom.input.value.trim(); if (v) { ask(v); dom.input.value = ""; } } });
      (dom.chips || []).forEach(c => c.addEventListener("click", () => ask(c.textContent.trim())));
      if (dom.compare) dom.compare.addEventListener("change", e => { COMPARE = e.target.value || null; });
      if (dom.mode) dom.mode.textContent = window.API_BASE ? "live · real coach" : "offline preview · in-page coach";
    },
    setLibrary(clips, metricsByClipId) {
      for (const clip of clips) { const rows = (metricsByClipId[clip.id] || {}).metrics; if (rows) buildFromMetrics(clip, rows); }
    },
    activate(clipId) {
      ACTIVE = String(clipId); COMPARE = null;
      if (!CLIPDATA[ACTIVE]) return;
      if (dom.stream) dom.stream.innerHTML = "";
      if (dom.active) dom.active.textContent = `${CLIPDATA[ACTIVE].name} · ${CLIPDATA[ACTIVE].view}`;
      populateCompare(); greet();
    },
  };
})();
