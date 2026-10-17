"use strict";
// Autoresearch UI. Three pages, routed by the URL hash:
//   #/                        recent projects
//   #/p/<id>/<view>[/<sel>]   a project's architecture, as baseline | evolution | current, linked to the loss chart
//   #/p/<id>/agents           the agent loop: design choice, prompts, start/stop, and every LLM call as it happens
// A project is a directory with train.py, results.tsv and runs/. Its optional architecture.json maps experiments
// to model components; without one, experiments are grouped by the planner's idea category. No dependencies.

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const POLL_MS = 4000;
const VIEWS = ["baseline", "evolution", "current"];
const VIEW_NOTE = {
  baseline: "The architecture the loop started from.",
  evolution: "Every idea tried, to the right of the component it changed. Solid green = kept.",
  current: "The architecture today, after every kept change.",
};

const S = {
  route: { page: "home" }, pid: null, cards: null,
  raw: "", data: null, m: null,          // project state and the model built from it
  agentsRaw: "", agents: null,           // agents page state
  detail: null,                          // agents page right pane: {kind: "prompt"|"trace", key}
  diffs: new Map(), timer: null, error: null,
};

// ------------------------------------------------------------------ formatting

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmtLoss = (v) => (v == null ? "—" : v.toFixed(4));
function fmtPct(p) {
  if (p == null || !isFinite(p)) return "—";
  const a = Math.abs(p);
  return (p < 0 ? "−" : p > 0 ? "+" : "±") + (a >= 1 ? a.toFixed(1) : a.toFixed(2)) + "%";
}
const pctOf = (d, ref) => (d == null || !ref ? null : (d / ref) * 100);
const tone = (p) => (p == null ? "" : p < 0 ? "good" : "worse");
const clip = (t, n) => { t = String(t ?? ""); return t.length > n ? t.slice(0, n - 1) + "…" : t; };
const humanize = (s) => String(s || "other").replace(/[_-]+/g, " ").replace(/^\w/, (c) => c.toUpperCase());
function ago(sec) {
  sec = Math.max(0, Math.round(sec));
  if (sec < 60) return sec + "s";
  if (sec < 3600) return Math.floor(sec / 60) + "m";
  if (sec < 86400) return Math.floor(sec / 3600) + "h " + Math.floor((sec % 3600) / 60) + "m";
  return Math.floor(sec / 86400) + "d";
}
const shortPath = (p) => String(p || "").replace(/^\/Users\/[^/]+/, "~");
const safeUrl = (u) => (/^https?:\/\//i.test(String(u)) ? String(u) : null);
const STATUS = { keep: "kept", discard: "discarded", crash: "crashed" };
const ICON = { keep: "✓", discard: "–", crash: "✕" };
const VERDICT = { validated: "✓", invalidated: "✕", inconclusive: "?", untested: "○" };
const GLYPH = `<svg class="glyph" viewBox="0 0 16 16" aria-hidden="true">${[0, 1, 2, 3].map((r) => [0, 1, 2, 3].map((c) =>
  `<rect x="${c * 4 + 0.5}" y="${r * 4 + 0.5}" width="3" height="3" rx="0.6" opacity="${(r + c) % 3 ? 0.55 : 1}"/>`).join("")).join("")}</svg>`;

async function getJSON(url, opts) {
  const r = await fetch(url, { cache: "no-store", ...opts });
  const j = await r.json().catch(() => null);
  if (!r.ok) throw new Error((j && j.error) || "HTTP " + r.status);
  return j;
}
const post = (url, body) => getJSON(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

// ------------------------------------------------------------------ routing

function parseRoute() {
  const p = location.hash.replace(/^#\/?/, "").split("/").filter(Boolean).map(decodeURIComponent);
  if (p[0] !== "p" || !p[1]) return { page: "home" };
  if (p[2] === "agents") return { page: "agents", pid: p[1] };
  return { page: "project", pid: p[1], view: VIEWS.includes(p[2]) ? p[2] : "evolution", sel: p[3] || null };
}
const projHref = (pid, view, sel) => `#/p/${pid}/${view || "evolution"}` + (sel ? "/" + encodeURIComponent(sel) : "");

function select(sel) {
  if (S.route.page !== "project") return;
  S.route.sel = sel || null;
  history.replaceState(null, "", projHref(S.pid, S.route.view, S.route.sel));
  renderArch(); renderSide(); renderChart();
}

async function onRoute() {
  const r = parseRoute();
  const samePid = r.pid && r.pid === S.pid;
  const samePage = r.page === S.route.page && samePid;
  S.route = r;
  clearInterval(S.timer);
  if (r.page === "home") {
    S.pid = null;
    renderHome();
    loadHome();
    S.timer = setInterval(loadHome, 10000);
    return;
  }
  if (!samePid) { S.raw = ""; S.data = null; S.m = null; S.agentsRaw = ""; S.agents = null; S.detail = null; S.diffs.clear(); }
  S.pid = r.pid;
  if (!samePage) renderShell();
  else if (r.page === "project") { renderArch(); renderSide(); renderChart(); }
  await poll(!samePid);
  S.timer = setInterval(() => poll(false), POLL_MS);
}

async function poll(open) {
  try {
    // Redraw only when something other than the server clock changed, so clicks and scroll positions survive polls.
    const unclocked = (t) => t.replace(/"server_time":\s*[\d.eE+-]+/, "");
    const raw = await (await fetch(`/api/p/${S.pid}` + (open ? "?open=1" : ""), { cache: "no-store" })).text();
    S.error = null;
    const data = JSON.parse(raw);
    if (unclocked(raw) !== S.raw) {
      S.raw = unclocked(raw);
      S.data = data;
      S.m = buildModel(S.data);
      if (S.route.page === "project") { renderArch(); renderSide(); renderChart(); }
    } else S.data.server_time = data.server_time;
    if (S.route.page === "agents") {
      const a = await (await fetch(`/api/p/${S.pid}/agents`, { cache: "no-store" })).text();
      if (unclocked(a) !== S.agentsRaw) { S.agentsRaw = unclocked(a); S.agents = JSON.parse(a); renderAgents(); }
    }
  } catch (e) {
    S.error = "Can't reach the server (" + e.message + ")";
  }
  renderHeader();
}

// ------------------------------------------------------------------ model

// Turns the server state into components, each with its baseline, the experiments that changed it (in order),
// its current state, and the queued ideas aimed at it.
function buildModel(data) {
  const arch = data.architecture || {};
  const changes = arch.changes || {};
  const exps = data.experiments.map((e) => ({ ...e, n: e.idx }));
  const comps = [], byId = new Map();
  const addComp = (c) => { comps.push(c); byId.set(c.id, c); return c; };
  for (const c of arch.components || []) {
    addComp({ id: c.id, name: c.name || humanize(c.id), group: c.group || "components", baseline: c.baseline || {}, steps: [], next: [] });
  }
  const groups = (arch.groups || []).map((g) => ({ ...g, comps: [] }));
  const gById = new Map(groups.map((g) => [g.id, g]));

  let best = null, bestExp = null;
  for (const e of exps) {
    e.bestExpBefore = bestExp;
    if (e.n > 0) {
      const ch = changes[e.n] || {};
      const cid = ch.component || (e.idea && e.idea.category) || "other";
      const c = byId.get(cid) || addComp({ id: cid, name: humanize(cid), group: "components", baseline: { label: "baseline" }, steps: [], next: [] });
      e.comp = c;
      e.change = ch;
      e.label = ch.label || (e.idea && e.idea.title) || clip(e.description, 32);
      c.steps.push(e);
    }
    if (e.status === "keep" && e.val_loss != null && (best == null || e.val_loss < best)) { best = e.val_loss; bestExp = e; }
    e.pct = pctOf(e.delta_best, e.best_before);
    e.verdict = verdictOf(e);
  }
  for (const c of comps) {
    let cur = null;
    for (const e of c.steps) {
      e.before = cur ? (cur.change.after || cur.label) : c.baseline.label;
      if (e.status === "keep") cur = e;
    }
    c.current = cur;
    c.currentLabel = cur ? (cur.change.after || cur.label) : c.baseline.label;
    c.currentCost = cur ? (cur.change.cost || c.baseline.cost) : c.baseline.cost;
    if (!gById.has(c.group)) { const g = { id: c.group, name: humanize(c.group), comps: [] }; groups.push(g); gById.set(g.id, g); }
    gById.get(c.group).comps.push(c);
  }

  // Queued ideas: status "new", not already run, attached to a component when the map (or the category) names one.
  const tried = new Set(arch.tried || []);
  const nextMap = arch.next || {};
  const ideas = (data.ideas || []).filter((i) => (i.status || "new") === "new" && !tried.has(i.id) && (i.title || i.change))
    .map((i) => ({ ...i, title: i.title || clip(i.change, 40) }));
  for (const i of ideas) {
    const c = byId.get(nextMap[i.id]) || (!arch.components && byId.get(i.category));
    i.comp = c || null;
    if (c) c.next.push(i);
  }

  const hyps = (arch.hypotheses || []).length ? arch.hypotheses : exps.filter((e) => e.n > 0 && e.verdict).map((e) => ({
    id: "e" + e.n, status: e.verdict, evidence: [e.n],
    text: `${e.label}${e.idea && e.idea.prediction && e.idea.prediction.delta_pct != null ? ` lowers ${metricName(arch)} by ${Math.abs(e.idea.prediction.delta_pct)}%` : ""}`,
  }));
  return { arch, exps, comps, groups: groups.filter((g) => g.comps.length), byId, ideas, hyps, best, bestExp, hasMap: !!arch.components };
}
const metricName = (arch) => (arch.metric && arch.metric.name) || "val_loss";

function verdictOf(e) {
  if (e.n === 0) return null;
  if (e.status === "crash") return "untested";
  const p = e.idea && e.idea.prediction && e.idea.prediction.delta_pct;
  const a = (e.idea && e.idea.outcome && e.idea.outcome.delta_pct) ?? e.pct;
  if (p == null || a == null) return null;
  return (p < 0) === (a < 0) ? "validated" : "invalidated";
}

// ------------------------------------------------------------------ home

async function loadHome() {
  try { S.cards = await getJSON("/api/projects"); S.error = null; } catch (e) { S.error = e.message; }
  if (S.route.page === "home") renderHomeTiles();
}

function renderHome() {
  $("#app").className = "page-home";
  $("#app").innerHTML = `
    <main class="home">
      <div class="home-brand">${GLYPH}<span>autoresearch</span></div>
      <div class="tiles" id="tiles"></div>
      <form class="add-form" id="add-form" hidden>
        <input id="add-path" placeholder="/path/to/project" spellcheck="false" autocomplete="off">
        <button type="submit">Open</button>
        <p class="add-note" id="add-note">A project is a folder with train.py, results.tsv and runs/.</p>
      </form>
    </main>`;
  $("#add-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    try {
      const r = await post("/api/projects", { path: $("#add-path").value.trim() });
      location.hash = projHref(r.id);
    } catch (err) { $("#add-note").textContent = err.message; $("#add-note").classList.add("bad"); }
  });
  renderHomeTiles();
}

function sparkIcon(curve, baseline) {
  const pts = curve.map((v, i) => [i, v]).filter((p) => p[1] != null);
  if (pts.length < 2) return `<svg viewBox="0 0 64 64" aria-hidden="true">${GLYPH.replace(/^<svg[^>]*>|<\/svg>$/g, "")}</svg>`;
  const vs = pts.map((p) => p[1]), lo = Math.min(...vs), hi = Math.max(...vs, baseline ?? -Infinity);
  const x = (i) => 10 + (44 * i) / Math.max(1, curve.length - 1);
  const y = (v) => (hi === lo ? 32 : 50 - (36 * (hi - v)) / (hi - lo));  // lower loss sits higher
  let d = `M${x(pts[0][0]).toFixed(1)},${y(pts[0][1]).toFixed(1)}`;
  for (let k = 1; k < pts.length; k++) d += `H${x(pts[k][0]).toFixed(1)}V${y(pts[k][1]).toFixed(1)}`;
  return `<svg viewBox="0 0 64 64" aria-hidden="true"><path class="spark" d="${d}"/><circle class="spark-end" cx="${x(pts.at(-1)[0])}" cy="${y(pts.at(-1)[1])}" r="2.6"/></svg>`;
}

function renderHomeTiles() {
  const el = $("#tiles");
  if (!el) return;
  const cards = S.cards || [];
  el.innerHTML = cards.map((c) => {
    const p = c.best != null && c.baseline ? ((c.best - c.baseline) / c.baseline) * 100 : null;
    return `<a class="tile" href="${projHref(c.id)}" title="${esc(c.path)}">
      <span class="tile-icon">${sparkIcon(c.curve, c.baseline)}${c.live ? '<span class="live-dot"></span>' : ""}</span>
      <span class="tile-name">${esc(c.name)}</span>
      <span class="tile-sub">${c.total} runs${p != null ? ` · <b class="${tone(p)}">${fmtPct(p)}</b>` : ""}</span>
      <span class="tile-path">${esc(shortPath(c.path).split("/").slice(-2).join("/"))}</span>
    </a>`;
  }).join("") + `<button type="button" class="tile add" id="add-tile" title="Open a project folder">
      <span class="tile-icon"><svg viewBox="0 0 64 64" aria-hidden="true"><path d="M32 22v20M22 32h20"/></svg></span>
      <span class="tile-name">Open folder</span></button>`;
  $("#add-tile").onclick = () => { $("#add-form").hidden = false; $("#add-path").focus(); };
  if (S.error && !cards.length) el.insertAdjacentHTML("beforebegin", `<p class="add-note bad">${esc(S.error)}</p>`);
}

// ------------------------------------------------------------------ project shell + header

function renderShell() {
  const agents = S.route.page === "agents";
  $("#app").className = agents ? "page-agents" : "page-project";
  $("#app").innerHTML = `
    <header class="bar" id="bar"></header>
    ${agents ? `<main class="agents" id="agents"><p class="muted pad">Loading…</p></main>` : `
    <main class="proj">
      <section class="canvas panel">
        <div class="arch" id="arch"></div>
        <div class="chart-wrap"><div class="chart" id="chart"></div></div>
      </section>
      <aside class="side panel" id="side" aria-live="polite"></aside>
    </main>`}`;
  renderHeader();
  // Switching pages within a project keeps the loaded state, so draw it now rather than on the next change.
  if (agents) renderAgents();
  else { renderArch(); renderSide(); new ResizeObserver(() => renderChart()).observe($("#chart")); }
}

function agentStatus() {
  const d = S.data;
  if (!d) return { text: "Loading…", cls: "" };
  const st = d.harness && d.harness.status, live = d.live;
  if (live && live.pct != null && !live.finished) {
    return { text: `Training #${d.experiments.length} · ${Math.round(live.pct)}%`, cls: "on" };
  }
  if (st && st.phase && st.phase !== "stopped") {
    return { text: `${humanize(st.phase)} #${st.n}${st.title ? " · " + clip(st.title, 34) : ""}`, cls: "on" };
  }
  const last = d.experiments.reduce((m, e) => Math.max(m, e.started_at || 0), 0);
  return { text: last ? `Idle · last run ${ago(d.server_time - last)} ago` : "Idle", cls: "" };
}

function renderHeader() {
  const el = $("#bar");
  if (!el) return;
  const d = S.data, m = S.m, s = d && d.summary;
  const p = s && s.best_val_loss != null && s.baseline_val_loss ? ((s.best_val_loss - s.baseline_val_loss) / s.baseline_val_loss) * 100 : null;
  const st = S.error ? { text: S.error, cls: "bad" } : agentStatus();
  const view = S.route.view || "evolution";
  el.innerHTML = `
    <a class="home-link" href="#/" title="All projects">${GLYPH}</a>
    <div class="ptitle"><b>${esc(d ? d.name : "…")}</b><span class="mono">${esc(shortPath(d && d.root))}</span></div>
    ${s ? `<div class="kpi mono" title="${esc(metricName((m && m.arch) || {}))}: baseline → best">${fmtLoss(s.baseline_val_loss)}<span class="arrow">→</span><b>${fmtLoss(s.best_val_loss)}</b>
      <span class="${tone(p)}">${fmtPct(p)}</span></div>
      <div class="counts">${s.total} runs · ${s.counts.keep || 0} kept</div>` : ""}
    <div class="status ${st.cls}" title="${esc(st.text)}"><span class="dot"></span><span>${esc(st.text)}</span></div>
    <nav class="seg" aria-label="Page">
      <a href="${projHref(S.pid, view)}" class="${S.route.page === "project" ? "on" : ""}">Architecture</a>
      <a href="#/p/${S.pid}/agents" class="${S.route.page === "agents" ? "on" : ""}">Agents</a>
    </nav>
    <button type="button" class="icon-btn" id="theme" title="Light or dark (T)" aria-label="Toggle theme">
      <svg viewBox="0 0 20 20" aria-hidden="true"><circle cx="10" cy="10" r="7.25" fill="none" stroke="currentColor" stroke-width="1.5"/><path d="M10 2.75a7.25 7.25 0 0 1 0 14.5z" fill="currentColor"/></svg>
    </button>`;
  $("#theme").onclick = toggleTheme;
}

// ------------------------------------------------------------------ architecture

function renderArch() {
  const el = $("#arch");
  if (!el) return;
  const m = S.m, v = S.route.view;
  if (!m) { el.innerHTML = `<p class="muted pad">Loading…</p>`; return; }
  const scroll = {};
  $$(".chain", el).forEach((c) => (scroll[c.dataset.comp] = c.scrollLeft));
  const keepTop = el.scrollTop;
  el.innerHTML = `
    <div class="arch-head">
      <nav class="seg small" aria-label="Architecture view">${VIEWS.map((k, i) =>
        `<a href="${projHref(S.pid, k, S.route.sel)}" class="${k === v ? "on" : ""}" title="${esc(VIEW_NOTE[k])} (${i + 1})">${humanize(k)}</a>`).join("")}</nav>
      <p class="muted">${esc(VIEW_NOTE[v])}${m.hasMap ? "" : " No architecture.json yet, so experiments are grouped by idea category."}</p>
      ${v === "evolution" ? `<ul class="legend"><li class="keep">kept</li><li class="discard">discarded</li><li class="crash">crashed</li><li class="next">queued</li></ul>` : ""}
    </div>
    ${m.groups.map((g) => `
      <section class="grp ${g.flow && v !== "evolution" ? "flow" : ""}">
        <h3>${esc(g.name)}</h3>
        ${g.comps.map((c) => lane(c, v)).join("")}
      </section>`).join("")}`;
  $$(".chain", el).forEach((c) => { if (scroll[c.dataset.comp]) c.scrollLeft = scroll[c.dataset.comp]; });
  el.scrollTop = keepTop;
  markSelected();
}

function lane(c, v) {
  let nodes;
  if (v === "baseline") nodes = [baseNode(c, true)];
  else if (v === "current") nodes = [currentNode(c)];
  else nodes = [baseNode(c, false), ...c.steps.map((e) => stepNode(e, c)), ...c.next.map(nextNode)];
  const tried = c.steps.length;
  return `<div class="lane" data-comp="${esc(c.id)}">
    <button type="button" class="lane-name" data-sel="c:${esc(c.id)}" data-comp="${esc(c.id)}">${esc(c.name)}${
      v === "evolution" && tried ? `<small>${tried} tried</small>` : ""}${v === "current" && c.current ? `<small class="good">changed</small>` : ""}</button>
    <div class="chain" data-comp="${esc(c.id)}">${nodes.join("")}</div>
  </div>`;
}

function baseNode(c, wide) {
  const live = !c.current;
  return `<button type="button" class="node base${wide ? " wide" : ""}${live ? " live" : ""}" data-sel="c:${esc(c.id)}" data-comp="${esc(c.id)}">
    <span class="tag">baseline</span><span class="lbl">${esc(c.baseline.label || "baseline")}</span>
    ${wide && c.baseline.cost ? `<span class="cost mono">${esc(c.baseline.cost)}</span>` : ""}</button>`;
}

function stepNode(e, c) {
  const live = c.current === e;
  const d = e.status === "crash" ? `<span class="d worse">crashed</span>` : `<span class="d ${tone(e.pct)}">${fmtPct(e.pct)}</span>`;
  return `<button type="button" class="node ${esc(e.status)}${live ? " live" : ""}" data-sel="e${e.n}" data-comp="${esc(c.id)}">
    <span class="tag">#${e.n} <i>${ICON[e.status] || ""}</i></span><span class="lbl">${esc(clip(e.label, 34))}</span>${d}</button>`;
}

function nextNode(i) {
  return `<button type="button" class="node next" data-sel="i:${esc(i.id)}"><span class="tag">queued</span><span class="lbl">${esc(clip(i.title, 34))}</span></button>`;
}

function currentNode(c) {
  const e = c.current;
  return `<button type="button" class="node wide live ${e ? "keep" : "base"}" data-sel="${e ? "e" + e.n : "c:" + esc(c.id)}" data-comp="${esc(c.id)}">
    <span class="tag">${e ? `changed in #${e.n}` : "unchanged"}</span>
    <span class="lbl">${esc(c.currentLabel || "baseline")}</span>
    ${c.currentCost ? `<span class="cost mono">${esc(c.currentCost)}</span>` : ""}
    ${e ? `<span class="was">was ${esc(c.baseline.label)}</span>` : ""}</button>`;
}

function markSelected() {
  const sel = S.route.sel;
  $$("[data-sel].sel").forEach((n) => n.classList.remove("sel"));
  if (sel) $$(`[data-sel="${CSS.escape(sel)}"]`).forEach((n) => n.classList.add("sel"));
  const node = sel && $(`#arch .node[data-sel="${CSS.escape(sel)}"]`);
  if (node) {
    const chain = node.closest(".chain"), r = node.getBoundingClientRect(), cr = chain.getBoundingClientRect();
    if (r.right > cr.right || r.left < cr.left) chain.scrollLeft += r.left - cr.left - 40;
  }
}

// Hovering anything with data-sel highlights the same experiment in the architecture and in the chart.
function highlight(sel, on) {
  if (!sel) return;
  let q = `[data-sel="${CSS.escape(sel)}"]`;
  if (sel.startsWith("c:")) q += `, .pt[data-comp="${CSS.escape(sel.slice(2))}"]`;
  $$(q).forEach((n) => n.classList.toggle("hl", on));
  document.body.classList.toggle("hovering", on);
}

// ------------------------------------------------------------------ chart

function renderChart() {
  const box = $("#chart");
  if (!box || !S.m) return;
  const exps = S.m.exps, s = S.data.summary, v = S.route.view, sel = S.route.sel;
  const W = Math.max(240, box.clientWidth), H = Math.max(120, box.clientHeight);
  if (!exps.length) { box.innerHTML = `<p class="muted pad">No results yet.</p>`; return; }
  const L = 52, R = 14, T = 12, B = 22;
  const vals = exps.filter((e) => e.val_loss != null).map((e) => e.val_loss);
  const base = s.baseline_val_loss ?? Math.max(...vals);
  const lo = Math.min(...vals);
  const cap = base + Math.max((base - lo) * 0.8, base * 0.002);  // worse runs above this sit on the top edge
  const yLo = lo - (cap - lo) * 0.08, yHi = cap;
  const x = (i) => L + ((W - L - R) * i) / Math.max(1, exps.length - 1);
  const y = (val) => T + ((H - T - B) * (yHi - Math.min(val, yHi))) / (yHi - yLo);
  const out = [];
  const ticks = 4;
  for (let k = 0; k <= ticks; k++) {
    const val = yLo + ((yHi - yLo) * k) / ticks;
    out.push(`<line class="ax-grid" x1="${L}" x2="${W - R}" y1="${y(val)}" y2="${y(val)}"/>`,
      `<text class="ax-label" x="${L - 6}" y="${y(val)}" dy="0.35em" text-anchor="end">${val.toFixed(4)}</text>`);
  }
  const step = Math.max(1, Math.ceil(exps.length / 14));
  exps.forEach((e, i) => { if (i % step === 0 || i === exps.length - 1) out.push(`<text class="ax-label" x="${x(i)}" y="${H - 6}" text-anchor="middle">${e.n}</text>`); });
  let d = "", prev = null;
  exps.forEach((e, i) => {
    if (e.best_after == null) return;
    d += prev == null ? `M${x(i)},${y(e.best_after)}` : `H${x(i)}V${y(e.best_after)}`;
    prev = e.best_after;
  });
  if (d) out.push(`<path class="best-line" d="${d}H${W - R}"/>`);
  if (sel && sel.startsWith("e")) {
    const i = exps.findIndex((e) => "e" + e.n === sel);
    if (i >= 0) out.push(`<line class="sel-line" x1="${x(i)}" x2="${x(i)}" y1="${T}" y2="${H - B}"/>`);
  }
  for (const [i, e] of exps.entries()) {
    const comp = e.comp ? e.comp.id : "";
    const dim = (v === "baseline" && e.n !== 0) || (v === "current" && e.status !== "keep") ||
      (sel && sel.startsWith("c:") && e.n !== 0 && comp !== sel.slice(2));
    const cls = `pt ${e.status}${dim ? " dim" : ""}${sel === "e" + e.n ? " sel" : ""}`;
    const X = x(i);
    let mark;
    if (e.val_loss == null) mark = `<path d="M${X - 4},${T + 2}l8,8M${X + 4},${T + 2}l-8,8"/>`;
    else if (e.val_loss > yHi) mark = `<path d="M${X},${T}l5,8h-10z"/>`;
    else mark = `<circle cx="${X}" cy="${y(e.val_loss)}" r="${e.status === "keep" ? 5 : 3.8}"/>`;
    const cy = e.val_loss == null || e.val_loss > yHi ? T + 6 : y(e.val_loss);
    out.push(`<g class="${cls}" data-sel="e${e.n}" data-comp="${esc(comp)}"><circle class="hit" cx="${X}" cy="${cy}" r="11"/><circle class="ring" cx="${X}" cy="${cy}" r="9"/>${mark}</g>`);
  }
  box.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="${esc(metricName(S.m.arch))} by experiment">${out.join("")}</svg>`;
}

function showTip(target) {
  const sel = target.dataset.sel, tip = $("#tip");
  const e = sel && sel.startsWith("e") && S.m && S.m.exps.find((x) => "e" + x.n === sel);
  if (!e) return;
  tip.innerHTML = `<b>#${e.n}</b> · ${esc(STATUS[e.status] || e.status)}${e.comp ? " · " + esc(e.comp.name) : ""}
    <div>${esc(e.n === 0 ? "baseline" : e.label)}</div>
    <div class="t-sub mono">${fmtLoss(e.val_loss)}${e.n && e.pct != null ? " · " + fmtPct(e.pct) + " vs best" : ""}</div>`;
  tip.hidden = false;
  const r = target.getBoundingClientRect(), tw = tip.offsetWidth, th = tip.offsetHeight;
  let tx = r.left + r.width / 2 - tw / 2, ty = r.top - th - 8;
  if (ty < 4) ty = r.bottom + 8;
  tip.style.left = Math.max(4, Math.min(innerWidth - tw - 4, tx)) + "px";
  tip.style.top = ty + "px";
}

// ------------------------------------------------------------------ side panel

function renderSide() {
  const el = $("#side");
  if (!el || !S.m) return;
  const sel = S.route.sel, m = S.m;
  let html;
  if (sel && sel.startsWith("e")) html = expDetail(m.exps.find((e) => "e" + e.n === sel));
  else if (sel && sel.startsWith("c:")) html = compDetail(m.byId.get(sel.slice(2)));
  else if (sel && sel.startsWith("i:")) html = ideaDetail(m.ideas.find((i) => i.id === sel.slice(2)));
  const keep = el.dataset.sel === String(sel) ? el.scrollTop : 0;  // a live update keeps the reader's place
  el.innerHTML = html || overview();
  el.dataset.sel = String(sel);
  el.scrollTop = keep;
}

const backBtn = `<button type="button" class="back" data-sel="" title="Back to the overview (Esc)">← Overview</button>`;
const refChips = (ns) => ns.map((n) => `<button type="button" class="chip" data-sel="e${n}">#${n}</button>`).join("");

function overview() {
  const m = S.m, arch = m.arch, v = S.route.view;
  const lessons = (S.data.harness && S.data.harness.lessons) || [];
  const baseExp = m.exps[0], bestExp = m.bestExp;
  const samples = v === "current" && baseExp && bestExp && baseExp.files["samples.png"] && bestExp.files["samples.png"] && bestExp !== baseExp
    ? `<div class="sec"><h4>Baseline vs now</h4><div class="pair">
        <figure><img src="${esc(baseExp.files["samples.png"])}" alt="baseline samples"><figcaption>#0 · ${fmtLoss(baseExp.val_loss)}</figcaption></figure>
        <figure><img src="${esc(bestExp.files["samples.png"])}" alt="best samples"><figcaption>#${bestExp.n} · ${fmtLoss(bestExp.val_loss)}</figcaption></figure></div></div>` : "";
  const order = { validated: 0, invalidated: 1, inconclusive: 2, untested: 3 };
  const hyps = [...m.hyps].sort((a, b) => (order[a.status] ?? 9) - (order[b.status] ?? 9));
  return `
    ${arch.summary ? `<p class="summary">${esc(arch.summary)}</p>` : ""}
    ${samples}
    <div class="sec"><h4>Hypotheses <span class="muted">${hyps.filter((h) => h.status === "validated").length} validated · ${hyps.filter((h) => h.status === "invalidated").length} invalidated</span></h4>
      <ul class="hyps">${hyps.map((h) => `<li class="${esc(h.status)}"><span class="hs" title="${esc(h.status)}">${VERDICT[h.status] || "?"}</span>
        <span>${esc(h.text)}</span><span class="ev">${refChips(h.evidence || [])}</span></li>`).join("") || `<li class="muted">None yet.</li>`}</ul></div>
    <div class="sec"><h4>Next up <span class="muted">${m.ideas.length} queued</span></h4>
      <ul class="ideas">${m.ideas.slice(0, 8).map((i) => `<li><button type="button" data-sel="i:${esc(i.id)}">${esc(i.title)}</button>
        <span class="muted">${esc(i.comp ? i.comp.name : humanize(i.category))}</span></li>`).join("") || `<li class="muted">The idea bank is empty.</li>`}</ul></div>
    ${lessons.length ? `<details class="sec" ${lessons.length <= 4 ? "open" : ""}><summary><h4>Theories <span class="muted">${lessons.length} from the playbook</span></h4></summary>
      <ul class="lessons">${lessons.map((l) => `<li>${lessonHtml(l.text)}</li>`).join("")}</ul></details>` : ""}`;
}

// "(exp 5, 9)" and "exp 6" in playbook text become clickable chips.
function lessonHtml(text) {
  return esc(text).replace(/\(exps? ([\d,\s]+)(?:,\s*[a-z ]+)?\)/gi, (all, list) => refChips(list.split(/[,\s]+/).filter(Boolean)))
    .replace(/\bexp (\d+)\b/g, (all, n) => refChips([n]));
}

function expDetail(e) {
  if (!e) return null;
  if (e.n === 0) {
    return `${backBtn}<div class="d-head"><span class="tag">#0</span><span class="pill keep">baseline</span></div>
      <h3>Baseline</h3>${metricsRow(e)}${samplesFig(e)}
      <div class="sec"><h4>Components</h4><ul class="ideas">${S.m.comps.map((c) =>
        `<li><button type="button" data-sel="c:${esc(c.id)}">${esc(c.name)}</button><span class="muted">${esc(c.baseline.label || "")}</span></li>`).join("")}</ul></div>`;
  }
  const idea = e.idea || {}, out = idea.outcome || {}, pred = idea.prediction || {};
  const actual = out.delta_pct ?? e.pct;
  const diffUrl = e.files["change.diff"];
  const lessons = ((S.data.harness && S.data.harness.lessons) || []).filter((l) => new RegExp(`\\bexps? [\\d,\\s]*\\b${e.n}\\b`).test(l.text));
  return `${backBtn}
    <div class="d-head"><span class="tag">#${e.n}</span><span class="pill ${esc(e.status)}">${esc(STATUS[e.status] || e.status)}</span>
      ${e.comp ? `<button type="button" class="chip" data-sel="c:${esc(e.comp.id)}">${esc(e.comp.name)}</button>` : ""}</div>
    <h3>${esc(e.label)}</h3>
    <p class="muted">${esc(e.description)}</p>
    ${metricsRow(e)}
    <div class="sec"><h4>Change</h4>
      <div class="ba"><span class="before">${esc(e.before || "baseline")}</span><span class="arrow">→</span><span class="after">${esc(e.change.after || e.label)}</span></div>
      ${e.change.cost ? `<p class="cost mono">${esc(e.change.cost)}</p>` : ""}
      ${e.change.note ? `<p>${esc(e.change.note)}</p>` : ""}</div>
    ${idea.rationale || pred.delta_pct != null ? `<div class="sec"><h4>Hypothesis ${e.verdict ? `<span class="verdict ${e.verdict}">${VERDICT[e.verdict]} ${e.verdict}</span>` : ""}</h4>
      ${idea.rationale ? `<p class="clamp">${esc(idea.rationale)}</p>` : ""}
      ${pred.delta_pct != null ? `<p class="pred mono">predicted <b class="${tone(pred.delta_pct)}">${fmtPct(pred.delta_pct)}</b> · actual <b class="${tone(actual)}">${e.status === "crash" ? "crashed" : fmtPct(actual)}</b>${pred.confidence ? ` · ${esc(pred.confidence)} confidence` : ""}</p>` : ""}</div>` : ""}
    ${out.prediction_check || lessons.length ? `<div class="sec"><h4>Why</h4>
      ${out.prediction_check ? `<p>${esc(out.prediction_check)}</p>` : ""}
      ${lessons.map((l) => `<p class="lesson">${lessonHtml(l.text)}</p>`).join("")}</div>` : ""}
    ${diffUrl ? `<details class="sec code" data-url="${esc(diffUrl)}"><summary><h4>Code <span class="mono"><span class="good">+${e.diff ? e.diff.add : "?"}</span> <span class="worse">−${e.diff ? e.diff.del : "?"}</span> lines</span></h4></summary><div class="diff"></div></details>` : ""}
    ${samplesFig(e)}`;
}

function metricsRow(e) {
  const prev = e.bestExpBefore && e.bestExpBefore.metrics, cur = e.metrics || {};
  const delta = (a, b, f) => (a != null && b != null && a !== b ? ` <small class="${a > b ? "up" : "down"}">${a > b ? "+" : "−"}${f(Math.abs(a - b))}</small>` : "");
  return `<div class="metrics">
    <div><span>${esc(metricName(S.m.arch))}</span><b class="mono">${fmtLoss(e.val_loss)}</b></div>
    ${e.n ? `<div><span>vs best before</span><b class="mono ${tone(e.pct)}">${e.status === "crash" ? "—" : fmtPct(e.pct)}</b></div>` : ""}
    ${cur.num_steps != null ? `<div><span>steps</span><b class="mono">${cur.num_steps}${prev ? delta(cur.num_steps, prev.num_steps, (x) => x) : ""}</b></div>` : ""}
    ${cur.num_params_M != null ? `<div><span>params</span><b class="mono">${cur.num_params_M}M${prev ? delta(cur.num_params_M, prev.num_params_M, (x) => x.toFixed(2) + "M") : ""}</b></div>` : ""}
  </div>`;
}

const samplesFig = (e) => (e.files && e.files["samples.png"] ? `<figure class="samples"><img src="${esc(e.files["samples.png"])}" alt="samples from run #${e.n}" loading="lazy">
  <figcaption>${esc((S.data.prompt && `“${S.data.prompt}”`) || "samples")} · fixed seeds</figcaption></figure>` : "");

function compDetail(c) {
  if (!c) return null;
  const b = c.baseline || {};
  return `${backBtn}<div class="d-head"><span class="tag">component</span>${c.current ? `<span class="pill keep">changed</span>` : `<span class="pill">unchanged</span>`}</div>
    <h3>${esc(c.name)}</h3>
    <div class="sec"><h4>Baseline</h4><p><b>${esc(b.label || "baseline")}</b></p>${b.detail ? `<p>${esc(b.detail)}</p>` : ""}
      ${b.cost || b.params ? `<p class="cost mono">${esc([b.cost, b.params && "params " + b.params].filter(Boolean).join(" · "))}</p>` : ""}</div>
    ${c.current ? `<div class="sec"><h4>Now</h4><div class="ba"><span class="before">${esc(b.label)}</span><span class="arrow">→</span><span class="after">${esc(c.currentLabel)}</span></div>
      <p class="muted">since <button type="button" class="chip" data-sel="e${c.current.n}">#${c.current.n}</button></p></div>` : ""}
    <div class="sec"><h4>Tried <span class="muted">${c.steps.length}</span></h4>
      <ul class="steps">${c.steps.map((e) => `<li class="${esc(e.status)}"><button type="button" data-sel="e${e.n}"><span class="tag">#${e.n} ${ICON[e.status] || ""}</span>${esc(e.label)}</button>
        <span class="mono ${tone(e.pct)}">${e.status === "crash" ? "crashed" : fmtPct(e.pct)}</span></li>`).join("") || `<li class="muted">Nothing yet.</li>`}</ul></div>
    ${c.next.length ? `<div class="sec"><h4>Queued</h4><ul class="ideas">${c.next.map((i) => `<li><button type="button" data-sel="i:${esc(i.id)}">${esc(i.title)}</button></li>`).join("")}</ul></div>` : ""}`;
}

function ideaDetail(i) {
  if (!i) return null;
  const links = (i.sources || []).map(safeUrl).filter(Boolean);
  return `${backBtn}<div class="d-head"><span class="tag">${esc(i.id)}</span><span class="pill next">queued</span>
      ${i.comp ? `<button type="button" class="chip" data-sel="c:${esc(i.comp.id)}">${esc(i.comp.name)}</button>` : ""}</div>
    <h3>${esc(i.title)}</h3>
    ${i.change ? `<div class="sec"><h4>Change</h4><p>${esc(i.change)}</p></div>` : ""}
    ${i.mechanism ? `<div class="sec"><h4>Why it might work</h4><p>${esc(i.mechanism)}</p></div>` : ""}
    ${i.mlx_risk ? `<p class="muted">Implementation risk: ${esc(i.mlx_risk)}</p>` : ""}
    ${links.length ? `<div class="sec"><h4>Sources</h4><ul class="links">${links.map((u) => `<li><a href="${esc(u)}" target="_blank" rel="noopener">${esc(clip(u.replace(/^https?:\/\//, ""), 48))}</a></li>`).join("")}</ul></div>` : ""}`;
}

async function loadDiff(details) {
  const url = details.dataset.url, slot = $(".diff", details);
  if (!url || slot.dataset.done) return;
  slot.dataset.done = "1";
  try {
    if (!S.diffs.has(url)) S.diffs.set(url, await (await fetch(url)).text());
    slot.innerHTML = S.diffs.get(url).split("\n").filter((l) => !/^(diff --git|index |--- |\+\+\+ )/.test(l)).map((l) =>
      `<div class="${l.startsWith("+") ? "add" : l.startsWith("-") ? "del" : l.startsWith("@@") ? "hunk" : ""}">${esc(l) || " "}</div>`).join("");
  } catch (e) { slot.textContent = "Couldn't load the diff."; }
}

// ------------------------------------------------------------------ agents page

function renderAgents() {
  const el = $("#agents"), a = S.agents;
  if (!el || !a) return;
  if (!$("#ag-detail")) {
    el.innerHTML = `
      <section class="panel ag-col" id="ag-designs"></section>
      <section class="panel ag-col" id="ag-live"></section>
      <section class="panel ag-col" id="ag-detail"></section>`;
    renderDetail();
  }
  const status = a.status && a.status.phase && a.status.phase !== "stopped" ? a.status : null;
  const busy = a.running || !!status;
  $("#ag-designs").innerHTML = `
    <h3>Agent design</h3>
    <p class="muted">${a.managed ? "Each design runs in its own workspace, so they never share code, memory or context." : "This project isn't under this repo's workspaces/, so it's view-only here."}</p>
    ${a.designs.map((d) => {
      const mine = d.id === a.design;
      return `<div class="dcard${mine ? " on" : ""}">
        <div class="dname"><b>${esc(d.name)}</b><span class="tag">${esc(d.id)}</span>${mine ? `<span class="pill keep">this project</span>` : ""}</div>
        <p>${esc(d.about)}</p>
        <ol class="roles">${d.roles.map((r) => `<li>
          <button type="button" class="role${S.detail && S.detail.key === r.prompt ? " on" : ""}" data-prompt="${esc(r.prompt || "")}" ${r.prompt ? "" : "disabled"} title="Edit the ${esc(r.role)} prompt">${esc(r.role)}</button>
          <span class="mono model" title="${esc(r.model)}">${esc(String(r.model).split("/").pop())}</span>
          <small>${esc(r.about)}</small></li>`).join("")}</ol>
        ${a.managed ? `<div class="dact">${mine
          ? (busy ? `<button type="button" data-act="stop" ${a.stop_requested ? "disabled" : ""}>${a.stop_requested ? "Stopping after this experiment…" : "Stop after this experiment"}</button>
                     ${a.running ? `<button type="button" data-act="kill" class="danger">Stop now</button>` : ""}`
                  : `<label>until <input type="time" id="until-${esc(d.id)}"></label><button type="button" class="primary" data-act="start" data-design="${esc(d.id)}">Start</button>`)
          : `${d.project ? `<a href="#/p/${esc(d.project)}/agents">Open its workspace</a>` : ""}
             <label>until <input type="time" id="until-${esc(d.id)}"></label>
             <button type="button" data-act="start" data-design="${esc(d.id)}">${d.exists ? "Start" : "Start in a new workspace"}</button>`}</div>` : ""}
      </div>`;
    }).join("")}`;

  const byExp = new Map();
  for (const c of a.calls) { if (!byExp.has(c.exp)) byExp.set(c.exp, []); byExp.get(c.exp).push(c); }
  $("#ag-live").innerHTML = `
    <h3>Activity <span class="status ${busy ? "on" : ""}"><span class="dot"></span>${status ? `${esc(humanize(status.phase))} #${status.n}${status.title ? " · " + esc(clip(status.title, 40)) : ""}` : a.running ? "Starting…" : "Idle"}</span></h3>
    <div class="calls">${[...byExp.entries()].map(([n, calls]) => `
      <div class="call-grp"><div class="call-exp">Experiment #${n}</div>
        ${calls.map((c) => `<button type="button" class="call${S.detail && S.detail.key === c.file ? " on" : ""}${c.error ? " err" : ""}" data-trace="${esc(c.file)}">
          <span class="call-role role-${esc(c.role)}">${esc(c.role)}${c.step ? " · " + esc(c.step) : ""}</span>
          <span class="call-title">${esc(c.title || clip(c.preview, 90) || "(empty reply)")}</span>
          <span class="call-meta mono">${new Date(c.ts * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })} · ${c.seconds ?? "?"}s${c.tokens ? " · " + (c.tokens / 1000).toFixed(1) + "k tok" : ""}</span>
        </button>`).join("")}</div>`).join("") || `<p class="muted">No agent calls yet.</p>`}</div>
    <details class="log-box"><summary>Loop log</summary><pre class="log mono">${esc(a.log.join("\n")) || "empty"}</pre></details>`;
}

async function renderDetail() {
  const el = $("#ag-detail");
  if (!el) return;
  const d = S.detail;
  if (!d) { el.innerHTML = `<h3>Details</h3><p class="muted">Pick a role to edit its prompt, or a call to see what the agent read, reasoned and replied.</p>`; return; }
  el.innerHTML = `<p class="muted">Loading…</p>`;
  el.scrollTop = 0;
  try {
    if (d.kind === "prompt") {
      const p = await getJSON(`/api/prompts/${encodeURIComponent(d.key)}`);
      el.innerHTML = `<h3>${esc(d.key)} prompt <span class="muted mono">harness/prompts/${esc(d.key)}.md</span></h3>
        <textarea class="prompt mono" id="prompt-text" spellcheck="false"></textarea>
        <div class="row"><span class="muted" id="prompt-note">Changes apply from the next call. Shared by every workspace.</span>
          <button type="button" id="prompt-revert">Revert</button><button type="button" class="primary" id="prompt-save">Save</button></div>`;
      const ta = $("#prompt-text");
      ta.value = p.text;
      $("#prompt-revert").onclick = () => { ta.value = p.text; $("#prompt-note").textContent = "Reverted."; };
      $("#prompt-save").onclick = async () => {
        try { await post(`/api/prompts/${encodeURIComponent(d.key)}`, { text: ta.value }); p.text = ta.value; $("#prompt-note").textContent = "Saved."; }
        catch (e) { $("#prompt-note").textContent = "Not saved: " + e.message; }
      };
    } else {
      const t = await getJSON(`/api/p/${S.pid}/trace/${encodeURIComponent(d.key)}`);
      const msgs = t.messages || [];
      el.innerHTML = `<h3>${esc(t.role)} <span class="muted mono">${esc(String(t.model || "").split("/").pop())} · ${t.seconds ?? "?"}s</span></h3>
        ${t.error ? `<p class="worse">${esc(t.error)}</p>` : ""}
        ${msgs.map((msg, k) => `<details class="msg" ${k === msgs.length - 1 && msg.role !== "system" ? "open" : ""}><summary>${esc(msg.role)} <span class="muted">${String(msg.content || "").length.toLocaleString()} chars</span></summary>
          <pre class="mono">${esc(msg.content)}</pre></details>`).join("")}
        ${t.reasoning ? `<details class="msg" open><summary>reasoning</summary><pre class="mono">${esc(t.reasoning)}</pre></details>` : ""}
        <details class="msg" open><summary>reply</summary><pre class="mono">${esc(t.reply || "")}</pre></details>`;
    }
  } catch (e) { el.innerHTML = `<p class="worse">${esc(e.message)}</p>`; }
}

async function agentAction(act, design) {
  try {
    if (act === "start") {
      const d = S.agents.designs.find((x) => x.id === design);
      const until = ($(`#until-${design}`) || {}).value || "";
      if (!confirm(`Start the ${d.name} loop${d.exists ? "" : " in a new workspace"}${until ? " until " + until : ""}?\nIt trains on this machine and calls the LLM provider.`)) return;
      const r = await post(`/api/p/${S.pid}/run`, { design, until: until || null });
      if (r.id !== S.pid) location.hash = `#/p/${r.id}/agents`;
    } else if (act === "stop") await post(`/api/p/${S.pid}/stop`, {});
    else if (act === "kill" && confirm("Stop the loop now? The current experiment is abandoned.")) await post(`/api/p/${S.pid}/stop`, { now: true });
    S.agentsRaw = "";
    poll(false);
  } catch (e) { alert(e.message); }
}

// ------------------------------------------------------------------ events

function toggleTheme() {
  const dark = document.documentElement.dataset.theme ? document.documentElement.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  document.documentElement.dataset.theme = dark ? "light" : "dark";
  try { localStorage.setItem("ar-theme", document.documentElement.dataset.theme); } catch (e) {}
  renderChart();
}

document.addEventListener("click", (e) => {
  const t = e.target;
  const selEl = t.closest("[data-sel]");
  if (selEl && S.route.page === "project" && !t.closest("a")) {
    const sel = selEl.dataset.sel;
    if (sel && sel.startsWith("e") && S.route.view !== "evolution" && selEl.closest("#chart")) {
      location.hash = projHref(S.pid, "evolution", sel);  // a point may have no node in this view
      return;
    }
    select(sel === S.route.sel && selEl.closest("#arch") ? null : sel);
    return;
  }
  const prompt = t.closest("[data-prompt]");
  if (prompt && prompt.dataset.prompt) { S.detail = { kind: "prompt", key: prompt.dataset.prompt }; renderAgents(); renderDetail(); return; }
  const trace = t.closest("[data-trace]");
  if (trace) { S.detail = { kind: "trace", key: trace.dataset.trace }; renderAgents(); renderDetail(); return; }
  const act = t.closest("[data-act]");
  if (act && S.route.page === "agents" && act.closest("#ag-designs")) agentAction(act.dataset.act, act.dataset.design);
});

document.addEventListener("toggle", (e) => { if (e.target.matches && e.target.matches("details.code") && e.target.open) loadDiff(e.target); }, true);

document.addEventListener("mouseover", (e) => {
  const t = e.target.closest && e.target.closest("[data-sel]");
  if (!t || S.route.page !== "project" || !t.closest("#arch, #chart")) return;
  highlight(t.dataset.sel, true);
  if (t.closest("#chart")) showTip(t);
});
document.addEventListener("mouseout", (e) => {
  const t = e.target.closest && e.target.closest("[data-sel]");
  if (!t || !t.closest("#arch, #chart")) return;
  highlight(t.dataset.sel, false);
  $("#tip").hidden = true;
});

document.addEventListener("keydown", (e) => {
  if (e.metaKey || e.ctrlKey || e.altKey || /^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName)) return;
  const k = e.key.toLowerCase();
  if (k === "t") return toggleTheme();
  if (S.route.page !== "project" || !S.m) return;
  if (k === "1" || k === "2" || k === "3") location.hash = projHref(S.pid, VIEWS[+k - 1], S.route.sel);
  else if (k === "escape") select(null);
  else if (k === "arrowright" || k === "arrowleft") {
    const exps = S.m.exps, cur = S.route.sel && S.route.sel.startsWith("e") ? +S.route.sel.slice(1) : null;
    const i = cur == null ? (k === "arrowright" ? 0 : exps.length - 1) : exps.findIndex((x) => x.n === cur) + (k === "arrowright" ? 1 : -1);
    if (exps[i]) select("e" + exps[i].n);
    e.preventDefault();
  }
});

addEventListener("hashchange", onRoute);
onRoute();
