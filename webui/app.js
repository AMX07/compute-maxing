"use strict";
// Turing UI. Three pages, routed by the URL hash:
//   #/                        recent projects
//   #/p/<id>/<view>[/<sel>]   a project's architecture, as baseline | evolution | current, linked to the loss chart
//   #/p/<id>/agents           the agent loop: design choice, prompts, start/stop, and every LLM call as it happens
// A project is a directory with train.py, results.tsv and runs/. Its optional architecture.json maps experiments
// to model components; without one, experiments are grouped by the planner's idea category. No dependencies.

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const POLL_MS = 4000;
const VIEWS = ["baseline", "eval", "evolution", "current"];
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
  ui: { chart: true, side: true, architecture: "diagram" },       // which project panels are open (remembered per browser)
};
try { Object.assign(S.ui, JSON.parse(localStorage.getItem("ar-ui") || "{}")); } catch (e) {}

const CHEVRON = `<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M6 3.5 10.5 8 6 12.5" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>`;

// Opens or minimizes the loss chart or the side panel.
function setPanel(name, open) {
  S.ui[name] = open;
  try { localStorage.setItem("ar-ui", JSON.stringify(S.ui)); } catch (e) {}
  const app = $("#app");
  app.classList.toggle("chart-min", !S.ui.chart);
  app.classList.toggle("side-min", !S.ui.side);
  $$("[data-toggle]").forEach((b) => {
    const on = S.ui[b.dataset.toggle];
    b.setAttribute("aria-expanded", String(on));
    b.title = (on ? "Minimize the " : "Show the ") + (b.dataset.toggle === "chart" ? "loss chart" : "side panel");
  });
}

// ------------------------------------------------------------------ formatting

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmtLoss = (v) => (v == null ? "—" : S.m?.arch?.metric?.unit === "s" ? v.toFixed(2) + " s" : v.toFixed(4));
function fmtPct(p) {
  if (p == null || !isFinite(p)) return "—";
  const a = Math.abs(p);
  return (p < 0 ? "−" : p > 0 ? "+" : "±") + (a >= 1 ? a.toFixed(1) : a.toFixed(2)) + "%";
}
const pctOf = (d, ref) => (d == null || !ref ? null : (d / ref) * 100);
const tone = (p, goal = S.m?.arch?.metric?.goal || "min") => (p == null || p === 0 ? "" : (goal === "max" ? p > 0 : p < 0) ? "good" : "worse");
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
const STATUS = { keep: "kept", discard: "discarded", crash: "crashed", retime: "re-timed" };
const statusLabel = (e) => S.m?.arch?.history_only && e.status === "keep" ? "published record" : STATUS[e.status] || e.status;
const ICON = { keep: "✓", discard: "–", crash: "✕", retime: "↻" };
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
  if (p[2] === "eval") return { page: "eval", pid: p[1] };
  if (p[2] === "agents") return { page: "agents", pid: p[1] };
  return { page: "project", pid: p[1], view: VIEWS.includes(p[2]) ? p[2] : "evolution", sel: p[3] || null };
}
const projHref = (pid, view, sel) => `#/p/${pid}/${view || "evolution"}` + (sel ? "/" + encodeURIComponent(sel) : "");

function select(sel) {
  if (S.route.page !== "project") return;
  const focusedComponent = document.activeElement?.closest?.("#arch [data-sel]")?.dataset.sel;
  S.route.sel = sel || null;
  if (sel && !S.ui.side) setPanel("side", true);  // a selection is read in the side panel
  history.replaceState(null, "", projHref(S.pid, S.route.view, S.route.sel));
  renderArch(); renderSide(); renderChart();
  if (focusedComponent) $(`#arch [data-sel="${CSS.escape(focusedComponent)}"]`)?.focus({ preventScroll: true });
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
  else if (r.page === "eval") renderEval();
  else if (r.page === "project") { renderArch(); renderSide(); renderChart(); }
  await poll(!samePid);
  S.timer = setInterval(() => poll(false), r.page === "agents" ? 1000 : POLL_MS);
}

async function poll(open) {
  try {
    // Redraw only when something other than the server clock changed, so clicks and scroll positions survive polls.
    const unclocked = (t) => t.replace(/"server_time":\s*[\d.eE+-]+/, "");
    const raw = await (await fetch(`/api/p/${S.pid}` + (open ? "?open=1" : ""), { cache: "no-store" })).text();
    S.error = null;
    const data = JSON.parse(raw);
    if (unclocked(raw) !== S.raw) {
      S.data = data;
      S.m = buildModel(S.data);
      if (S.route.page === "eval") renderEval();
      if (S.route.page === "project") { renderArch(); renderSide(); renderChart(); }
      S.raw = unclocked(raw);  // only after a successful draw, so a failed one is retried on the next poll
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
  const arch = { ...(data.architecture || {}), metric: { name: data.metric, goal: data.goal, ...(data.architecture?.metric || {}) } };
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
    if (e.comparison_reset) { best = null; bestExp = null; }
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
    if (["keep", "retime"].includes(e.status) && e.val_loss != null && (best == null || (arch.metric?.goal === "max" ? e.val_loss > best : e.val_loss < best))) { best = e.val_loss; bestExp = e; }
    e.pct = pctOf(e.delta_best, e.best_before);
    e.verdict = verdictOf(e);
  }
  for (const c of comps) {
    let cur = null;
    for (const e of c.steps) {
      e.before = cur ? (cur.change.after || cur.label) : c.baseline.label;
      if (["keep", "retime"].includes(e.status)) cur = e;
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
const viewNote = (view) => S.m?.arch?.history_only ? ({
  baseline: "The published starting point, grouped by research category.",
  evolution: "Published records and re-timings, grouped by the changes described upstream.",
  current: "The latest published change in each category; this is not a full model specification.",
})[view] : VIEW_NOTE[view];
const metricName = (arch) => (arch && arch.metric && arch.metric.name) || (S.data && S.data.metric) || "val_loss";

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
      <div class="home-brand">${GLYPH}<span>Turing</span></div>
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
      <span class="tile-sub">${c.total} runs${p != null ? ` · <b class="${tone(p, c.goal || "min")}">${fmtPct(p)}</b>` : ""}</span>
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
  const evaluation = S.route.page === "eval";
  $("#app").className = evaluation ? "page-eval" : agents ? "page-agents" : "page-project";
  $("#app").innerHTML = `
    <header class="bar" id="bar"></header>
    ${evaluation ? `<main class="eval-view" id="eval-view"></main>` : agents ? `<main class="agents-view">
      <div class="panel agents-nav" id="agents-nav">${viewTabs("agents")}</div>
      <section class="panel research-panel" id="research-panel"></section>
      <section class="panel ag-col research-designs" id="ag-designs"></section>
      <div class="agents" id="agents"><p class="muted pad">Loading…</p></div>
      <section class="panel training-panel" id="training-panel"></section>
    </main>` : `
    <main class="proj">
      <section class="canvas panel">
        <div class="arch" id="arch"></div>
        <div class="chart-wrap">
          <button type="button" class="panel-head" data-toggle="chart"><span class="tag" id="chart-label">loss</span><span class="caret">${CHEVRON}</span></button>
          <div class="chart" id="chart"></div>
        </div>
      </section>
      <aside class="side panel" id="side">
        <div class="side-head"><span id="side-label"></span>
          <button type="button" class="caret-btn" data-toggle="side">${CHEVRON}</button></div>
        <div class="side-body" id="side-body" aria-live="polite"></div>
      </aside>
    </main>`}`;
  renderHeader();
  // Switching pages within a project keeps the loaded state, so draw it now rather than on the next change.
  if (evaluation) renderEval();
  else if (agents) renderAgents();
  else { setPanel("chart", S.ui.chart); setPanel("side", S.ui.side); renderArch(); renderSide(); new ResizeObserver(() => renderChart()).observe($("#chart")); }
}

function agentStatus() {
  const d = S.data;
  if (!d) return { text: "Loading…", cls: "" };
  const research = S.agents?.research;
  if (research?.active) return { text: research.latest?.message || "Agentic research running", cls: research.latest?.status === "orphaned" ? "bad" : "on" };
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
  // Just the project name; the path and the agent's status are in its tooltip, and a dot shows only while it runs.
  const d = S.data;
  const st = S.error ? { text: S.error, cls: "bad" } : agentStatus();
  el.innerHTML = `
    <a class="home-link" href="#/" title="All projects">${GLYPH}</a>
    <h1 class="ptitle status ${st.cls}" title="${esc(shortPath(d && d.root))}\n${esc(st.text)}">${st.cls ? `<span class="dot"></span>` : ""}${esc(d ? d.name : "…")}</h1>
    <div class="bar-end">
    <button type="button" class="icon-btn" id="theme" title="Light or dark (T)" aria-label="Toggle theme">
      <svg viewBox="0 0 20 20" aria-hidden="true"><circle cx="10" cy="10" r="7.25" fill="none" stroke="currentColor" stroke-width="1.5"/><path d="M10 2.75a7.25 7.25 0 0 1 0 14.5z" fill="currentColor"/></svg>
    </button></div>`;
  $("#theme").onclick = toggleTheme;
}

// The same view selector stays available in both architecture and agent views.
function viewTabs(active) {
  return `<nav class="seg small" aria-label="Project view">${[...VIEWS, "agents"].map((view, i) => {
    const label = view === "current" && S.m?.arch?.history_only ? "Latest" : humanize(view);
    const href = ["agents", "eval"].includes(view) ? `#/p/${S.pid}/${view}` : projHref(S.pid, view, S.route.sel);
    return `<a href="${href}" class="${active === view ? "on" : ""}"${active === view ? ' aria-current="page"' : ""} title="${esc(label)} (${i + 1})">${esc(label)}</a>`;
  }).join("")}</nav>`;
}

// ------------------------------------------------------------------ evaluation
function renderEval() {
  const el = $("#eval-view"), d = S.data;
  if (!el) return;
  if (!d) { el.innerHTML = '<p class="muted pad">Loading evaluation…</p>'; return; }
  const m = S.m.arch.metric || {}, protocol = d.evaluation || {};
  const formatScore = value => !Number.isFinite(value) ? "—" : `${value.toFixed(m.unit === "s" ? 2 : 4)}${m.unit ? " " + esc(m.unit) : ""}`;
  const exps = S.m.exps, scored = exps.filter(e => Number.isFinite(e.val_loss));
  const group = exps.at(-1)?.comparison_group;
  const comparable = scored.filter(e => e.comparison_group === group && ["keep", "retime"].includes(e.status));
  const best = comparable.reduce((a, e) => !a || (m.goal === "max" ? e.val_loss > a.val_loss : e.val_loss < a.val_loss) ? e : a, null);
  const baseline = exps[0], latest = exps.at(-1), history = !!S.m.arch.history_only;
  const scoreCard = (label, e, note) => `<article class="eval-score"><span class="tag">${label}</span><strong class="mono">${formatScore(e?.val_loss)}</strong><span class="muted small">${esc(note)}</span></article>`;
  const source = protocol.source;
  el.innerHTML = `
    <div class="panel agents-nav">${viewTabs("eval")}</div>
    <section class="panel eval-panel">
      <div class="eval-title"><div><span class="tag">Evaluation metric</span><h2>${esc(metricName(S.m.arch))}</h2><p class="muted">${esc(protocol.title || (history ? "Published benchmark score" : "Recorded evaluation score"))}</p></div><span class="pill">${m.goal === "max" ? "Higher" : "Lower"} is better</span></div>
      <div class="eval-scores">
        ${scoreCard("Baseline", baseline, "First recorded experiment")}
        ${scoreCard(history ? "Best in current series" : "Best kept", best, best ? `Experiment #${best.n}${history ? " · " + (group || "current series") : ""}` : "No accepted score recorded")}
        ${scoreCard("Latest result", latest, latest ? `Experiment #${latest.n} · ${statusLabel(latest)}` : "No results yet")}
        <article class="eval-score"><span class="tag">Recorded scores</span><strong class="mono">${scored.length}<small> / ${exps.length}</small></strong><span class="muted small">${exps.length - scored.length} unavailable · ${history ? "published history" : "experiment history"}</span></article>
      </div>
    </section>
    <div class="eval-columns">
      <section class="panel eval-panel"><h3>How evaluation works</h3><p class="eval-description">${esc(protocol.description || "No evaluator documentation is available for this project.")}</p>
        ${protocol.formula ? `<div class="eval-formula mono">${esc(protocol.formula)}</div>` : ""}
        ${(protocol.facts || []).length ? `<dl class="eval-facts">${protocol.facts.map(f => `<div><dt>${esc(f.label)}</dt><dd class="mono">${esc(typeof f.value === "number" ? f.value.toLocaleString() : f.value)} ${esc(f.unit)}</dd></div>`).join("")}</dl>` : ""}
      </section>
      <section class="panel eval-panel"><h3>Reading these results</h3><p>${esc(protocol.note || "Scores come from the project's results.tsv.")}</p>
        <p>Memory is reported separately and is not the optimization score. An unavailable score is not treated as zero.</p>
        ${source ? `<div class="eval-source"><span class="tag">Evaluator source · current checkout</span><p class="mono">${esc(source.file)}:${source.line} · ${esc(source.function)}()</p><p class="muted small mono">SHA-256 ${esc(source.sha256.slice(0, 16))}</p></div>` : ""}
        ${protocol.metadata_file ? `<p class="small muted">Evaluation documentation: <span class="mono">${esc(protocol.metadata_file)}</span></p>` : ""}
        <p class="small muted">Standalone training output appears in Agents. A completed process does not automatically produce an evaluated result here.</p>
      </section>
    </div>
    <section class="panel eval-panel"><div class="eval-title"><h3>Recorded results</h3><span class="muted small">Newest first · select a run to inspect it</span></div>
      ${exps.length ? `<div class="eval-table-wrap"><table class="eval-table"><thead><tr><th scope="col">Run</th><th scope="col">${esc(metricName(S.m.arch))}</th><th scope="col">Memory · GB</th><th scope="col">Status</th><th scope="col">Description</th></tr></thead><tbody>${[...exps].reverse().map(e => `<tr><td><a href="${projHref(S.pid, "evolution", "e" + e.n)}">#${e.n}</a></td><td class="mono">${formatScore(e.val_loss)}</td><td class="mono">${Number.isFinite(e.memory_gb) ? e.memory_gb.toFixed(1) : "—"}</td><td>${esc(statusLabel(e))}</td><td>${esc(e.description || "—")}${e.comparison_reset ? '<span class="eval-reset">New comparison series</span>' : ""}</td></tr>`).join("")}</tbody></table></div>` : '<p class="muted">No evaluation results yet. Recorded scores will appear here when the project writes results.tsv.</p>'}
    </section>`;
}

// BEGIN SHARED ARCHITECTURE DIAGRAM
// Architecture diagrams use recorded topology only. Category/component maps never imply edges.
function diagramIcon(kind) {
  const paths = {
    component: '<rect x="4" y="4" width="6" height="6" rx="1.2"/><rect x="14" y="4" width="6" height="6" rx="1.2"/><rect x="4" y="14" width="6" height="6" rx="1.2"/><rect x="14" y="14" width="6" height="6" rx="1.2"/>',
    repeat: '<path d="m4 7 8-4 8 4-8 4-8-4Zm0 5 8 4 8-4M4 17l8 4 8-4"/>',
    input: '<rect x="3" y="6" width="5" height="12" rx="1.5"/><rect x="10" y="6" width="5" height="12" rx="1.5"/><path d="M18 6h3v12h-3"/>',
    output: '<path d="M5 19V9m7 10V4m7 15v-7"/>',
    operation: '<path d="M7 4H4v16h3m10-16h3v16h-3M9 12h6"/>',
    parallel: '<path d="M12 3v4M5 11V7h14v4M5 15v3h14v-3m-7 3v3"/><rect x="2" y="11" width="6" height="4" rx="1"/><rect x="16" y="11" width="6" height="4" rx="1"/>',
    residual: '<path d="M12 3v18M12 5H4v14h8m-3-3 3 3-3 3"/>',
    add: '<path d="M6 12h12M12 6v12"/>',
  };
  return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${paths[kind] || paths.component}</svg>`;
}

function renderArchitectureDiagram(m, view) {
  const text = value => typeof value === 'string' || typeof value === 'number' ? String(value) : '';
  const e = value => esc(text(value));
  const declared = Array.isArray(m.arch?.components) ? new Set(m.arch.components.map(c => c.id)) : null;
  const comps = (Array.isArray(m.comps) ? m.comps : []).filter(c => !declared || declared.has(c.id));
  const byId = new Map(comps.map(c => [c.id, c]));
  const arch = m.arch || {}, used = new Set();
  const current = view === 'current';
  const label = c => text(current ? c.currentLabel : c.baseline?.label) || 'Value not recorded';
  const cost = c => text(current ? c.currentCost : c.baseline?.cost);
  const selection = c => current && c.current ? 'e' + c.current.n : 'c:' + c.id;
  const badge = c => current ? (c.current ? 'Changed in #' + c.current.n : 'Unchanged') : 'Baseline';
  const nodeButton = (c, node = {}, extra = '') => {
    const name = text(c.name) || humanize(text(c.id));
    const title = text(node.title) || name;
    const selected = typeof S !== 'undefined' && S.route?.sel === selection(c);
    return `<button type="button" class="diagram-node ${extra}${current && c.current ? ' diagram-changed' : ''}${selected ? ' sel' : ''}" data-sel="${e(selection(c))}" data-comp="${e(c.id)}" aria-label="${e(title + (title !== name ? ' (' + name + ')' : '') + ': ' + label(c) + '. Open component details')}">
      <span class="diagram-node-top"><span class="diagram-node-name">${e(title)}</span><span class="diagram-badge">${e(badge(c))}</span></span>
      ${title !== name ? `<span class="diagram-node-source">${e(name)}</span>` : ''}
      <span class="diagram-value">${e(label(c))}</span>
      ${node.note ? `<span class="diagram-note">${e(node.note)}</span>` : ''}
      ${cost(c) ? `<span class="diagram-cost">${e(cost(c))}</span>` : ''}
      ${current && c.current && c.baseline?.label ? `<span class="diagram-previous">Baseline: ${e(c.baseline.label)}</span>` : ''}
    </button>`;
  };
  const groups = Array.isArray(m.groups) && m.groups.length ? m.groups : [{ id: 'components', name: 'Components', comps }];
  const groupCards = (remaining, fallback = false) => {
    const wanted = new Set(remaining.map(c => c.id));
    const shown = new Set();
    const sections = groups.map(g => {
      const entries = (Array.isArray(g.comps) ? g.comps : []).filter(c => wanted.has(c.id) && !shown.has(c.id));
      entries.forEach(c => shown.add(c.id));
      if (!entries.length) return '';
      return `<section class="diagram-setting-group"><h3>${e(g.name || humanize(text(g.id)))}</h3><div class="diagram-card-grid">${entries.map(c => nodeButton(c, {}, 'diagram-setting')).join('')}</div></section>`;
    });
    const ungrouped = remaining.filter(c => !shown.has(c.id));
    if (ungrouped.length) sections.push(`<section class="diagram-setting-group"><h3>Other components</h3><div class="diagram-card-grid">${ungrouped.map(c => nodeButton(c, {}, 'diagram-setting')).join('')}</div></section>`);
    return `<div class="diagram-settings${fallback ? ' diagram-map-cards' : ''}">${sections.join('')}</div>`;
  };
  const diagram = arch.diagram;
  let valid = !!diagram && diagram.version === 1 && Array.isArray(diagram.flow) && diagram.flow.length > 0;
  let count = 0;
  const validateNode = (node, depth) => {
    if (!node || typeof node !== 'object' || Array.isArray(node) || depth > 8 || ++count > 160) return false;
    if (node.title != null && typeof node.title !== 'string' || node.note != null && typeof node.note !== 'string') return false;
    if (node.component != null && (typeof node.component !== 'string' || !byId.has(node.component))) return false;
    const kind = node.kind || (node.component ? 'component' : 'operation');
    if (['repeat', 'residual', 'parallel'].includes(kind)) {
      return Array.isArray(node.children) && node.children.length > 0 && node.children.length <= 160 && node.children.every(n => validateNode(n, depth + 1));
    }
    if (!['component', 'operation', 'input', 'output'].includes(kind) || node.children != null) return false;
    return !!node.component || (typeof node.title === 'string' && !!node.title.trim());
  };
  if (valid) valid = diagram.flow.length <= 160 && diagram.flow.every(n => validateNode(n, 0));
  if (!valid || current || arch.history_only) {
    const history = !!arch.history_only;
    const title = history ? 'Research category map' : 'Component map';
    const description = history
      ? 'Published milestones are grouped by research category. They do not describe a connected model architecture.'
      : current
        ? 'Recorded component values. Connections are not reconstructed from experiment changes.'
        : 'Explore the recorded values below. Add an explicit topology to show how the components connect.';
    return `<section class="architecture-diagram diagram-fallback" aria-label="${e(title)}">
      <header class="diagram-heading"><div class="diagram-title-row"><span class="diagram-title-icon">${diagramIcon('component')}</span><div><h2>${e(title)}</h2><p class="diagram-subtitle">${history ? 'Categories · connections not recorded' : 'Component map · connections not recorded'}</p></div></div><p class="diagram-description">${e(description)}</p></header>
      ${comps.length ? groupCards(comps, true) : '<p class="diagram-empty">No architecture components have been recorded for this project yet.</p>'}
    </section>`;
  }
  // Geometry follows the recorded computational order, laid out from bottom to top.
  // The side label on a repeated block is descriptive; it is never a computation node.
  const gap = 24, baseWidth = 276;
  const tones = new Set(['embedding', 'attention', 'ffn', 'norm', 'head', 'input', 'output', 'neutral']);
  const safeTone = node => tones.has(node.tone) ? node.tone : 'neutral';
  const kindOf = node => node.kind || (node.component ? 'component' : 'operation');
  const nodeTitle = node => text(node.title) || (node.component ? text(byId.get(node.component).name) : humanize(kindOf(node)));
  const lineCount = (value, width, size) => {
    // Conservative wrapping budget; actual text remains intact in native HTML buttons.
    const columns = Math.max(10, Math.floor((width - 20) / (size * 0.58)));
    return Math.max(1, text(value).split('\n').reduce((n, line) => n + Math.ceil(Math.max(1, line.length) / columns), 0));
  };
  const measureSequence = nodes => {
    const children = nodes.map(measure);
    const axis = Math.max(...children.map(child => child.axis));
    return { children, axis, w: axis + Math.max(...children.map(child => child.w - child.axis)), h: children.reduce((sum, child) => sum + child.h, 0) + gap * (children.length - 1) };
  };
  function measure(node) {
    const kind = kindOf(node);
    if (kind === 'repeat') {
      const body = measureSequence(node.children);
      const frame = { w: body.w + 28, h: body.h + 32, axis: body.axis + 14 };
      const labelWidth = 106;
      return { node, kind, body, frame, labelWidth, axis: labelWidth + 12 + frame.axis, w: labelWidth + 12 + frame.w, h: frame.h };
    }
    if (kind === 'residual') {
      const body = measureSequence(node.children);
      return { node, kind, body, axis: body.axis + 26, w: body.w + 52, h: body.h + 54 };
    }
    if (kind === 'parallel') {
      const branches = node.children.map(measure);
      const w = branches.reduce((sum, branch) => sum + branch.w, 0) + 24 * (branches.length - 1);
      return { node, kind, branches, axis: w / 2, w, h: Math.max(...branches.map(branch => branch.h)) + 58 };
    }
    const component = node.component && byId.get(node.component);
    const title = nodeTitle(node);
    const h = component
      ? 16 + lineCount(title, baseWidth, 18) * 22 + lineCount(label(component), baseWidth, 17) * 21
      : Math.max(34, 12 + lineCount(title, baseWidth, 18) * 22);
    return { node, kind, axis: baseWidth / 2, w: baseWidth, h };
  }
  const measured = measureSequence(diagram.flow);
  // IDs are local to this one architecture figure; no user metadata enters SVG IDs.
  const markerId = 'architecture-flow-arrow';
  const arrow = (path, extra = '', edge = 'sequence') => `<path class="schematic-edge ${extra}" data-flow-edge="${edge}" d="${path}" marker-end="url(#${markerId})"/>`;
  const line = path => `<path class="schematic-edge" d="${path}"/>`;
  const componentBox = (c, node, x, y, width, height, extra = '') => {
    used.add(c.id);
    const title = nodeTitle(node), name = text(c.name) || humanize(text(c.id));
    const selected = typeof S !== 'undefined' && S.route?.sel === selection(c);
    return `<foreignObject x="${x}" y="${y}" width="${width}" height="${height}" class="schematic-label-object"><div xmlns="http://www.w3.org/1999/xhtml" class="schematic-button-wrap"><button type="button" class="diagram-node schematic-button ${extra}${selected ? ' sel' : ''}" data-sel="${e(selection(c))}" data-comp="${e(c.id)}" aria-label="${e(title + (title !== name ? ' (' + name + ')' : '') + ': ' + label(c) + '. Open component details')}" title="${e(name + ': ' + label(c) + (node.note ? '. ' + text(node.note) : ''))}"><span class="schematic-node-title">${e(title)}</span><span class="schematic-node-value">${e(label(c))}</span></button></div></foreignObject>`;
  };
  const operationBox = (node, x, y, width, height, extra = '') => `<foreignObject x="${x}" y="${y}" width="${width}" height="${height}"><div xmlns="http://www.w3.org/1999/xhtml" class="schematic-operation ${extra}" title="${e(node.note || nodeTitle(node))}">${e(nodeTitle(node))}</div></foreignObject>`;
  function drawSequence(sequence, x, y) {
    let cursor = y + sequence.h;
    let out = '';
    sequence.children.forEach((child, index) => {
      cursor -= child.h;
      const childX = x + sequence.axis - child.axis;
      out += draw(child, childX, cursor);
      if (index < sequence.children.length - 1) {
        out += arrow(`M${x + sequence.axis} ${cursor}V${cursor - gap}`);
        cursor -= gap;
      }
    });
    return out;
  }
  function draw(box, x, y) {
    const node = box.node, cx = x + box.axis, tone = safeTone(node);
    if (box.kind === 'repeat') {
      const frameX = x + box.labelWidth + 12, bodyX = frameX + 14, bodyY = y + 16;
      const c = node.component && byId.get(node.component);
      const heading = c ? { ...node, title: text(c.name) || nodeTitle(node) } : node;
      const labelHeight = Math.min(box.h - 8, 18 + lineCount(nodeTitle(heading), box.labelWidth, 17) * 21 + (c ? lineCount(label(c), box.labelWidth, 16) * 21 : 0));
      const labelY = y + (box.h - labelHeight) / 2;
      const labelHTML = c ? componentBox(c, heading, x, labelY, box.labelWidth, labelHeight, 'schematic-repeat-label') : operationBox(node, x, labelY, box.labelWidth, labelHeight, 'schematic-repeat-label');
      return `<g class="diagram-repeat" data-schematic-kind="repeat" data-node-y="${y}" aria-label="${e(nodeTitle(node))}"><rect class="schematic-repeat-frame" x="${frameX}" y="${y}" width="${box.frame.w}" height="${box.h}" rx="9"/>${labelHTML}${line(`M${cx} ${y + box.h}V${y + box.h - 16}`)}${drawSequence(box.body, bodyX, bodyY)}${line(`M${cx} ${bodyY}V${y}`)}</g>`;
    }
    if (box.kind === 'residual') {
      const childX = x + 26, childY = y + 38;
      const joinY = y + 12, inputY = y + box.h, railX = x + 7;
      const c = node.component && byId.get(node.component);
      // Containers that carry values retain those values in supporting settings.
      const extraLabel = c ? `<title>${e(text(c.name) + ': ' + label(c))}</title>` : '';
      return `<g class="diagram-residual" data-schematic-kind="residual" data-node-y="${y}" aria-label="${e(nodeTitle(node))}">${extraLabel}${arrow(`M${cx} ${inputY}H${railX}V${joinY}H${cx - 12}`, 'schematic-bypass', 'residual')}${line(`M${cx} ${inputY}V${childY + box.body.h}`)}<circle class="schematic-branch-point" cx="${cx}" cy="${inputY}" r="2.7"/>${drawSequence(box.body, childX, childY)}${arrow(`M${cx} ${childY}V${joinY + 12}`)}<g class="schematic-add" aria-label="Add residual connection"><circle cx="${cx}" cy="${joinY}" r="12"/><path d="M${cx - 5} ${joinY}h10M${cx} ${joinY - 5}v10"/></g></g>`;
    }
    if (box.kind === 'parallel') {
      let bx = x;
      const topY = y + 15, bottomY = y + box.h - 15;
      let paths = '', branches = '';
      box.branches.forEach(branch => {
        const branchY = y + 29 + (box.h - 58 - branch.h) / 2, branchCX = bx + branch.axis;
        paths += arrow(`M${cx} ${bottomY}H${branchCX}V${branchY + branch.h}`, '', 'parallel');
        paths += line(`M${branchCX} ${branchY}V${topY}H${cx}`);
        branches += draw(branch, bx, branchY);
        bx += branch.w + 24;
      });
      return `<g class="diagram-parallel" data-schematic-kind="parallel" data-node-y="${y}" aria-label="${e(nodeTitle(node))}">${line(`M${cx} ${y + box.h}V${bottomY}`)}${paths}${branches}${arrow(`M${cx} ${topY}V${y}`, '', 'parallel')}</g>`;
    }
    const c = node.component && byId.get(node.component);
    const isEndpoint = !c && ['input', 'output'].includes(box.kind);
    const shape = isEndpoint ? '' : `<rect class="schematic-box schematic-tone-${tone}" x="${x}" y="${y}" width="${box.w}" height="${box.h}" rx="3"/>`;
    const html = c ? componentBox(c, node, x, y, box.w, box.h) : operationBox(node, x, y, box.w, box.h, isEndpoint ? 'schematic-endpoint' : '');
    return `<g class="schematic-step schematic-${box.kind}" data-schematic-kind="${box.kind}" data-node-y="${y}"${c ? ` data-schematic-component="${e(c.id)}"` : ''}>${shape}${html}</g>`;
  }
  const padding = 18;
  const content = drawSequence(measured, padding, padding);
  const remaining = comps.filter(c => !used.has(c.id));
  const svg = `<svg class="diagram-flow diagram-schematic" data-flow-direction="bottom-up" viewBox="0 0 ${measured.w + padding * 2} ${measured.h + padding * 2}" xmlns="http://www.w3.org/2000/svg" aria-label="${e('Baseline architecture. Read from input at the bottom to output at the top. ' + (diagram.description || ''))}"><defs><marker id="${markerId}" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto" markerUnits="userSpaceOnUse"><path class="schematic-arrowhead" d="M0 0 7 3.5 0 7Z"/></marker></defs>${content}</svg>`;
  return `<section class="architecture-diagram architecture-schematic" aria-label="Model architecture diagram"><header class="schematic-heading"><h2>Model architecture</h2><span>${e(diagram.basis || 'Recorded baseline topology')}</span></header><div class="schematic-paper">${svg}</div><p class="schematic-caption">Read from bottom to top. Select a colored component to inspect its values.${diagram.description ? ` <span>${e(diagram.description)}</span>` : ''}</p>${remaining.length ? `<aside class="diagram-context" aria-label="Additional architecture and training settings">${groupCards(remaining)}</aside>` : ''}</section>`;
}

// END SHARED ARCHITECTURE DIAGRAM

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
      ${viewTabs(v)}
      <p class="muted">${esc(viewNote(v))}${m.hasMap ? "" : " No architecture.json yet, so experiments are grouped by idea category."}</p>
      ${v === "evolution" ? `<ul class="legend">${m.arch.history_only ? `<li class="keep">published record</li><li class="discard">re-timing</li>` : `<li class="keep">kept</li><li class="discard">discarded</li><li class="crash">crashed</li><li class="next">queued</li>`}</ul>` : ""}
    </div>
    ${v === "baseline" ? `<div class="architecture-toolbar"><p class="muted small">Recorded baseline values</p><div class="architecture-switch" role="group" aria-label="Architecture display"><button type="button" data-architecture="diagram" aria-pressed="${S.ui.architecture !== "components"}">Diagram</button><button type="button" data-architecture="components" aria-pressed="${S.ui.architecture === "components"}">Components</button></div></div>` : ""}
    ${v === "baseline" && S.ui.architecture !== "components" ? renderArchitectureDiagram(m, v) : m.groups.map((g) => `
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
      v === "evolution" && tried ? `<small>${tried} ${S.m.arch.history_only ? "milestones" : "tried"}</small>` : ""}${v === "current" && c.current ? `<small class="good">changed</small>` : ""}</button>
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
  // The top is the baseline or the 95th percentile, whichever is higher; rarer, worse runs sit on the top edge.
  const sorted = [...vals].sort((a, b) => a - b);
  const top = Math.max(base, sorted[Math.floor(0.95 * (sorted.length - 1))]);
  const cap = top + Math.max((top - lo) * 0.06, top * 0.0005);
  const yLo = lo - (cap - lo) * 0.06, yHi = cap;
  const x = (i) => L + ((W - L - R) * i) / Math.max(1, exps.length - 1);
  const logScale = S.m.arch.metric?.scale === "log" && lo > 0;
  const axisLow = logScale ? Math.log10(lo) - 0.04 : yLo;
  const axisHigh = logScale ? Math.log10(cap) : yHi;
  const axisValue = (val) => logScale ? Math.log10(val) : val;
  const y = (val) => T + ((H - T - B) * (axisHigh - axisValue(Math.min(val, yHi)))) / (axisHigh - axisLow);
  const out = [];
  const ticks = 4;
  for (let k = 0; k <= ticks; k++) {
    const tick = axisLow + ((axisHigh - axisLow) * k) / ticks;
    const val = logScale ? 10 ** tick : tick;
    out.push(`<line class="ax-grid" x1="${L}" x2="${W - R}" y1="${y(val)}" y2="${y(val)}"/>`,
      `<text class="ax-label" x="${L - 6}" y="${y(val)}" dy="0.35em" text-anchor="end">${logScale ? val.toFixed(1) : val.toFixed(4)}</text>`);
  }
  const step = Math.max(1, Math.ceil(exps.length / 14));
  exps.forEach((e, i) => { if (i % step === 0 || i === exps.length - 1) out.push(`<text class="ax-label" x="${x(i)}" y="${H - 6}" text-anchor="middle">${e.n}</text>`); });
  let d = "", prev = null;
  exps.forEach((e, i) => {
    if (e.best_after == null) return;
    d += prev == null || e.comparison_reset ? `M${x(i)},${y(e.best_after)}` : `H${x(i)}V${y(e.best_after)}`;
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
  tip.innerHTML = `<b>#${e.n}</b> · ${esc(statusLabel(e))}${e.comp ? " · " + esc(e.comp.name) : ""}
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
  const el = $("#side-body");
  if (!el || !S.m) return;
  const sel = S.route.sel, m = S.m;
  $("#side-label").innerHTML = sel ? backBtn : `<span class="tag">Overview</span>`;
  $("#chart-label").textContent = metricName(m.arch) + (m.arch.history_only ? " by milestone · log scale" : " by experiment");
  let html;
  if (sel && sel.startsWith("e")) html = expDetail(m.exps.find((e) => "e" + e.n === sel));
  else if (sel && sel.startsWith("c:")) html = compDetail(m.byId.get(sel.slice(2)));
  else if (sel && sel.startsWith("i:")) html = ideaDetail(m.ideas.find((i) => i.id === sel.slice(2)));
  const keep = el.dataset.selection === String(sel) ? el.scrollTop : 0;  // a live update keeps the reader's place
  el.innerHTML = html || overview();
  el.dataset.selection = String(sel);
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
    ${safeUrl(arch.source) ? `<p><a href="${esc(safeUrl(arch.source))}" target="_blank" rel="noopener noreferrer">Published benchmark and rules ↗</a></p>` : ""}
    ${samples}
    ${!arch.history_only ? `<div class="sec"><h4>Hypotheses <span class="muted">${hyps.filter((h) => h.status === "validated").length} validated · ${hyps.filter((h) => h.status === "invalidated").length} invalidated</span></h4>
      <ul class="hyps">${hyps.map((h) => `<li class="${esc(h.status)}"><span class="hs" title="${esc(h.status)}">${VERDICT[h.status] || "?"}</span>
        <span>${esc(h.text)}</span><span class="ev">${refChips(h.evidence || [])}</span></li>`).join("") || `<li class="muted">None yet.</li>`}</ul></div>` : ""}
    ${m.ideas.length ? `<div class="sec"><h4>Next up <span class="muted">${m.ideas.length} queued</span></h4>
      <ul class="ideas">${m.ideas.slice(0, 8).map((i) => `<li><button type="button" data-sel="i:${esc(i.id)}">${esc(i.title)}</button>
        <span class="muted">${esc(i.comp ? i.comp.name : humanize(i.category))}</span></li>`).join("")}</ul></div>` : ""}
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
    return `<div class="d-head"><span class="tag">#0</span><span class="pill keep">baseline</span></div>
      <h3>Baseline</h3>${metricsRow(e)}${provenanceDetail(e)}${samplesFig(e)}
      <div class="sec"><h4>Components</h4><ul class="ideas">${S.m.comps.map((c) =>
        `<li><button type="button" data-sel="c:${esc(c.id)}">${esc(c.name)}</button><span class="muted">${esc(c.baseline.label || "")}</span></li>`).join("")}</ul></div>`;
  }
  const idea = e.idea || {}, out = idea.outcome || {}, pred = idea.prediction || {};
  const actual = out.delta_pct ?? e.pct;
  const diffUrl = e.files["change.diff"];
  // Upstream autoresearch resets discarded experiments in git, so only kept ones have code to show.
  const codeLost = !diffUrl && S.m.exps.some((x) => x.files["change.diff"]);
  const sameAsLabel = e.description.replace(/\s*\(.*?\)\s*$/, "").toLowerCase() === String(e.label).toLowerCase();
  const lessons = ((S.data.harness && S.data.harness.lessons) || []).filter((l) => new RegExp(`\\bexps? [\\d,\\s]*\\b${e.n}\\b`).test(l.text));
  return `
    <div class="d-head"><span class="tag">#${e.n}</span><span class="pill ${esc(e.status)}">${esc(statusLabel(e))}</span>
      ${e.comp ? `<button type="button" class="chip" data-sel="c:${esc(e.comp.id)}">${esc(e.comp.name)}</button>` : ""}</div>
    <h3>${esc(e.label)}</h3>
    ${sameAsLabel ? "" : `<p class="muted">${esc(e.description)}</p>`}
    ${metricsRow(e)}${provenanceDetail(e)}
    <div class="sec"><h4>${S.m.arch.history_only ? "Published change" : "Change"}</h4>
      <div class="ba"><span class="before">${esc(e.before || "baseline")}</span><span class="arrow">→</span><span class="after">${esc(e.change.after || e.label)}</span></div>
      ${e.change.cost ? `<p class="cost mono">${esc(e.change.cost)}</p>` : ""}
      ${e.change.note ? `<p>${esc(e.change.note)}</p>` : ""}</div>
    ${idea.rationale || pred.delta_pct != null ? `<div class="sec"><h4>Hypothesis ${e.verdict ? `<span class="verdict ${e.verdict}">${VERDICT[e.verdict]} ${e.verdict}</span>` : ""}</h4>
      ${idea.rationale ? `<p class="clamp">${esc(idea.rationale)}</p>` : ""}
      ${pred.delta_pct != null ? `<p class="pred mono">predicted <b class="${tone(pred.delta_pct)}">${fmtPct(pred.delta_pct)}</b> · actual <b class="${tone(actual)}">${e.status === "crash" ? "crashed" : fmtPct(actual)}</b>${pred.confidence ? ` · ${esc(pred.confidence)} confidence` : ""}</p>` : ""}</div>` : ""}
    ${out.prediction_check || lessons.length ? `<div class="sec"><h4>Why</h4>
      ${out.prediction_check ? `<p>${esc(out.prediction_check)}</p>` : ""}
      ${lessons.map((l) => `<p class="lesson">${lessonHtml(l.text)}</p>`).join("")}</div>` : ""}
    ${diffUrl ? `<details class="sec code" data-url="${esc(diffUrl)}"><summary><h4>Code <span class="mono"><span class="good">+${e.diff ? e.diff.add : "?"}</span> <span class="worse">−${e.diff ? e.diff.del : "?"}</span> lines</span></h4></summary><div class="diff"></div></details>`
      : S.m.arch.history_only ? `<p class="muted small">No comparable source diff was extracted for this milestone. The upstream evidence is linked above.</p>` : codeLost ? `<p class="muted small">No code for this run: discarded experiments are reset in git, so only kept changes keep their diff.</p>` : ""}
    ${samplesFig(e)}`;
}

function provenanceDetail(e) {
  const p = e.idea?.provenance;
  if (!p) return "";
  const links = (p.sources || []).filter((s) => safeUrl(s.url));
  return `<div class="sec"><h4>Upstream record #${esc(p.number)}${p.kind === "retime" ? " · re-timing" : ""}</h4>
    <p class="mono">${esc(p.date)} · target val_loss ≤ ${esc(e.metrics?.target_val_loss ?? "—")}</p>
    <p>${esc(p.contributors)}</p><p class="muted small">${esc(p.evidence)}</p>
    <p class="muted small">${esc(p.comparison_note)}</p>
    ${p.diff_from_record != null ? `<p class="small">Code compared with published record #${esc(p.diff_from_record)}.</p>` : ""}
    <ul class="source-links">${links.map((s) => `<li><a href="${esc(safeUrl(s.url))}" target="_blank" rel="noopener noreferrer">${esc(s.label)} ↗</a></li>`).join("")}</ul></div>`;
}

function metricsRow(e) {
  const prev = e.bestExpBefore && e.bestExpBefore.metrics, cur = e.metrics || {};
  const delta = (a, b, f) => (a != null && b != null && a !== b ? ` <small class="${a > b ? "up" : "down"}">${a > b ? "+" : "−"}${f(Math.abs(a - b))}</small>` : "");
  return `<div class="metrics">
    <div><span>${esc(metricName(S.m.arch))}</span><b class="mono">${fmtLoss(e.val_loss)}</b></div>
    ${e.n ? `<div><span>vs best before</span><b class="mono ${tone(e.pct)}">${e.status === "crash" ? "—" : fmtPct(e.pct)}</b></div>` : ""}
    ${cur.num_steps != null ? `<div><span>steps</span><b class="mono">${cur.num_steps}${prev ? delta(cur.num_steps, prev.num_steps, (x) => x) : ""}</b></div>` : ""}
    ${cur.num_params_M != null ? `<div><span>params</span><b class="mono">${cur.num_params_M}M${prev ? delta(cur.num_params_M, prev.num_params_M, (x) => x.toFixed(2) + "M") : ""}</b></div>` : ""}
    ${cur.num_params_M == null && e.memory_gb ? `<div><span>memory</span><b class="mono">${e.memory_gb} GB${e.bestExpBefore && e.bestExpBefore.memory_gb ? delta(e.memory_gb, e.bestExpBefore.memory_gb, (x) => x.toFixed(1)) : ""}</b></div>` : ""}
  </div>`;
}

const samplesFig = (e) => (e.files && e.files["samples.png"] ? `<figure class="samples"><img src="${esc(e.files["samples.png"])}" alt="samples from run #${e.n}" loading="lazy">
  <figcaption>${esc((S.data.prompt && `“${S.data.prompt}”`) || "samples")} · fixed seeds</figcaption></figure>` : "");

function compDetail(c) {
  if (!c) return null;
  const b = c.baseline || {};
  return `<div class="d-head"><span class="tag">component</span>${c.current ? `<span class="pill keep">changed</span>` : `<span class="pill">unchanged</span>`}</div>
    <h3>${esc(c.name)}</h3>
    <div class="sec"><h4>Baseline</h4><p><b>${esc(b.label || "baseline")}</b></p>${b.detail ? `<p>${esc(b.detail)}</p>` : ""}
      ${b.cost || b.params ? `<p class="cost mono">${esc([b.cost, b.params && "params " + b.params].filter(Boolean).join(" · "))}</p>` : ""}</div>
    ${c.current ? `<div class="sec"><h4>Now</h4><div class="ba"><span class="before">${esc(b.label)}</span><span class="arrow">→</span><span class="after">${esc(c.currentLabel)}</span></div>
      <p class="muted">since <button type="button" class="chip" data-sel="e${c.current.n}">#${c.current.n}</button></p></div>` : ""}
    <div class="sec"><h4>${S.m.arch.history_only ? "Milestones" : "Tried"} <span class="muted">${c.steps.length}</span></h4>
      <ul class="steps">${c.steps.map((e) => `<li class="${esc(e.status)}"><button type="button" data-sel="e${e.n}"><span class="tag">#${e.n} ${ICON[e.status] || ""}</span>${esc(e.label)}</button>
        <span class="mono ${tone(e.pct)}">${e.status === "crash" ? "crashed" : fmtPct(e.pct)}</span></li>`).join("") || `<li class="muted">Nothing yet.</li>`}</ul></div>
    ${c.next.length ? `<div class="sec"><h4>Queued</h4><ul class="ideas">${c.next.map((i) => `<li><button type="button" data-sel="i:${esc(i.id)}">${esc(i.title)}</button></li>`).join("")}</ul></div>` : ""}`;
}

function ideaDetail(i) {
  if (!i) return null;
  const links = (i.sources || []).map(safeUrl).filter(Boolean);
  return `<div class="d-head"><span class="tag">${esc(i.id)}</span><span class="pill next">queued</span>
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

// Standalone training uses its own durable jobs; it does not edit or score published history.
function renderTraining() {
  const el = $("#training-panel"), t = S.agents?.training;
  if (!el || !t) return;
  if (el.dataset.project !== S.pid) {
    const c = t.config;
    el.dataset.project = S.pid;
    el.innerHTML = `
      <div class="training-heading"><div><h3>Start training</h3><p class="muted">Run the current model once. Configure a machine and start training. Setup is checked automatically before launch.</p></div><span class="pill" id="training-badge">Ready to configure</span></div>
      <div class="training-grid">
        <form id="training-form" class="training-form">
          <div class="training-fields">
            <label>Training profile<select name="profile"><option value="project">Project command</option><option value="modded-nanogpt">modded-nanogpt · 8×H100</option><option value="custom">Custom command</option></select></label>
            <label>Run on<select name="target"><option value="local">This computer</option><option value="ssh">Remote machine · SSH</option></select></label>
            <label class="training-host">SSH host<input name="host" placeholder="gpu-box or user@hostname" autocomplete="off" spellcheck="false"></label>
            <label>Time limit · minutes<input name="timeout_minutes" type="number" min="1" max="1440" required></label>
            <label class="full">Working directory<input name="directory" placeholder="/path/to/project" spellcheck="false" required></label>
            <label class="full">Training command<input class="mono" name="command" spellcheck="false" required></label>
          </div>
          <p class="muted small" id="training-help"></p>
          <div class="training-actions"><button type="button" data-training="save">Save setup</button><button type="button" data-training="check">Check setup</button><button type="submit" class="primary" id="training-start" disabled>Start training</button></div>
          <div id="training-feedback" role="status" aria-live="polite"></div>
        </form>
        <div class="training-monitor"><div class="training-monitor-head"><h4>Training output</h4><div id="training-run-actions"></div></div>
          <pre class="training-log mono" id="training-log">No training runs yet. Start training to check the setup and launch.</pre>
          <p id="training-job-message" class="muted small"></p><div id="training-history"></div>
        </div>
      </div>`;
    const form = $("#training-form");
    for (const [key, value] of Object.entries(c)) if (form.elements[key]) form.elements[key].value = value;
    form.addEventListener("input", () => { form.dataset.checked = ""; trainingFields(); });
    form.addEventListener("change", (event) => {
      if (form.elements.profile.value === "modded-nanogpt") {
        if (event.target.name === "profile") form.elements.command.value = "uv run --no-project torchrun --standalone --nproc_per_node=8 train_gpt.py";
        if (["target", "profile"].includes(event.target.name)) form.elements.directory.value = form.elements.target.value === "ssh" ? "~/modded-nanogpt" : t.source_directory;
      }
      form.dataset.checked = ""; trainingFields();
    });
    form.addEventListener("submit", (event) => { event.preventDefault(); trainingAction("start"); });
    el.addEventListener("click", (event) => {
      const button = event.target.closest("[data-training]");
      if (button) trainingAction(button.dataset.training);
    });
    trainingFields();
  }
  const job = t.latest, busy = job && ["queued", "running", "stopping"].includes(job.status);
  const badge = $("#training-badge");
  badge.textContent = job ? humanize(job.status) : "Ready to configure";
  badge.className = "pill" + (job?.status === "completed" ? " keep" : ["failed", "connection_lost", "interrupted"].includes(job?.status) ? " crash" : "");
  const log = $("#training-log"), atEnd = log.scrollHeight - log.scrollTop - log.clientHeight < 35;
  const text = t.log || (busy ? "Starting the training process…" : "No output yet. Start training checks the setup before launching.");
  if (log.textContent !== text) { log.textContent = text; if (atEnd) log.scrollTop = log.scrollHeight; }
  $("#training-job-message").textContent = job?.message || (busy ? `Started ${new Date(job.started_at * 1000).toLocaleTimeString()} · ${job.config.target === "ssh" ? job.config.host : "this computer"}` : "");
  $("#training-run-actions").innerHTML = `${job ? `<a href="/api/p/${S.pid}/training-log/${encodeURIComponent(job.id)}" target="_blank" rel="noopener">Full log ↗</a>` : ""}${busy ? `<button type="button" class="danger" data-training="stop" ${job.status === "stopping" ? "disabled" : ""}>${job.status === "stopping" ? "Stopping…" : "Stop training"}</button>` : ""}`;
  $("#training-history").innerHTML = t.jobs.length > 1 ? `<details><summary>Previous runs · ${t.jobs.length - 1}</summary><ul>${t.jobs.slice(1, 6).map(j => `<li><a href="/api/p/${S.pid}/training-log/${encodeURIComponent(j.id)}" target="_blank" rel="noopener">${esc(j.id)}</a><span>${esc(humanize(j.status))}</span></li>`).join("")}</ul></details>` : "";
  trainingFields();
}

function trainingFields() {
  const form = $("#training-form");
  if (!form) return;
  const remote = form.elements.target.value === "ssh";
  $(".training-host", form).hidden = !remote;
  $("#training-help").textContent = remote
    ? "Uses your SSH keys or host alias. The checkout, uv environment, and dataset must already exist on the remote machine. Files are not uploaded."
    : "Runs in this directory using the current source. The command must include its executable and arguments; shell redirection is not supported.";
  const busy = ["queued", "running", "stopping"].includes(S.agents?.training?.latest?.status);
  // Start performs its own server-side preflight; a separate check is optional.
  $("#training-start").disabled = !!form.dataset.pending || busy || researchBusy();
  $("#training-start").textContent = form.dataset.pending === "start" ? "Checking and starting…" : "Start training";
  form.elements.host.required = remote;
  form.querySelectorAll('input, select, [data-training]').forEach(e => e.disabled = !!form.dataset.pending || busy);
}

function trainingFeedback(result, message) {
  const el = $("#training-feedback");
  if (!el) return;
  el.innerHTML = result?.checks ? `<ul class="training-checks">${result.checks.map(c => `<li class="${c.ok ? "good" : "worse"}"><span>${c.ok ? "✓" : "×"} ${esc(c.name)}</span><small>${esc(c.detail)}</small></li>`).join("")}</ul><p>${result.ready ? "Setup checked. Start training when you’re ready." : "Resolve the failed checks, then check again."}</p>` : `<p>${esc(message || "")}</p>`;
}

async function trainingAction(action) {
  const form = $("#training-form"), pid = S.pid;
  if (!form || form.dataset.pending) return;
  if (action === "start" && !form.reportValidity()) return;
  // Read values before disabling controls: disabled inputs are omitted from FormData.
  const config = Object.fromEntries(new FormData(form));
  config.timeout_minutes = Number(config.timeout_minutes);
  form.dataset.pending = action;
  trainingFields();
  trainingFeedback(null, ({save:"Saving setup…",check:"Checking the training environment…",start:"Checking setup and starting training…",stop:"Requesting stop…"})[action]);
  try {
    const body = action === "stop" ? {job_id:S.agents.training.latest.id} : {config};
    const result = await post(`/api/p/${pid}/training-${action}`, body);
    if (S.pid !== pid || !form.isConnected) return;
    if (action === "check") { form.dataset.checked = result.ready ? "yes" : ""; trainingFeedback(result); }
    else if (action === "start" && !result.started) { form.dataset.checked = ""; trainingFeedback(result.preflight); }
    else trainingFeedback(null, ({save:"Setup saved.",start:"Training started. Output will appear here.",stop:"Stop requested. Waiting for the process to exit."})[action]);
    S.agentsRaw = "";
    await poll(false);
  } catch (error) {
    if (S.pid === pid && form.isConnected) trainingFeedback(null, error.message);
  } finally {
    delete form.dataset.pending;
    if (S.pid === pid) trainingFields();
  }
}

// ------------------------------------------------------------------ agents page

// Fixed vector geometry keeps research icons legible across fonts and platforms.
const RESEARCH_ICON_PATHS = {
  model: '<rect x="6" y="6" width="12" height="12" rx="2"/><rect x="9" y="9" width="6" height="6" rx="1"/><path d="M9 3v3m6-3v3M9 18v3m6-3v3M3 9h3m-3 6h3m12-6h3m-3 6h3"/>',
  tool: '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="m7 9 3 3-3 3m6 0h4"/>',
  delegation: '<circle cx="12" cy="5" r="2"/><circle cx="5" cy="19" r="2"/><circle cx="19" cy="19" r="2"/><path d="M12 7v5M5 17v-3a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2v3"/>',
  experiment: '<path d="M9 3h6m-5 0v6l-5.5 9a2 2 0 0 0 1.7 3h11.6a2 2 0 0 0 1.7-3L14 9V3M8 15h8"/>',
  lifecycle: '<circle cx="12" cy="12" r="9"/><path d="m16 8-2.5 5.5L8 16l2.5-5.5Z"/>',
  file: '<path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9Zm0 0v6h6M8 13h8m-8 4h6"/>',
  parameters: '<path d="M4 7h6m4 0h6M4 17h10m4 0h2"/><circle cx="12" cy="7" r="2"/><circle cx="16" cy="17" r="2"/>',
  launch: '<path d="M7 17 17 7M7 7h10v10"/>',
  down: '<path d="M12 4v16m-6-6 6 6 6-6"/>',
  linked: '<path d="M5 4v8a4 4 0 0 0 4 4h10m-5-5 5 5-5 5"/>',
  completed: '<path d="m5 12 4 4L19 6"/>',
  failed: '<path d="m6 6 12 12M6 18 18 6"/>',
  stopped: '<rect x="6" y="6" width="12" height="12" rx="2"/>',
};
function researchIcon(name) {
  return `<svg class="research-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false">${RESEARCH_ICON_PATHS[name] || RESEARCH_ICON_PATHS.lifecycle}</svg>`;
}
const RESEARCH_KINDS = {
  model: { key: "model", label: "Model", icon: researchIcon("model") },
  tool: { key: "tool", label: "Tool", icon: researchIcon("tool") },
  delegation: { key: "delegation", label: "Delegation", icon: researchIcon("delegation") },
  experiment: { key: "experiment", label: "Experiment", icon: researchIcon("experiment") },
  lifecycle: { key: "lifecycle", label: "Research", icon: researchIcon("lifecycle") },
};
const eventKind = type => RESEARCH_KINDS[type] || RESEARCH_KINDS.lifecycle;
const eventTime = ts => new Date(typeof ts === "number" ? ts * 1000 : ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
function researchBusy() {
  const r = S.agents?.research;
  return !!r?.active || ["queued", "running", "stopping", "orphaned"].includes(r?.latest?.status);
}

function renderResearch() {
  const el = $("#research-panel"), r = S.agents?.research;
  if (!el) return;
  if (!r) { el.innerHTML = '<h3>Agentic research</h3><p class="muted">Research controls are not available from this server.</p>'; return; }
  if (el.dataset.project !== S.pid) {
    el.dataset.project = S.pid;
    const c = researchBusy() ? { ...(r.config || {}), ...(r.latest?.config || {}) } : r.config || {};
    el.innerHTML = `<div class="research-heading"><div><span class="tag">Research workspace</span><h2>Agentic research</h2><p class="muted">Give the agents an objective for parameter research. They inspect the project, delegate analysis, and test model and training parameters against a fixed evaluation.</p></div><span id="research-badge" class="pill">Ready</span></div>
      <form id="research-form" class="research-form">
        <label class="research-objective">Research objective<textarea name="objective" rows="2" maxlength="4000" required placeholder="Improve the validation metric within the experiment budget.">${esc(c.objective || "Improve the validation metric while preserving the evaluation setup.")}</textarea></label>
        <div class="research-budget"><label>Candidate experiments<input name="max_experiments" type="number" min="1" max="20" required value="${esc(c.max_experiments ?? 3)}"></label><label>Minutes / experiment<input name="experiment_timeout_minutes" type="number" min="1" max="120" required value="${esc(c.experiment_timeout_minutes ?? 10)}"></label></div>
        <div class="research-launch"><button type="submit" id="research-start" class="primary">Start agentic research ${researchIcon("launch")}</button><button type="button" id="research-stop" class="danger" hidden>Stop research</button><p class="muted small">One baseline run is also measured. Uses the configured model provider and training environment.</p></div>
        <div id="research-feedback" role="status" aria-live="polite"></div>
      </form>
      <p class="research-support muted small" id="research-support"></p>`;
    $("#research-form").addEventListener("submit", event => { event.preventDefault(); researchAction("start"); });
    $("#research-stop").addEventListener("click", () => researchAction("stop"));
  }
  // Active session settings are immutable. Show the actual launch config on reload
  // or when another client starts a session, while leaving idle form edits intact.
  if (researchBusy() && r.latest?.config && el.dataset.researchRun !== r.latest.id) {
    const form = $("#research-form");
    for (const key of ["objective", "max_experiments", "experiment_timeout_minutes"]) {
      if (r.latest.config[key] != null) form.elements[key].value = r.latest.config[key];
    }
    el.dataset.researchRun = r.latest.id;
  }
  const badge = $("#research-badge"), state = r.latest?.status;
  badge.textContent = state === "orphaned" ? "Needs attention" : researchBusy() ? (state === "stopping" ? "Stopping" : "Running") : state ? humanize(state) : r.supported ? "Ready" : "Setup needed";
  badge.className = "pill" + (["failed", "interrupted", "orphaned"].includes(state) ? " crash" : researchBusy() ? " next" : state === "completed" ? " keep" : "");
  const params = r.parameters || r.config?.parameters || {};
  const names = Array.isArray(params) ? params.map(p => typeof p === "string" ? p : p.name || p.key).filter(Boolean) : Object.keys(params);
  $("#research-support").textContent = !r.supported ? r.reason || "This project needs a research configuration before it can run." : names.length ? `Tunable parameters: ${names.join(", ")}. Evaluation remains fixed.` : r.reason || "Parameter experiments use this project’s configured evaluation.";
  researchFields();
}

function researchFields() {
  const form = $("#research-form");
  if (!form) return;
  const r = S.agents?.research, pending = !!form.dataset.pending, busy = researchBusy();
  const training = ["queued", "running", "stopping"].includes(S.agents?.training?.latest?.status);
  const legacy = !!S.agents?.running || (!!S.agents?.status?.phase && S.agents.status.phase !== "stopped");
  $("#research-start").disabled = pending || busy || training || legacy || !r?.supported;
  $("#research-start").innerHTML = form.dataset.pending === "start" ? "Checking and starting…" : `Start agentic research ${researchIcon("launch")}`;
  $("#research-stop").hidden = !busy;
  $("#research-stop").disabled = pending || r?.latest?.status === "stopping";
  $("#research-stop").textContent = r?.latest?.status === "stopping" ? "Stopping…" : r?.latest?.status === "orphaned" ? "Stop evaluator" : "Stop research";
  form.querySelectorAll("input, textarea").forEach(input => input.disabled = pending || busy);
  $("#research-start").title = training ? "Stop the standalone training run before starting research." : legacy ? "Stop the active harness loop before starting research." : "";
}

async function researchAction(action) {
  const form = $("#research-form"), pid = S.pid;
  if (!form || form.dataset.pending || (action === "start" && !form.reportValidity())) return;
  const values = Object.fromEntries(new FormData(form));
  const body = action === "stop" ? { run_id: S.agents.research.latest?.id } : {
    objective: values.objective, max_experiments: Number(values.max_experiments), experiment_timeout_minutes: Number(values.experiment_timeout_minutes),
  };
  form.dataset.pending = action;
  researchFields();
  const feedback = $("#research-feedback");
  feedback.className = "muted small";
  feedback.textContent = action === "start" ? "Checking the research setup…" : "Requesting stop…";
  try {
    const result = await post(`/api/p/${pid}/research-${action}`, body);
    if (S.pid !== pid || !form.isConnected) return;
    if (action === "start" && result.started === false) {
      feedback.className = "worse small";
      feedback.textContent = result.reason || result.error || result.message || "Research could not start. Check the project configuration.";
    } else feedback.textContent = action === "start" ? "Research started. Follow the agents in Activity below." : "Stop requested. Waiting for the active work to finish stopping.";
    S.agentsRaw = "";
    await poll(false);
  } catch (error) {
    if (S.pid === pid && form.isConnected) { feedback.className = "worse small"; feedback.textContent = error.message; }
  } finally {
    delete form.dataset.pending;
    if (S.pid === pid) researchFields();
  }
}

function renderAgentDefinitions() {
  const a = S.agents, r = a.research, el = $("#ag-designs");
  const status = a.status && a.status.phase && a.status.phase !== "stopped" ? a.status : null;
  const busy = a.running || !!status;
  const agentic = (a.designs || []).find(d => d.id === "agentic");
  const models = agentic?.roles || [];
  const roles = r?.agents?.length ? r.agents : [
    { role: "Orchestrator", model: models.find(r => r.role === "orchestrator")?.model, about: "Reads the project, chooses the next action, and proposes parameter experiments." },
    { role: "Advisors", model: (a.designs || []).flatMap(d => d.roles || []).find(r => r.role === "researcher")?.model, about: "Analyze delegated questions in parallel and return recommendations to the orchestrator." },
  ];
  const open = $("#legacy-designs", el)?.open || false;
  const timeValues = Object.fromEntries($$('input[type="time"]', el).map(input => [input.id, input.value]));
  const html = `<div class="research-definitions-heading"><h3>Agent definitions</h3><span class="muted small">Orchestrator → advisors → experiment → evaluation</span></div>
    <div class="research-roles">${roles.map(role => `<div class="research-role"><span class="role-mark" aria-hidden="true">${researchIcon(/advisor|subagent/i.test(role.role || role.name) ? "delegation" : "model")}</span><div><b>${esc(role.role || role.name)}</b>${role.model ? `<span class="model mono" title="${esc(role.model)}">${esc(String(role.model).split("/").pop())}</span>` : ""}<p>${esc(role.about || role.description || "")}</p></div></div>`).join("")}</div>
    <div class="research-tools"><span class="tag">Available actions</span><span>${researchIcon("file")} Read files</span><span>${researchIcon("delegation")} Delegate analysis</span><span>${researchIcon("parameters")} Propose parameters</span><span>${researchIcon("experiment")} Run & evaluate</span></div>
    ${(a.designs || []).length ? `<details class="legacy-designs" id="legacy-designs" ${open ? "open" : ""}><summary>${a.managed ? "Additional research harnesses" : "Harness designs & prompts"}</summary><p class="muted small">${a.managed ? "These harnesses use their own workspaces, prompts, and research loops." : "These designs are available for configured harness workspaces. The research controls above use this project’s parameter experiments."}</p><div class="legacy-design-grid">${a.designs.map(d => {
      const mine = d.id === a.design;
      return `<div class="dcard${mine ? " on" : ""}"><div class="dname"><b>${esc(d.name)}</b><span class="tag">${esc(d.id)}</span>${mine ? '<span class="pill keep">this project</span>' : ""}</div><p>${esc(d.about)}</p>
        <ol class="roles">${d.roles.map(role => `<li><button type="button" class="role${S.detail?.key === role.prompt ? " on" : ""}" data-prompt="${esc(role.prompt || "")}" ${role.prompt ? "" : "disabled"}>${esc(role.role)}</button><span class="mono model" title="${esc(role.model)}">${esc(String(role.model).split("/").pop())}</span><small>${esc(role.about)}</small></li>`).join("")}</ol>
        ${a.managed ? `<div class="dact">${mine ? busy ? `<button type="button" data-act="stop" ${a.stop_requested ? "disabled" : ""}>${a.stop_requested ? "Stopping after this experiment…" : "Stop after this experiment"}</button>${a.running ? '<button type="button" data-act="kill" class="danger">Stop now</button>' : ""}` : `<label>until <input type="time" id="until-${esc(d.id)}"></label><button type="button" class="primary" data-act="start" data-design="${esc(d.id)}" ${researchBusy() ? "disabled" : ""}>Start harness</button>` : `${d.project ? `<a href="#/p/${esc(d.project)}/agents">Open its workspace</a>` : ""}<label>until <input type="time" id="until-${esc(d.id)}"></label><button type="button" data-act="start" data-design="${esc(d.id)}" ${researchBusy() ? "disabled" : ""}>${d.exists ? "Start harness" : "Start in a new workspace"}</button>`}</div>` : ""}</div>`;
    }).join("")}</div></details>` : ""}`;
  if (el._definitionsHTML !== html) {
    el.innerHTML = html;
    el._definitionsHTML = html;
    for (const [id, value] of Object.entries(timeValues)) if (document.getElementById(id)) document.getElementById(id).value = value;
  }
}

// Each card is an emitted activity, not a reconstruction of a model's private thoughts.
// Operation IDs link starts and completions while parent IDs express delegation.
function activeOperations(events) {
  const active = new Map();
  for (const event of events) {
    const key = event.data?.operation_id || event.operation_id || event.id;
    if (event.phase === "started") active.set(key, event);
    else if (["completed", "failed", "stopped"].includes(event.phase)) {
      active.delete(key);
    }
  }
  return [...active.values()].filter(e => e.type !== "lifecycle");
}

function renderActivity() {
  const el = $("#ag-live"), a = S.agents;
  if (!el || !a) return;
  if (!$("#activity-timeline", el)) {
    el.innerHTML = `<div class="activity-heading"><h3>Activity <span id="activity-status" class="status"></span></h3><span class="activity-refresh muted small">Updates every second</span></div>
      <div id="activity-agents" class="activity-agents"></div>
      <div class="activity-toolbar"><div class="activity-filters" role="group" aria-label="Filter activity">${[["all","All"],["model","Models"],["tool","Tools"],["delegation","Delegation"],["experiment","Experiments"]].map(([key,label]) => `<button type="button" data-activity-filter="${key}" aria-pressed="${key === "all"}">${label}</button>`).join("")}</div><button type="button" id="activity-follow" class="chip" aria-pressed="true">Following live ${researchIcon("down")}</button></div>
      <p class="activity-announcement muted small" id="activity-announcement" role="status" aria-live="polite"></p>
      <div id="activity-experiment" class="activity-experiment" hidden></div>
      <div class="activity-timeline" id="activity-timeline" tabindex="0" aria-label="Research activity timeline"></div>
      <div id="activity-legacy"></div>`;
    el.dataset.filter = "all";
    el.dataset.follow = "yes";
    el.addEventListener("click", event => {
      const filter = event.target.closest("[data-activity-filter]");
      if (filter) { el.dataset.filter = filter.dataset.activityFilter; renderActivity(); }
      if (event.target.closest("#activity-follow")) { el.dataset.follow = el.dataset.follow === "yes" ? "no" : "yes"; renderActivity(); }
    });
    $("#activity-timeline", el).addEventListener("scroll", event => {
      const timeline = event.target;
      if (timeline.scrollHeight - timeline.scrollTop - timeline.clientHeight > 40) {
        el.dataset.follow = "no";
        $("#activity-follow", el).innerHTML = `Follow live ${researchIcon("down")}`;
        $("#activity-follow", el).setAttribute("aria-pressed", "false");
      }
    });
  }
  const r = a.research || {}, events = r.events || [], active = researchBusy();
  const orphaned = r.latest?.status === "orphaned";
  const operations = active && !orphaned ? activeOperations(events) : [];
  const legacyActive = a.running || (a.status?.phase && a.status.phase !== "stopped");
  const status = $("#activity-status", el);
  status.className = "status" + (orphaned ? " bad" : active || legacyActive ? " on" : "");
  status.innerHTML = `<span class="dot"></span>${esc(orphaned ? "Evaluator still running" : active ? r.latest?.status === "stopping" ? "Stopping" : "Live" : legacyActive ? humanize(a.status?.phase || "starting") : r.latest?.status ? humanize(r.latest.status) : "Idle")}`;
  const byActor = new Map();
  for (const operation of operations) byActor.set(operation.actor || "Research", operation);
  $("#activity-agents", el).innerHTML = byActor.size ? [...byActor].map(([actor, operation]) => {
    const kind = eventKind(operation.type);
    return `<div class="active-agent event-${kind.key}"><span class="agent-pulse" aria-hidden="true"></span><b>${esc(actor)}</b><span>${operation.type === "model" ? "Model working" : esc(operation.title || kind.label)}</span></div>`;
  }).join("") : orphaned ? '<div class="active-agent"><b>Evaluator</b><span>Worker disconnected · stop required</span></div>' : active ? '<div class="active-agent"><span class="agent-pulse" aria-hidden="true"></span><b>Research</b><span>Waiting for the next activity</span></div>' : "";
  $("#activity-announcement", el).textContent = r.latest?.message || (events.length ? `${events.length} recorded activities` : "Start research to see model calls, tools, delegation, and experiments here.");
  const monitor = $("#activity-experiment", el), progress = r.progress;
  const current = active && r.latest?.current_experiment && progress?.experiment_id === r.latest.current_experiment;
  const outputText = current ? progress.log || "Waiting for evaluator output…" : r.log || progress?.log || "";
  monitor.hidden = !current && !outputText;
  if (!monitor.hidden) {
    const open = $("details", monitor)?.open ?? !!current;
    const previousLog = $("pre", monitor), scrollTop = previousLog?.scrollTop || 0;
    const atEnd = !previousLog || previousLog.scrollHeight - previousLog.scrollTop - previousLog.clientHeight < 30;
    monitor.innerHTML = `<div class="research-experiment-heading"><b>${esc(orphaned ? "Evaluator still running" : current ? r.latest.current_step || "Experiment running" : "Research output")}</b>${current ? `<span class="muted mono">${orphaned ? "Last report · " : ""}${esc(ago(progress.elapsed_seconds || 0))} elapsed</span>` : ""}</div><details class="log-box" ${open ? "open" : ""}><summary>Evaluator output</summary><pre class="log mono">${esc(outputText)}</pre></details>`;
    const output = $("pre", monitor);
    output.scrollTop = atEnd ? output.scrollHeight : scrollTop;
  }
  const filter = el.dataset.filter;
  $$("[data-activity-filter]", el).forEach(button => button.setAttribute("aria-pressed", String(button.dataset.activityFilter === filter)));
  const timeline = $("#activity-timeline", el), scrollTop = timeline.scrollTop;
  const recent = events.filter(event => filter === "all" || event.type === filter).slice(-100);
  const html = recent.map(event => {
    const kind = eventKind(event.type), running = operations.some(op => op.id === event.id);
    const done = event.phase === "completed", failed = event.phase === "failed";
    return `<button type="button" class="activity-event event-${kind.key}${running ? " is-running" : ""}${failed ? " is-failed" : ""}${S.detail?.kind === "event" && S.detail.key === String(event.id) ? " selected" : ""}" data-event="${esc(event.id)}"><span class="event-icon" aria-hidden="true">${kind.icon}</span><span class="event-body"><span class="event-top"><span class="event-kind">${kind.label}</span><span class="event-actor">${esc(event.actor || "Research")}</span><time>${event.ts ? esc(eventTime(event.ts)) : ""}</time></span><strong>${esc(event.title || kind.label)}</strong>${event.detail ? `<span class="event-description">${esc(clip(event.detail, 240))}</span>` : ""}<span class="event-footer"><span class="event-phase">${running ? '<span class="event-spinner" aria-hidden="true"></span>' : failed ? researchIcon("failed") : done ? researchIcon("completed") : event.phase === "stopped" ? researchIcon("stopped") : ""}${running && event.type === "model" ? "Model working" : esc(humanize(event.phase || "info"))}</span>${event.duration_ms != null || event.data?.seconds != null ? `<span>${esc((event.duration_ms != null ? event.duration_ms / 1000 : event.data.seconds).toFixed(1))}s</span>` : ""}${event.parent_id ? `<span class="event-linked">${researchIcon("linked")} Linked activity</span>` : ""}</span></span></button>`;
  }).join("") || `<div class="activity-empty"><span aria-hidden="true">${eventKind(filter).icon}</span><b>${filter === "all" ? "Your research, as it happens" : "No " + (filter === "model" ? "model" : filter) + " activity yet"}</b><p>${filter === "all" ? "Follow the agents from an idea to a measured result. Each activity appears as it starts and finishes." : "Matching activities will appear here when they are recorded."}</p></div>`;
  if (timeline.innerHTML !== html) timeline.innerHTML = html;
  timeline.scrollTop = el.dataset.follow === "yes" ? timeline.scrollHeight : scrollTop;
  $("#activity-follow", el).innerHTML = `${el.dataset.follow === "yes" ? "Following live" : "Follow live"} ${researchIcon("down")}`;
  $("#activity-follow", el).setAttribute("aria-pressed", String(el.dataset.follow === "yes"));
  const legacy = $("#activity-legacy", el), open = $("details", legacy)?.open || !!legacyActive;
  legacy.innerHTML = (a.calls || []).length || (a.log || []).length || legacyActive ? `<details class="log-box" ${open ? "open" : ""}><summary>Harness activity${a.calls?.length ? " · " + a.calls.length + " calls" : ""}</summary><div class="calls">${(a.calls || []).map(call => `<button type="button" class="call${S.detail?.key === call.file ? " on" : ""}${call.error ? " err" : ""}" data-trace="${esc(call.file)}"><span class="call-role">${esc(call.role)}</span><span class="call-title">${esc(call.title || clip(call.preview, 90) || "(empty reply)")}</span><span class="call-meta mono">Experiment #${esc(call.exp)} · ${esc(eventTime(call.ts))} · ${esc(call.seconds ?? "?")}s</span></button>`).join("")}</div><pre class="log mono">${esc((a.log || []).join("\n")) || "No loop output yet."}</pre></details>` : "";
}

function renderAgents() {
  const nav = $("#agents-nav");
  if (nav) nav.innerHTML = viewTabs("agents");
  const el = $("#agents"), a = S.agents;
  if (!el || !a) return;
  if (!$("#ag-detail")) {
    el.innerHTML = '<section class="panel ag-col" id="ag-live"></section><section class="panel ag-col" id="ag-detail"></section>';
    renderDetail();
  }
  renderResearch();
  renderAgentDefinitions();
  renderActivity();
  renderTraining();
}

async function renderDetail() {
  const el = $("#ag-detail");
  if (!el) return;
  const d = S.detail;
  if (!d) { el.innerHTML = `<h3>Details</h3><p class="muted">Select an activity to inspect its recorded details, or a role to read its prompt.</p>`; return; }
  el.innerHTML = `<p class="muted">Loading…</p>`;
  el.scrollTop = 0;
  try {
    if (d.kind === "event") {
      const event = (S.agents?.research?.events || []).find(e => String(e.id) === d.key);
      if (!event) { el.innerHTML = '<h3>Activity details</h3><p class="muted">This activity is no longer in the recent feed.</p>'; return; }
      const kind = eventKind(event.type);
      el.innerHTML = `<h3>${esc(event.title || kind.label)}</h3>
        <div class="research-event-meta"><span class="event-kind event-${kind.key}">${kind.icon} ${kind.label}</span><span class="pill ${event.phase === "failed" ? "crash" : ""}">${esc(humanize(event.phase))}</span></div>
        <p class="muted small">${esc(event.actor || "Research")}${event.ts ? " · " + esc(eventTime(event.ts)) : ""}${(event.duration_ms != null || event.data?.seconds != null) ? " · " + esc((event.duration_ms != null ? event.duration_ms / 1000 : event.data.seconds).toFixed(1)) + "s" : ""}</p>
        ${event.detail ? `<p class="activity-detail">${esc(event.detail)}</p>` : ""}
        ${event.data && Object.keys(event.data).length ? `<details class="msg" open><summary>Recorded data</summary><pre class="mono">${esc(JSON.stringify(event.data, null, 2))}</pre></details>` : ""}
        ${event.parent_id ? `<p class="muted small">Linked activity: <span class="mono">${esc(event.parent_id)}</span></p>` : ""}`;
    } else if (d.kind === "prompt") {
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
  const display = t.closest("[data-architecture]");
  if (display && S.route.page === "project") {
    S.ui.architecture = display.dataset.architecture;
    try { localStorage.setItem("ar-ui", JSON.stringify(S.ui)); } catch (error) {}
    renderArch();
    $("[data-architecture='" + S.ui.architecture + "']")?.focus();
    return;
  }
  const tg = t.closest("[data-toggle]");
  if (tg) { setPanel(tg.dataset.toggle, !S.ui[tg.dataset.toggle]); return; }
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
  const activity = t.closest("[data-event]");
  if (activity) { S.detail = { kind: "event", key: activity.dataset.event }; renderActivity(); renderDetail(); return; }
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
  if (!["project", "eval", "agents"].includes(S.route.page) || !S.m) return;
  if (["1", "2", "3", "4", "5"].includes(k)) {
    const view = [...VIEWS, "agents"][+k - 1];
    location.hash = ["eval", "agents"].includes(view) ? `#/p/${S.pid}/${view}` : projHref(S.pid, view, S.route.sel);
    return;
  }
  if (S.route.page !== "project") return;
  if (k === "escape") select(null);
  else if (k === "arrowright" || k === "arrowleft") {
    const exps = S.m.exps, cur = S.route.sel && S.route.sel.startsWith("e") ? +S.route.sel.slice(1) : null;
    const i = cur == null ? (k === "arrowright" ? 0 : exps.length - 1) : exps.findIndex((x) => x.n === cur) + (k === "arrowright" ? 1 : -1);
    if (exps[i]) select("e" + exps[i].n);
    e.preventDefault();
  }
});

addEventListener("hashchange", onRoute);
onRoute();
