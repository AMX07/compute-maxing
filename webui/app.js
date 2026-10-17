"use strict";
// Presentation UI for autoresearch runs. Polls /api/runs and renders a progress chart, the
// experiment log, the selected run, and a two-run comparison. No dependencies.

const POLL_MS = 5000;
const $ = (sel, el = document) => el.querySelector(sel);

const state = {
  raw: "",
  data: null,
  items: [],          // experiments in order, then pending runs
  byKey: new Map(),
  sel: null,          // key of the selected item
  follow: true,       // auto-select the newest run until the presenter picks one
  view: "progress",
  cmpA: null, cmpB: null, cmpPinned: false,
  known: null,        // keys seen in the previous poll, to flash new results
  fresh: new Set(),
  diffs: new Map(),   // url -> text
  clockOffset: 0,     // server_time - local time
  detailSig: "", compareSig: "",
  error: null,
};

// ------------------------------------------------------------------ formatting

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmtLoss = (v) => (v == null ? "—" : v.toFixed(4));
function fmtPct(d, ref) {
  if (d == null || !ref) return "—";
  const p = (d / ref) * 100, a = Math.abs(p);
  const s = a >= 1 ? a.toFixed(1) : a.toFixed(2);
  return (p < 0 ? "−" : p > 0 ? "+" : "±") + s + "%";
}
const clip = (t, n) => (t.length > n ? t.slice(0, Math.max(1, n - 1)) + "…" : t);
const tone = (d) => (d == null ? "" : d < 0 ? "good" : "worse");
function fmtDur(s) {
  s = Math.max(0, Math.round(s));
  if (s < 60) return s + "s";
  const m = Math.floor(s / 60);
  if (m < 60) return m + "m " + String(s % 60).padStart(2, "0") + "s";
  return Math.floor(m / 60) + "h " + String(m % 60).padStart(2, "0") + "m";
}
const now = () => Date.now() / 1000 + state.clockOffset;
const statusLabel = { keep: "kept", discard: "discarded", crash: "crashed", pending: "not logged" };
const pill = (it) => `<span class="pill ${esc(it.status)}">${esc(statusLabel[it.status] || it.status)}</span>`;
const safeUrl = (u) => (/^https?:\/\//i.test(String(u)) ? String(u) : null);

// ------------------------------------------------------------------ data

function buildItems(data) {
  const s = data.summary;
  const items = data.experiments.map((e) => ({
    ...e,
    key: e.run_id || "row" + e.idx,
    n: e.idx,
    isBest: e.idx === s.best_idx,
    isBaseline: e.idx === 0,
  }));
  data.pending.forEach((p, j) => {
    const v = p.metrics && typeof p.metrics.val_loss === "number" ? p.metrics.val_loss : null;
    items.push({
      ...p,
      key: p.run_id,
      n: data.experiments.length + j,
      idx: null,
      pending: true,
      status: "pending",
      val_loss: v,
      description: (p.idea && p.idea.title) || "Finished, waiting for the agent to log it",
      delta_baseline: v != null && s.baseline_val_loss != null ? v - s.baseline_val_loss : null,
      delta_best: v != null && s.best_val_loss != null ? v - s.best_val_loss : null,
      best_before: s.best_val_loss,
      commit: p.metrics && p.metrics.commit,
    });
  });
  return items;
}

async function poll() {
  try {
    const res = await fetch("/api/runs", { cache: "no-store" });
    if (!res.ok) throw new Error("HTTP " + res.status);
    const text = await res.text();
    state.error = null;
    if (text !== state.raw) {
      state.raw = text;
      ingest(JSON.parse(text));
    }
  } catch (err) {
    state.error = "Can't reach the server (" + err.message + "). Retrying every 5 s.";
  }
  renderLive();
}

function ingest(data) {
  state.data = data;
  state.clockOffset = data.server_time - Date.now() / 1000;
  state.items = buildItems(data);
  state.byKey = new Map(state.items.map((it) => [it.key, it]));

  const keys = new Set(state.items.map((it) => it.key));
  state.fresh = new Set();
  if (state.known) for (const k of keys) if (!state.known.has(k) || wasPendingNowLogged(k)) state.fresh.add(k);
  state.known = keys;
  state.prevPending = new Set(state.items.filter((it) => it.pending).map((it) => it.key));

  if (state.follow || !state.byKey.has(state.sel)) state.sel = newestKey();
  if (!state.cmpPinned || !state.byKey.has(state.cmpA) || !state.byKey.has(state.cmpB)) setDefaultCompare();
  if (data.prompt) $("#prompt").textContent = data.prompt;
  renderAll();
  if (state.fresh.size) setTimeout(() => state.fresh.clear(), 4000);
}

function wasPendingNowLogged(k) {
  const it = state.byKey.get(k);
  return state.prevPending && state.prevPending.has(k) && it && !it.pending;
}

function newestKey() {
  for (let i = state.items.length - 1; i >= 0; i--) if (state.items[i].files && state.items[i].files["samples.png"]) return state.items[i].key;
  return state.items.length ? state.items[state.items.length - 1].key : null;
}

function setDefaultCompare() {
  const s = state.data.summary;
  const withImg = state.items.filter((it) => it.files["samples.png"]);
  const base = state.items.find((it) => it.isBaseline && it.files["samples.png"]) || withImg[0];
  const best = state.items.find((it) => it.idx === s.best_idx && it.files["samples.png"]) || withImg[withImg.length - 1];
  state.cmpA = base ? base.key : null;
  state.cmpB = best ? best.key : null;
  state.cmpPinned = false;
}

// ------------------------------------------------------------------ render: header

function renderAll() {
  renderKpis();
  renderLive();
  renderChart();
  renderList();
  renderDetail();
  renderCompare();
  renderPlaybook();
}

function renderKpis() {
  const s = state.data.summary, c = s.counts;
  const d = s.best_val_loss != null && s.baseline_val_loss != null ? s.best_val_loss - s.baseline_val_loss : null;
  const improvements = Math.max(0, (c.keep || 0) - (s.total ? 1 : 0)); // the baseline row is "keep" too
  $("#kpis").innerHTML = s.total
    ? `<div class="kpi"><span class="kpi-label">val_loss baseline → best</span>
         <span class="kpi-value">${fmtLoss(s.baseline_val_loss)}<span class="arrow">→</span>${fmtLoss(s.best_val_loss)}${
           d ? `<small class="good">${fmtPct(d, s.baseline_val_loss)}</small>` : ""}</span></div>
       <div class="kpi"><span class="kpi-label">${s.total} experiments</span>
         <span class="counts"><b>${improvements}</b> improved · <b>${c.discard || 0}</b> discarded · <b>${c.crash || 0}</b> crashed</span></div>`
    : `<div class="kpi"><span class="kpi-label">results.tsv</span><span class="counts">Waiting for the baseline run</span></div>`;
}

function renderLive() {
  const el = $("#live");
  if (state.error) {
    el.className = "live";
    el.innerHTML = `<span class="dot" style="background:var(--crash)"></span><span class="live-text">${esc(state.error)}</span>`;
    return;
  }
  if (!state.data) return;
  const d = state.data, live = d.live, pend = state.items.filter((it) => it.pending);
  const hs = d.harness && d.harness.status;
  let active = false, html;
  if (live && live.pct != null && !live.finished) {
    active = true;
    const n = hs && hs.phase === "training" ? hs.n : state.data.experiments.length + pend.length;
    const title = hs && hs.phase === "training" && hs.title ? ": " + esc(hs.title) : "";
    html = `<span class="live-text"><b>Training #${n}${title}</b> · ${live.pct.toFixed(0)}%${
      live.remaining_s != null ? " · " + fmtDur(live.remaining_s) + " left" : ""}${
      live.train_loss != null ? " · loss " + live.train_loss.toFixed(4) : ""}</span>
      <span class="bar"><i style="width:${Math.min(100, live.pct).toFixed(1)}%"></i></span>`;
  } else if (hs && hs.phase === "planning") {
    active = true;
    html = `<span class="live-text"><b>Planning #${hs.n}</b> · the planner is choosing the next idea (${fmtDur(now() - hs.since)})</span>`;
  } else if (pend.length) {
    const p = pend[pend.length - 1];
    active = true;
    html = p.stage === "sampling"
      ? `<span class="live-text"><b>#${p.n} sampling</b> the benchmark images…</span>`
      : `<span class="live-text"><b>#${p.n} finished</b>${p.val_loss != null ? " at " + fmtLoss(p.val_loss) : ""} · waiting for the agent to log it</span>`;
  } else if (hs && hs.phase === "reflecting") {
    active = true;
    html = `<span class="live-text"><b>Reflecting on #${hs.n}</b> · updating the playbook</span>`;
  } else if (live && !live.finished) {
    active = true;
    html = `<span class="live-text"><b>Running</b> · ${esc(live.line)}</span>`;
  } else {
    const last = state.items.filter((it) => it.started_at).pop();
    html = `<span class="live-text">${last ? `Idle · last run saved <b>${fmtDur(now() - last.started_at)}</b> ago` : "Waiting for the first run"}</span>`;
  }
  el.className = "live" + (active ? " active" : "");
  el.innerHTML = `<span class="dot"></span>${html}`;
  state.liveActive = active;
}

// ------------------------------------------------------------------ render: chart

function niceTicks(lo, hi, count) {
  const span = hi - lo || 1;
  const raw = span / count, mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= count) || 10 * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-6; v += step) out.push(+v.toFixed(10));
  return { ticks: out, step };
}

function renderChart() {
  const box = $("#chart");
  const items = state.items;
  const W = box.clientWidth, H = box.clientHeight;
  if (!W || !H) return;
  const rem = parseFloat(getComputedStyle(document.documentElement).fontSize);
  const vals = items.map((it) => it.val_loss).filter((v) => v != null);
  if (!vals.length) {
    box.innerHTML = `<div class="empty-state"><div><h3>No results yet</h3><p>The first point appears when the baseline run is logged to results.tsv.</p></div></div>`;
    return;
  }

  // y range: everything from the best result up to the baseline, plus moderate regressions.
  // Far-off results (diverged runs) are drawn as triangles on the top edge.
  const base = state.data.summary.baseline_val_loss ?? vals[0];
  const lo = Math.min(...vals);
  const cap = base + Math.max((base - lo) * 0.6, base * 0.04);
  const hiData = Math.max(base, ...vals.filter((v) => v <= cap));
  const pad = (hiData - lo) * 0.08 || base * 0.02;
  const y0 = lo - pad, y1 = hiData + pad;

  const m = { l: 4.2 * rem, r: 1.4 * rem, t: 1.0 * rem, b: 2.3 * rem };
  const n = items.length;
  const xmax = Math.max(n - 1, 1);
  const X = (i) => m.l + (i / xmax) * (W - m.l - m.r) * 0.97 + (W - m.l - m.r) * 0.015;
  const Y = (v) => m.t + (1 - (v - y0) / (y1 - y0)) * (H - m.t - m.b);
  const fs = (k) => (k * rem).toFixed(1);
  const parts = [];
  const xy = new Map();

  // axes
  const yt = niceTicks(y0, y1, Math.max(3, Math.min(6, Math.floor((H - m.t - m.b) / (2.6 * rem)))));
  const dec = Math.max(2, Math.min(5, -Math.floor(Math.log10(yt.step)) + (yt.step / Math.pow(10, Math.floor(Math.log10(yt.step))) === 2.5 ? 1 : 0)));
  for (const t of yt.ticks) {
    if (t < y0 || t > y1) continue;
    parts.push(`<line class="ax-grid" x1="${m.l}" x2="${W - m.r}" y1="${Y(t)}" y2="${Y(t)}"/>`,
      `<text class="ax-label" x="${m.l - 0.6 * rem}" y="${Y(t)}" dy="0.35em" text-anchor="end" font-size="${fs(0.72)}">${t.toFixed(dec)}</text>`);
  }
  parts.push(`<line class="ax-base" x1="${m.l}" x2="${W - m.r}" y1="${H - m.b}" y2="${H - m.b}"/>`);
  const xt = niceTicks(0, xmax, Math.max(2, Math.min(12, Math.floor((W - m.l - m.r) / (3.5 * rem)))));
  for (const t of xt.ticks) {
    if (t !== Math.round(t) || t > xmax) continue;
    parts.push(`<text class="ax-label" x="${X(t)}" y="${H - m.b + 1.15 * rem}" text-anchor="middle" font-size="${fs(0.72)}">${t}</text>`);
  }
  parts.push(`<text class="ax-title" x="${W - m.r}" y="${H - 0.15 * rem}" text-anchor="end" font-size="${fs(0.62)}">experiment #</text>`);

  // running best step line (kept results only)
  let d = "", prevY = null;
  items.forEach((it, i) => {
    const b = it.pending ? null : it.best_after;
    if (b == null) return;
    const x = X(i), y = Y(b);
    if (prevY == null) d += `M${x},${y}`;
    else if (y !== prevY) d += `H${x}V${y}`;
    prevY = y;
  });
  if (d) parts.push(`<path class="m-best" d="${d}H${X(n - 1)}"/>`);

  // selected guide
  const selIdx = items.findIndex((it) => it.key === state.sel);
  if (selIdx >= 0) parts.push(`<line class="m-guide" x1="${X(selIdx)}" x2="${X(selIdx)}" y1="${m.t}" y2="${H - m.b}"/>`);

  // points: discards first so kept points sit on top
  const order = items.map((it, i) => [it, i]).sort((a, b) => (a[0].status === "keep") - (b[0].status === "keep"));
  const r = Math.max(4, 0.3 * rem);
  for (const [it, i] of order) {
    const x = X(i), fresh = state.fresh.has(it.key) ? " fresh-mark" : "";
    let mark, cy;
    if (it.status === "crash" || it.val_loss == null && !it.pending) {
      cy = m.t + 0.5 * rem;
      const k = 0.32 * rem;
      mark = `<path class="m-crash${fresh}" d="M${x - k},${cy - k}L${x + k},${cy + k}M${x + k},${cy - k}L${x - k},${cy + k}"/>`;
    } else if (it.val_loss == null) {
      cy = Y(state.data.summary.best_val_loss ?? base);
      mark = `<circle class="m-pending" cx="${x}" cy="${cy}" r="${r * 1.1}"/>`;
    } else if (it.val_loss > y1) {
      cy = m.t + 0.45 * rem;
      const k = 0.36 * rem;
      mark = `<path class="m-off${fresh}" d="M${x},${cy - k}L${x + k},${cy + k * 0.8}L${x - k},${cy + k * 0.8}Z"/>`;
    } else {
      cy = Y(it.val_loss);
      const cls = it.pending ? "m-pending" : it.status === "keep" ? "m-keep" : "m-discard";
      const rr = it.pending ? r * 1.1 : it.status === "keep" ? r * 1.3 : r * 0.85;
      mark = `<circle class="${cls}${fresh}" cx="${x}" cy="${cy}" r="${rr}"/>`;
    }
    if (it.key === state.sel) mark += `<circle class="m-sel" cx="${x}" cy="${cy}" r="${r * 2}"/>`;
    parts.push(mark);
    xy.set(it.key, [x, cy]);
  }

  // direct labels on kept results, skipping ones that would collide
  const fsLabel = 0.7 * rem;
  let lastX = -Infinity;
  items.forEach((it, i) => {
    const isSel = it.key === state.sel;
    if (!xy.has(it.key) || !(it.status === "keep" || isSel || it.pending) || it.val_loss == null) return;
    const [x, y] = xy.get(it.key);
    if (!isSel && x - lastX < 2.4 * rem) return;
    lastX = x;
    let text = it.isBaseline ? "baseline" : it.description || "";
    const charW = fsLabel * 0.56, sin = 0.5, cos = 0.866;
    // Angled up-right like progress.png; fall back to down-right near the top, or flat to the left near the right edge.
    const roomUp = Math.min((y - m.t) / sin, (W - m.r - x) / cos);
    const roomDown = Math.min((H - m.b - 1.2 * rem - y) / sin, (W - m.r - x) / cos);
    const fit = (room) => Math.min(34, Math.floor((room - 0.8 * rem) / charW));
    let attrs;
    if (fit(roomUp) >= Math.min(12, text.length)) {
      text = clip(text, fit(roomUp));
      attrs = `text-anchor="start" transform="translate(${x + 0.5 * rem},${y - 0.45 * rem}) rotate(-30)"`;
    } else if (fit(roomDown) >= Math.min(12, text.length)) {
      text = clip(text, fit(roomDown));
      attrs = `text-anchor="start" transform="translate(${x + 0.55 * rem},${y + 0.85 * rem}) rotate(30)"`;
    } else {
      text = clip(text, Math.min(22, Math.floor((x - m.l) / charW / 2)));
      attrs = `text-anchor="end" x="${x - 0.8 * rem}" y="${y - 0.75 * rem}"`;
    }
    parts.push(`<text class="pt-label${isSel ? " sel" : ""}" font-size="${fsLabel.toFixed(1)}" ${attrs}>${esc(text)}</text>`);
  });

  // hit targets, on top
  for (const [key, [x, y]] of xy) parts.push(`<circle class="hit" data-key="${esc(key)}" cx="${x}" cy="${y}" r="${Math.max(12, 0.75 * rem)}"/>`);

  box.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="val_loss for each experiment, with the running best">${parts.join("")}</svg>`;
}

// ------------------------------------------------------------------ render: list

function renderList() {
  const s = state.data.summary;
  const rows = state.items.slice().reverse().map((it) => {
    const cls = ["row", "is-" + it.status, state.fresh.has(it.key) ? "fresh" : ""].join(" ");
    const dBest = it.isBaseline ? null : it.delta_best;
    return `<button type="button" class="${cls}" data-key="${esc(it.key)}" aria-current="${it.key === state.sel}">
      <span class="c-n">${it.n}</span>
      <span class="c-status">${pill(it)}</span>
      <span class="c-desc" title="${esc(it.description)}">${it.isBest ? '<span class="best-tag">BEST</span>' : ""}${esc(it.description || "—")}</span>
      <span class="c-num">${fmtLoss(it.val_loss)}</span>
      <span class="c-num ${it.isBaseline ? "" : tone(it.delta_baseline)}">${it.isBaseline ? "baseline" : fmtPct(it.delta_baseline, s.baseline_val_loss)}</span>
      <span class="c-num ${tone(dBest)}">${fmtPct(dBest, it.best_before)}</span>
    </button>`;
  });
  $("#list").innerHTML = rows.join("") ||
    `<div class="empty-state"><div><h3>No experiments logged</h3><p>Rows appear here as the agent appends to results.tsv.</p></div></div>`;
  if (state.follow) $("#list").scrollTop = 0;
}

// ------------------------------------------------------------------ render: one run

function imgOrEmpty(it, file, alt, wide) {
  const url = it.files && it.files[file];
  if (url) return `<img src="${esc(url)}" alt="${esc(alt)}" loading="eager" decoding="async">`;
  const why = it.status === "crash" ? "Crashed before sampling" : it.pending ? "Sampling…" : "No image saved";
  return `<div class="empty-img${wide ? " wide" : ""}">${why}</div>`;
}

function samplesFig(it) {
  const m = it.metrics || {};
  return `<figure class="samples">${imgOrEmpty(it, "samples.png", "16 benchmark samples")}
    <figcaption><b>“${esc(m.prompt || state.data.prompt || "benchmark prompt")}”</b> · seeds 0–15${
      m.sample_seconds != null ? " · sampled in " + m.sample_seconds + " s" : ""}</figcaption></figure>`;
}

function trajFig(it) {
  return `<figure class="traj">${imgOrEmpty(it, "trajectory.png", "seed 0 denoised from noise to image", true)}
    <div class="traj-scale"><span>t = 1 · pure noise</span><span>seed 0 denoising →</span><span>t = 0 · sample</span></div></figure>`;
}

function stat(label, value, cls = "") {
  return `<div class="kpi d-stat"><span class="kpi-label">${label}</span><span class="kpi-value ${cls}">${value}</span></div>`;
}

function renderDetail() {
  const el = $("#detail");
  const it = state.byKey.get(state.sel);
  if (!it) {
    el.innerHTML = `<div class="empty-state"><div><h3>Nothing selected</h3><p>Pick an experiment from the chart or the list.</p></div></div>`;
    state.detailSig = "";
    return;
  }
  const sig = JSON.stringify([it, state.data.summary]);
  if (sig === state.detailSig) return;
  const sameRun = el.dataset.key === it.key;
  state.detailSig = sig;
  const s = state.data.summary, m = it.metrics || {};
  const idea = it.idea && typeof it.idea === "object" ? it.idea : null;
  const sources = idea && Array.isArray(idea.sources) ? idea.sources.map(safeUrl).filter(Boolean) : [];
  const facts = [
    ["Steps", m.num_steps != null ? m.num_steps.toLocaleString() : null],
    ["Params", m.num_params_M != null ? m.num_params_M + " M" : null],
    ["Training", m.training_seconds != null ? m.training_seconds + " s" : null],
    ["Peak memory", (m.peak_memory_gb ?? it.memory_gb) != null ? (m.peak_memory_gb ?? it.memory_gb) + " GB" : null],
    ["Commit", it.commit || m.commit],
    ["Run", it.run_id],
  ].filter(([, v]) => v != null && v !== "");

  el.dataset.key = it.key;
  el.innerHTML = `
    <div class="d-head">
      <div class="d-title"><span class="d-n">#${it.n}</span>${pill(it)}${it.isBest ? '<span class="best-tag">BEST</span>' : ""}</div>
      <h2 class="d-desc">${esc(it.description || "—")}</h2>
      <div class="d-stats">
        ${stat(it.pending ? "val_loss (unlogged)" : "val_loss", fmtLoss(it.val_loss))}
        ${it.isBaseline ? "" : stat("vs baseline", fmtPct(it.delta_baseline, s.baseline_val_loss), tone(it.delta_baseline))}
        ${it.isBaseline ? "" : stat("vs best before it", fmtPct(it.delta_best, it.best_before), tone(it.delta_best))}
      </div>
    </div>
    <div class="d-media">
      ${samplesFig(it)}
      <div class="side">
        ${idea ? `<div class="idea"><span class="eyebrow">Planner's idea</span>
          ${idea.title ? `<h3>${esc(idea.title)}</h3>` : ""}
          ${idea.rationale ? `<p>${esc(idea.rationale)}</p>` : ""}
          ${predictionHtml(idea)}
          ${Array.isArray(idea.agents) && idea.agents.length ? `<p class="agents-line">Spawned ${idea.agents.length} sub-agent${idea.agents.length > 1 ? "s" : ""}: ${idea.agents.map((g) => esc(g.role)).join(" · ")}</p>` : ""}
          ${sources.length ? `<ul>${sources.map((u) => `<li><a href="${esc(u)}" target="_blank" rel="noopener noreferrer">${esc(u.replace(/^https?:\/\//, ""))}</a></li>`).join("")}</ul>` : ""}
        </div>` : ""}
        ${facts.length ? `<dl class="facts">${facts.map(([k, v]) => `<dt>${k}</dt><dd>${esc(v)}</dd>`).join("")}</dl>` : ""}
      </div>
    </div>
    ${trajFig(it)}
    <div class="diff-box">
      <div class="diff-head"><h3 class="eyebrow">What changed in train.py</h3><span class="counts-diff" id="diff-counts"></span>
        ${it.files["train.py"] ? `<a class="link-btn" href="${esc(it.files["train.py"])}" target="_blank" rel="noopener">Full train.py ↗</a>` : ""}</div>
      <div id="diff-slot"></div>
    </div>`;
  if (!sameRun) el.scrollTop = 0;
  loadDiff(it);
}

const signedPct = (v) => (v > 0 ? "+" : v < 0 ? "−" : "±") + Math.abs(v).toFixed(2) + "%";

function predictionHtml(idea) {
  const pred = idea.prediction || {}, out = idea.outcome || {};
  if (typeof pred.delta_pct !== "number") return "";
  const actual = typeof out.delta_pct === "number"
    ? `<span>Actual <b class="${tone(out.delta_pct)}">${signedPct(out.delta_pct)}</b></span>` : "<span>Actual <b>—</b></span>";
  return `<div class="pred"><span>Predicted <b>${signedPct(pred.delta_pct)}</b>${pred.confidence ? ` · ${esc(pred.confidence)} confidence` : ""}</span>${actual}</div>
    ${out.prediction_check ? `<p class="pred-check">${esc(out.prediction_check)}</p>` : ""}`;
}

async function loadDiff(it) {
  const slot = $("#diff-slot");
  const url = it.files["change.diff"];
  if (!url) {
    slot.innerHTML = `<p class="note">${it.isBaseline ? "The baseline has nothing to compare against." :
      it.status === "crash" ? "This run crashed before it saved its files." : "No change.diff saved for this run."}</p>`;
    return;
  }
  let text = state.diffs.get(url);
  if (text == null) {
    slot.innerHTML = `<p class="note">Loading diff…</p>`;
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error("HTTP " + res.status);
      text = await res.text();
      state.diffs.set(url, text);
    } catch (err) {
      if ($("#detail").dataset.key === it.key) slot.innerHTML = `<p class="note">Couldn't load change.diff (${esc(err.message)}).</p>`;
      return;
    }
  }
  if ($("#detail").dataset.key !== it.key) return; // selection moved on while loading
  const { html, add, del } = diffHtml(text);
  slot.innerHTML = `<div class="diff">${html || '<div class="note" style="padding:0.5rem 1rem">Empty diff.</div>'}</div>`;
  $("#diff-counts").innerHTML = `<span class="add-n">+${add}</span> <span class="del-n">−${del}</span>`;
}

function diffHtml(text) {
  const out = [];
  let o = 0, n = 0, add = 0, del = 0;
  const line = (cls, a, b, code) => `<div class="dl ${cls}"><span class="ln">${a}</span><span class="ln">${b}</span><span class="code">${esc(code) || " "}</span></div>`;
  const lines = text.replace(/\n$/, "").split("\n");
  for (const l of lines) {
    if (/^(diff --git|index |--- |\+\+\+ |new file|deleted file|similarity|rename )/.test(l)) continue;
    if (l.startsWith("@@")) {
      const mm = /^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@(.*)$/.exec(l);
      if (mm) { o = +mm[1]; n = +mm[2]; }
      out.push(line("hunk", "", "", mm ? `@@ line ${mm[2]}${mm[3] ? " ·" + mm[3] : ""}` : l));
    } else if (l.startsWith("+")) { out.push(line("add", "", n++, "+ " + l.slice(1))); add++; }
    else if (l.startsWith("-")) { out.push(line("del", o++, "", "− " + l.slice(1))); del++; }
    else if (l.startsWith("\\")) continue;
    else out.push(line("ctx", o++, n++, "  " + l.slice(1)));
  }
  return { html: out.join(""), add, del };
}

// ------------------------------------------------------------------ render: compare

function renderCompare() {
  const el = $("#view-compare");
  if (state.view !== "compare" || !state.data) return;
  const A = state.byKey.get(state.cmpA), B = state.byKey.get(state.cmpB);
  const sig = JSON.stringify([A, B, state.cmpPinned, state.items.map((it) => it.key + it.status)]);
  if (sig === state.compareSig) return;
  state.compareSig = sig;
  const choices = state.items.filter((it) => it.files && it.files["samples.png"]);
  if (!choices.length) {
    el.innerHTML = `<div class="panel empty-state"><div><h3>Nothing to compare yet</h3><p>Runs appear here once they have sample images.</p></div></div>`;
    return;
  }
  const options = (selKey) => choices.slice().reverse().map((it) =>
    `<option value="${esc(it.key)}"${it.key === selKey ? " selected" : ""}>#${it.n} ${it.isBaseline ? "baseline" : esc(statusLabel[it.status] || it.status)} · ${fmtLoss(it.val_loss)} · ${esc((it.description || "").slice(0, 60))}</option>`).join("");
  const card = (side, it, other) => {
    if (!it) return `<article class="panel cmp-card"></article>`;
    const d = other && it.val_loss != null && other.val_loss != null ? it.val_loss - other.val_loss : null;
    return `<article class="panel cmp-card">
      <div class="cmp-head">
        <div class="cmp-pick"><label class="eyebrow" for="cmp-${side}">${side === "a" ? "Left" : "Right"}</label>
          <select id="cmp-${side}" data-side="${side}">${options(it.key)}</select></div>
        <div class="d-title"><span class="d-n">#${it.n}</span>${pill(it)}${it.isBest ? '<span class="best-tag">BEST</span>' : ""}</div>
        <h2 class="cmp-desc">${esc(it.isBaseline ? "Baseline" : it.description || "—")}</h2>
        <div class="cmp-stats">${stat("val_loss", fmtLoss(it.val_loss))}${
          side === "b" && other ? stat(`vs #${other.n}`, fmtPct(d, other.val_loss), tone(d)) : ""}${
          it.metrics && it.metrics.num_steps != null ? stat("steps", it.metrics.num_steps.toLocaleString()) : ""}</div>
      </div>
      ${samplesFig(it)}
      ${trajFig(it)}
    </article>`;
  };
  el.innerHTML = `
    <div class="cmp-bar">
      <button type="button" class="btn" id="cmp-reset" aria-pressed="${!state.cmpPinned}">Baseline vs best</button>
      <button type="button" class="btn" id="cmp-swap">Swap sides</button>
    </div>
    <div class="cmp-grid">${card("a", A, null)}${card("b", B, A)}</div>`;
}

// ------------------------------------------------------------------ interaction

function select(key, byUser = true) {
  if (!state.byKey.has(key)) return;
  state.sel = key;
  if (byUser) state.follow = key === newestKey();
  renderChart();
  renderList();
  renderDetail();
  const row = $(`#list .row[data-key="${CSS.escape(key)}"]`);
  if (row) row.scrollIntoView({ block: "nearest" });
}

function renderPlaybook() {
  const el = $("#view-playbook");
  if (state.view !== "playbook" || !state.data) return;
  const h = state.data.harness || {};
  const lessons = h.lessons || [];
  if (h.design === "agentic" || (!lessons.length && h.notes)) {
    el.innerHTML = `<section class="panel playbook">
      <div class="panel-head"><h2>Orchestrator notes <span class="sub">the model-driven design keeps its own memory; it rewrites these notes when it chooses to</span></h2></div>
      ${h.notes ? `<pre class="notes">${esc(h.notes)}</pre>` : `<div class="empty-state"><div><h3>No notes yet</h3><p>The orchestrator writes notes here when it decides they are worth keeping.</p></div></div>`}
    </section>`;
    return;
  }
  el.innerHTML = `<section class="panel playbook">
    <div class="panel-head"><h2>Playbook <span class="sub">lessons the loop has learned, rewritten by the reflector after every experiment</span></h2></div>
    ${lessons.length ? `<ol class="lessons">${lessons.map((l) => `<li><span class="l-id">${esc(l.id)}</span><span class="l-text">${esc(l.text)}</span>
      <span class="l-meta">added after #${esc(l.since)}${l.updated !== l.since ? ` · revised after #${esc(l.updated)}` : ""}</span></li>`).join("")}</ol>`
      : `<div class="empty-state"><div><h3>No lessons yet</h3><p>The reflector adds lessons here after each experiment, and the planner reads them before choosing the next one.</p></div></div>`}
  </section>`;
}

const VIEWS = ["progress", "compare", "playbook"];

function setView(view) {
  state.view = view;
  for (const v of VIEWS) {
    $("#view-" + v).hidden = view !== v;
    $("#tab-" + v).setAttribute("aria-pressed", view === v);
  }
  try { history.replaceState(null, "", "#" + view); } catch (e) {}
  state.compareSig = "";
  if (state.data) {
    if (view === "progress") renderChart();
    renderCompare();
    renderPlaybook();
  }
}

function toggleTheme() {
  const root = document.documentElement;
  const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  root.dataset.theme = dark ? "light" : "dark";
  try { localStorage.setItem("ar-theme", root.dataset.theme); } catch (e) {}
}

function showTip(target) {
  const it = state.byKey.get(target.dataset.key);
  if (!it) return;
  const tip = $("#tip");
  const s = state.data.summary;
  tip.innerHTML = `<div><b>#${it.n}</b> · ${esc(statusLabel[it.status] || it.status)}${it.isBest ? " · best" : ""}</div>
    <div>${esc(it.isBaseline ? "baseline" : it.description)}</div>
    <div class="t-sub">val_loss ${fmtLoss(it.val_loss)}${it.isBaseline || it.val_loss == null ? "" : " · " + fmtPct(it.delta_baseline, s.baseline_val_loss) + " vs baseline"}</div>`;
  tip.hidden = false;
  const r = target.getBoundingClientRect();
  const tw = tip.offsetWidth, th = tip.offsetHeight;
  let x = r.left + r.width / 2 - tw / 2, y = r.top - th - 6;
  if (y < 4) y = r.bottom + 6;
  x = Math.max(4, Math.min(innerWidth - tw - 4, x));
  tip.style.left = x + "px";
  tip.style.top = y + "px";
}

function bind() {
  $("#chart").addEventListener("click", (e) => { const t = e.target.closest("[data-key]"); if (t) select(t.dataset.key); });
  $("#chart").addEventListener("mouseover", (e) => { const t = e.target.closest("[data-key]"); if (t) showTip(t); });
  $("#chart").addEventListener("mouseout", (e) => { if (e.target.closest("[data-key]")) $("#tip").hidden = true; });
  $("#list").addEventListener("click", (e) => { const t = e.target.closest("[data-key]"); if (t) select(t.dataset.key); });
  document.querySelectorAll(".tabs [data-view]").forEach((b) => b.addEventListener("click", () => setView(b.dataset.view)));
  $("#theme").addEventListener("click", toggleTheme);

  $("#view-compare").addEventListener("change", (e) => {
    const sel = e.target.closest("select[data-side]");
    if (!sel) return;
    if (sel.dataset.side === "a") state.cmpA = sel.value; else state.cmpB = sel.value;
    state.cmpPinned = true;
    renderCompare();
  });
  $("#view-compare").addEventListener("click", (e) => {
    if (e.target.closest("#cmp-reset")) { setDefaultCompare(); renderCompare(); }
    if (e.target.closest("#cmp-swap")) { [state.cmpA, state.cmpB] = [state.cmpB, state.cmpA]; state.cmpPinned = true; renderCompare(); }
  });

  document.addEventListener("keydown", (e) => {
    if (e.metaKey || e.ctrlKey || e.altKey || /^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName)) return;
    const k = e.key.toLowerCase();
    const i = state.items.findIndex((it) => it.key === state.sel);
    if ((k === "arrowup" || k === "arrowright" || k === "k") && state.view === "progress") {
      if (i < state.items.length - 1) select(state.items[i + 1].key); e.preventDefault();
    } else if ((k === "arrowdown" || k === "arrowleft" || k === "j") && state.view === "progress") {
      if (i > 0) select(state.items[i - 1].key); e.preventDefault();
    } else if (k === "b" && state.data) {
      const best = state.items.find((it) => it.isBest); if (best) { setView("progress"); select(best.key); }
    } else if (k === "l" && state.data) {
      state.follow = true; setView("progress"); select(newestKey(), false);
    } else if (k === "c") setView(state.view === "compare" ? "progress" : "compare");
    else if (k === "p" || k === "1") setView("progress");
    else if (k === "2") setView("compare");
    else if (k === "3" || k === "y") setView("playbook");
    else if (k === "t") toggleTheme();
  });

  new ResizeObserver(() => { if (state.data && state.view === "progress") renderChart(); }).observe($("#chart"));
}

// The 4x4 glyph in the header: settles into a solid grid when idle, shimmers like noise while a run is active.
function glyph() {
  const g = $("#glyph");
  g.innerHTML = "<i></i>".repeat(16);
  const cells = [...g.children];
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const tick = () => {
    cells.forEach((c, i) => {
      c.style.opacity = state.liveActive && !reduced ? (0.15 + Math.random() * 0.85).toFixed(2) : (i % 5 === 0 ? 1 : 0.82);
    });
  };
  tick();
  setInterval(tick, 450);
}

// ------------------------------------------------------------------ boot

bind();
glyph();
setView(VIEWS.includes(location.hash.slice(1)) ? location.hash.slice(1) : "progress");
poll();
setInterval(poll, POLL_MS);
setInterval(() => { if (state.data && !state.error) renderLive(); }, 1000);
