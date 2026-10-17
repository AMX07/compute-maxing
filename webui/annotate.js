// Annotate mode, for design review. Off unless the URL has ?annotate (e.g. /?annotate#progress).
// A toggles picking. While picking: hover shows a component's name, ↑/↓ widen or narrow to the parent/child,
// click pins a numbered note to it. Notes are saved to webui/annotations.json through /api/annotations.
(() => {
  if (!new URLSearchParams(location.search).has("annotate")) return;

  const css = `
  #ann-root { --ann: #e5197b; font: 13px/1.4 -apple-system, BlinkMacSystemFont, system-ui, sans-serif; color: #111; }
  #ann-root * { box-sizing: border-box; }
  .ann-box { position: fixed; pointer-events: none; z-index: 99990; border: 2px solid var(--ann);
    background: rgba(229, 25, 123, 0.07); border-radius: 3px; }
  .ann-box.ann-ghost { border-style: dashed; background: none; }
  .ann-label { position: fixed; pointer-events: none; z-index: 99991; background: var(--ann); color: #fff;
    font: 600 11px/1 ui-monospace, Menlo, monospace; padding: 4px 6px; border-radius: 3px; max-width: 60vw;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .ann-pin { position: fixed; z-index: 99992; width: 22px; height: 22px; margin: -11px 0 0 -11px; border-radius: 50%;
    background: var(--ann); color: #fff; font: 700 11px/22px system-ui, sans-serif; text-align: center; cursor: pointer;
    box-shadow: 0 0 0 2px #fff, 0 2px 6px rgba(0,0,0,.3); }
  .ann-pin.ann-lost { opacity: 0.45; }
  .ann-bar { position: fixed; right: 12px; bottom: 12px; z-index: 99995; width: 320px; max-height: 60vh; display: flex;
    flex-direction: column; background: #fff; border-radius: 10px; box-shadow: 0 6px 24px rgba(0,0,0,.25), 0 0 0 1px rgba(0,0,0,.08); }
  .ann-bar header { display: flex; align-items: center; gap: 6px; padding: 8px 10px; border-bottom: 1px solid #eee; }
  .ann-bar header b { flex: 1; }
  .ann-bar button, .ann-ed button { font: inherit; border: 1px solid #ddd; background: #f6f6f6; border-radius: 6px;
    padding: 3px 8px; cursor: pointer; color: #111; }
  .ann-bar button.ann-on, .ann-ed button.ann-primary { background: var(--ann); border-color: var(--ann); color: #fff; }
  .ann-bar ol { margin: 0; padding: 4px 0; overflow: auto; list-style: none; }
  .ann-bar li { display: grid; grid-template-columns: 22px 1fr; gap: 8px; padding: 6px 10px; cursor: pointer; }
  .ann-bar li:hover { background: #fdf0f6; }
  .ann-bar li i { font-style: normal; font-weight: 700; color: var(--ann); }
  .ann-bar li code { display: block; font: 11px ui-monospace, Menlo, monospace; color: #777; overflow: hidden;
    white-space: nowrap; text-overflow: ellipsis; }
  .ann-bar .ann-hint { padding: 8px 10px; color: #777; }
  .ann-ed { position: fixed; z-index: 99996; width: 300px; background: #fff; border-radius: 10px; padding: 8px;
    box-shadow: 0 6px 24px rgba(0,0,0,.3), 0 0 0 1px rgba(0,0,0,.08); }
  .ann-ed code { display: block; font: 11px ui-monospace, Menlo, monospace; color: #777; margin-bottom: 6px;
    overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
  .ann-ed textarea { width: 100%; min-height: 72px; font: inherit; border: 1px solid #ddd; border-radius: 6px; padding: 6px; resize: vertical; }
  .ann-ed div { display: flex; gap: 6px; margin-top: 6px; justify-content: flex-end; }
  .ann-ed div span { flex: 1; color: #999; font-size: 11px; align-self: center; }`;

  const root = document.createElement("div");
  root.id = "ann-root";
  root.innerHTML = `<style>${css}</style>
    <div class="ann-box" hidden></div><div class="ann-label" hidden></div><div class="ann-pins"></div>
    <section class="ann-bar"><header><b>Annotations</b>
      <button type="button" data-act="pick" title="Toggle picking (A)">Pick</button>
      <button type="button" data-act="copy" title="Copy all notes as Markdown">Copy</button>
      <button type="button" data-act="clear" title="Delete all notes">Clear</button></header><ol></ol></section>`;
  document.body.appendChild(root);
  const q = (s) => root.querySelector(s);
  const box = q(".ann-box"), label = q(".ann-label");

  let notes = [];
  let picking = true;
  let hover = null, stack = [];  // hover = element under the outline; stack = narrower elements left by ↑
  let editor = null;

  const view = () => location.hash.slice(1) || "progress";
  const ours = (el) => el && el.closest && el.closest("#ann-root");

  function selectorFor(el) {
    const parts = [];
    for (let e = el; e && e.nodeType === 1 && e !== document.body; e = e.parentElement) {
      if (e.id) { parts.unshift("#" + CSS.escape(e.id)); break; }
      let s = e.tagName.toLowerCase();
      if (e.dataset && e.dataset.key) s += `[data-key="${CSS.escape(e.dataset.key)}"]`;
      else if (e.parentElement) {
        const same = [...e.parentElement.children].filter((c) => c.tagName === e.tagName);
        if (same.length > 1) s += `:nth-of-type(${same.indexOf(e) + 1})`;
      }
      parts.unshift(s);
    }
    return parts.join(" > ") || "body";
  }

  function nameOf(el) {
    const cls = (el.getAttribute("class") || "").trim().split(/\s+/).filter(Boolean).slice(0, 3);
    let s = el.tagName.toLowerCase() + (el.id ? "#" + el.id : "") + cls.map((c) => "." + c).join("");
    const head = el.getAttribute("aria-label") || el.querySelector?.("h1,h2,h3")?.textContent;
    if (head) s += ` “${head.trim().replace(/\s+/g, " ").slice(0, 40)}”`;
    return s;
  }

  function place(el, r) {
    el.hidden = false;
    Object.assign(el.style, { left: r.left + "px", top: r.top + "px", width: r.width + "px", height: r.height + "px" });
  }

  function showHover() {
    if (!hover || !picking || editor) { box.hidden = label.hidden = true; return; }
    const r = hover.getBoundingClientRect();
    box.classList.remove("ann-ghost");
    place(box, r);
    label.hidden = false;
    label.textContent = nameOf(hover) + `  ${Math.round(r.width)}×${Math.round(r.height)}`;
    label.style.left = Math.max(4, r.left) + "px";
    label.style.top = (r.top > 22 ? r.top - 20 : r.bottom + 2) + "px";
  }

  function find(n) {
    try { return document.querySelector(n.selector); } catch (e) { return null; }
  }

  function renderPins() {
    // Reuses the pin elements, so a pin's tooltip survives the timer that keeps re-placing them.
    const pins = q(".ann-pins");
    const shown = notes.filter((n) => n.view === view());
    while (pins.children.length > shown.length) pins.lastChild.remove();
    while (pins.children.length < shown.length) pins.appendChild(document.createElement("div"));
    shown.forEach((n, i) => {
      const el = find(n);
      let x, y;
      if (el && el.getClientRects().length) { const r = el.getBoundingClientRect(); x = r.left; y = r.top; }
      else { x = n.rect.x - scrollX; y = n.rect.y - scrollY; }
      const p = pins.children[i];
      p.className = "ann-pin" + (el ? "" : " ann-lost");
      p.textContent = n.n;
      p.title = n.note;
      p.dataset.n = n.n;
      p.style.left = Math.max(12, x) + "px";
      p.style.top = Math.max(12, y) + "px";
    });
  }

  function renderList() {
    q('[data-act="pick"]').classList.toggle("ann-on", picking);
    q(".ann-bar ol").innerHTML = notes.length
      ? notes.map((n) => `<li data-n="${n.n}"><i>${n.n}</i><span>${esc(n.note)}<code>${esc(n.view)} · ${esc(n.name)}</code></span></li>`).join("")
      : `<li class="ann-hint" style="display:block">Click a component to pin a note. ↑/↓ widen or narrow the outline. A toggles picking.</li>`;
    renderPins();
  }

  function esc(s) {
    return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  }

  async function load() {
    try { notes = await (await fetch("/api/annotations", { cache: "no-store" })).json(); } catch (e) { notes = []; }
    if (!Array.isArray(notes)) notes = [];
    renderList();
  }

  async function save() {
    renderList();
    try {
      const r = await fetch("/api/annotations", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(notes, null, 2) });
      if (!r.ok) throw new Error(r.status);
    } catch (e) {
      q(".ann-bar b").textContent = "Annotations (not saved!)";
    }
  }

  function openEditor(target, existing) {
    closeEditor();
    const el = existing ? find(existing) : target;
    const r = el ? el.getBoundingClientRect() : { left: 40, top: 40, bottom: 40 };
    label.hidden = true;
    if (el) { box.classList.add("ann-ghost"); place(box, r); }
    editor = document.createElement("div");
    editor.className = "ann-ed";
    editor.innerHTML = `<code>${esc(existing ? existing.name : nameOf(target))}</code><textarea placeholder="What should change here?"></textarea>
      <div><span title="Shift+Enter for a new line">Enter saves</span>${existing ? '<button type="button" data-act="del">Delete</button>' : ""}
      <button type="button" data-act="cancel">Cancel</button><button type="button" class="ann-primary" data-act="ok">Save</button></div>`;
    root.appendChild(editor);
    const ta = editor.querySelector("textarea");
    ta.value = existing ? existing.note : "";
    const x = Math.min(innerWidth - 310, Math.max(10, r.left));
    const y = r.bottom + 8 + 170 < innerHeight ? r.bottom + 8 : Math.max(10, r.top - 178);
    editor.style.left = x + "px";
    editor.style.top = y + "px";
    ta.focus();

    const commit = () => {
      const text = ta.value.trim();
      if (existing) {
        if (text) existing.note = text;
      } else if (text) {
        const rr = target.getBoundingClientRect();
        notes.push({
          n: notes.reduce((m, o) => Math.max(m, o.n), 0) + 1, view: view(), note: text,
          name: nameOf(target), selector: selectorFor(target),
          text: (target.innerText || target.textContent || "").trim().replace(/\s+/g, " ").slice(0, 160),
          rect: { x: Math.round(rr.left + scrollX), y: Math.round(rr.top + scrollY), w: Math.round(rr.width), h: Math.round(rr.height) },
          viewport: { w: innerWidth, h: innerHeight }, created_at: new Date().toISOString(),
        });
      }
      closeEditor();
      save();
    };
    ta.addEventListener("keydown", (e) => {
      e.stopPropagation();
      if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); commit(); }
      else if (e.key === "Escape") closeEditor();
    });
    editor.addEventListener("click", (e) => {
      const act = e.target.closest("[data-act]")?.dataset.act;
      if (act === "ok") commit();
      else if (act === "cancel") closeEditor();
      else if (act === "del") { notes = notes.filter((o) => o !== existing); closeEditor(); save(); }
    });
  }

  function closeEditor() {
    if (editor) editor.remove();
    editor = null;
    box.classList.remove("ann-ghost");
    showHover();
  }

  function setPicking(on) {
    picking = on;
    if (!on) hover = null;
    showHover();
    renderList();
  }

  function markdown() {
    return notes.map((n) => `${n.n}. [${n.view}] **${n.name}** (\`${n.selector}\`): ${n.note}`).join("\n");
  }

  // Capture phase on window, so the app's own handlers never see clicks or keys meant for annotating.
  const swallow = (e) => { if (picking && !ours(e.target)) { e.preventDefault(); e.stopImmediatePropagation(); } };
  for (const type of ["mousedown", "mouseup", "pointerdown", "pointerup", "dblclick", "mouseover", "mouseout"]) {
    addEventListener(type, swallow, true);
  }
  addEventListener("click", (e) => {
    if (!picking || ours(e.target)) return;
    e.preventDefault();
    e.stopImmediatePropagation();
    if (!editor && hover) openEditor(hover);
  }, true);
  addEventListener("mousemove", (e) => {
    if (!picking || editor) return;
    const t = ours(e.target) || e.target === document.body || e.target === document.documentElement ? null : e.target;
    if (t !== (stack.length ? stack[stack.length - 1] : hover)) { hover = t; stack = []; }
    showHover();
  }, true);
  addEventListener("keydown", (e) => {
    if (editor || e.metaKey || e.ctrlKey || e.altKey || /^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName)) return;
    const k = e.key.toLowerCase();
    if (k === "a") setPicking(!picking);
    else if (k === "escape" && picking) setPicking(false);
    else if (picking && hover && k === "arrowup") {
      const p = hover.parentElement;
      if (p && p !== document.documentElement && p !== document.body) { stack.unshift(hover); hover = p; showHover(); }
    } else if (picking && hover && k === "arrowdown") {
      if (stack.length) { hover = stack.shift(); showHover(); }
    } else return;
    e.preventDefault();
    e.stopImmediatePropagation();
  }, true);

  root.addEventListener("click", (e) => {
    const act = e.target.closest("[data-act]")?.dataset.act;
    const item = e.target.closest("[data-n]");
    if (act === "pick") setPicking(!picking);
    else if (act === "copy") navigator.clipboard.writeText(markdown()).then(() => {
      const b = q('[data-act="copy"]'); b.textContent = "Copied"; setTimeout(() => (b.textContent = "Copy"), 1200);
    });
    else if (act === "clear" && notes.length && confirm(`Delete all ${notes.length} notes?`)) { notes = []; save(); }
    else if (item && !editor) {
      const n = notes.find((o) => o.n === +item.dataset.n);
      if (!n) return;
      if (n.view !== view()) location.hash = n.view;
      find(n)?.scrollIntoView({ block: "nearest" });
      openEditor(null, n);
    }
  });

  // The app re-renders on every poll, so pins re-resolve their elements on a timer as well as on scroll and resize.
  addEventListener("scroll", () => { renderPins(); showHover(); }, true);
  addEventListener("resize", renderPins);
  addEventListener("hashchange", renderPins);
  setInterval(renderPins, 500);
  load();
})();
