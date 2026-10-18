"use strict";

const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { join } = require("node:path");
const { test } = require("node:test");
const vm = require("node:vm");

const escapeHTML = (value) => String(value ?? "").replace(/[&<>"']/g,
  (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char]));
const source = readFileSync(join(__dirname, "../webui/app.js"), "utf8").split("// BEGIN SHARED ARCHITECTURE DIAGRAM")[1].split("// END SHARED ARCHITECTURE DIAGRAM")[0];
const context = vm.createContext({
  esc: escapeHTML,
  humanize: (value) => String(value || "other").replace(/[_-]+/g, " ").replace(/^\w/, (c) => c.toUpperCase()),
  S: { route: { view: "baseline", sel: null } },
});
vm.runInContext(source, context);

function model() {
  const arch = JSON.parse(readFileSync(join(__dirname, "fixtures/cs336-architecture.json"), "utf8"));
  const comps = arch.components.map((c) => ({
    ...c, current: null, currentLabel: c.baseline.label, currentCost: c.baseline.cost,
    steps: [], next: [],
  }));
  const byId = new Map(comps.map((c) => [c.id, c]));
  return {
    arch, comps, byId,
    groups: arch.groups.map((g) => ({ ...g, comps: comps.filter((c) => c.group === g.id) })),
  };
}

function topology() {
  return {
    version: 1,
    basis: "Source defaults",
    description: "The recorded forward path in gpt.py.",
    flow: [
      { kind: "input", title: "Token IDs", note: "Batch × sequence" },
      { component: "embedding" },
      { kind: "repeat", component: "blocks", title: "Transformer stack", children: [
        { kind: "residual", title: "Attention residual", children: [
          { kind: "operation", title: "RMSNorm" }, { component: "attention" },
        ] },
        { kind: "residual", title: "Feed-forward residual", children: [
          { kind: "operation", title: "RMSNorm" }, { component: "ffn" },
        ] },
      ] },
      { component: "head" },
      { kind: "output", title: "Next-token logits" },
    ],
  };
}

function render(m, view = "baseline") {
  context.S.route.view = view;
  context.fixture = m;
  context.view = view;
  // A malformed project map must never be able to lock up the page.
  return vm.runInContext("renderArchitectureDiagram(fixture, view)", context, { timeout: 1000 });
}

function hasClass(html, name) {
  return new RegExp(`class="[^"]*\\b${name}\\b`).test(html);
}

function assertFallback(html) {
  assert.ok(hasClass(html, "diagram-fallback"), "unsafe or absent topology uses a component map");
  assert.equal(hasClass(html, "diagram-flow"), false, "component maps must not invent a forward path");
}

function attributes(tag) {
  const attrs = {};
  for (const match of tag.matchAll(/([\w:-]+)\s*=\s*(["'])(.*?)\2/g)) attrs[match[1]] = match[3];
  return attrs;
}

// Inspect semantic groups rather than snapshots or exact pixel dimensions. A button
// belongs to its closest marked schematic group, including inside foreignObject.
function schematicGroups(html) {
  const groups = [], stack = [];
  for (const match of html.matchAll(/<\/g\s*>|<g\b[^>]*>|<button\b[^>]*>/g)) {
    const tag = match[0];
    if (tag.startsWith("</")) { stack.pop(); continue; }
    if (tag.startsWith("<g")) {
      const group = { attrs: attributes(tag), buttons: [] };
      stack.push(group);
      if (group.attrs["data-schematic-kind"]) groups.push(group);
    } else {
      const owner = [...stack].reverse().find((group) => group.attrs["data-schematic-kind"]);
      if (owner) owner.buttons.push(attributes(tag));
    }
  }
  return groups;
}

function yOf(group) {
  assert.ok(group, "expected schematic node is present");
  const y = Number(group.attrs["data-node-y"]);
  assert.ok(Number.isFinite(y), "schematic node has a finite vertical position");
  return y;
}

test("baseline topology displays actual component values inside the model diagram", () => {
  const m = model();
  m.arch.diagram = topology();
  // Baseline must remain baseline even when a kept result is available.
  m.byId.get("attention").currentLabel = "8 heads · modified";
  const html = render(m);
  assert.ok(hasClass(html, "diagram-flow"));
  for (const id of ["embedding", "blocks", "attention", "ffn", "head"]) {
    assert.ok(html.includes(escapeHTML(m.byId.get(id).baseline.label)), `${id} value is visible`);
    assert.ok(html.includes(`data-comp="${id}"`), `${id} remains inspectable`);
    assert.ok(html.includes(`data-sel="c:${id}"`), `${id} links to existing component details`);
  }
  assert.equal(html.includes("8 heads · modified"), false);
  assert.ok(html.includes("Token IDs"));
  assert.ok(html.includes("Next-token logits"));
});

test("a diagram never drops training settings or components omitted from the recorded path", () => {
  const m = model();
  m.arch.diagram = { ...topology(), flow: [{ component: "embedding" }] };
  const html = render(m);
  for (const c of m.comps) {
    assert.ok(html.includes(escapeHTML(c.name)), `${c.id} name is retained`);
    assert.ok(html.includes(escapeHTML(c.baseline.label)), `${c.id} value is retained`);
  }
  assert.ok(hasClass(html, "diagram-settings"));
});

test("current shows kept mapped labels without claiming that baseline wiring is still current", () => {
  const m = model();
  m.arch.diagram = topology();
  const c = m.byId.get("attention");
  c.current = { n: 7, status: "keep" };
  c.currentLabel = "8 heads · RoPE θ 20,000";
  c.currentCost = "O(T²d)";
  const html = render(m, "current");
  assertFallback(html);
  assert.ok(html.includes(escapeHTML(c.currentLabel)));
  assert.ok(html.includes('data-sel="e7"'), "current value opens its supporting experiment");
  assert.ok(html.includes(escapeHTML(m.byId.get("optimizer").baseline.label)));
});

test("history-only maps cannot be mistaken for reconstructed model architectures", () => {
  const m = model();
  m.arch.diagram = topology();
  m.arch.history_only = true;
  const html = render(m);
  assertFallback(html);
  assert.match(html, /Research category map/i);
  for (const c of m.comps) assert.ok(html.includes(escapeHTML(c.baseline.label)));
});

test("projects without explicit topology use all recorded values without fake flow edges", () => {
  const m = model();
  delete m.arch.diagram;
  const html = render(m);
  assertFallback(html);
  assert.match(html, /connections not recorded/i);
  for (const c of m.comps) assert.ok(html.includes(escapeHTML(c.baseline.label)));
});

test("project-provided labels, descriptions, node titles, and IDs are HTML escaped", () => {
  const m = model();
  const c = m.byId.get("embedding");
  const oldId = c.id;
  c.id = 'embed" data-injected="yes';
  c.name = '<img src=x onerror="alert(1)">';
  c.baseline.label = '<script>alert("label")</script>';
  c.baseline.detail = '<svg onload="alert(1)">';
  m.arch.components.find(item => item.id === oldId).id = c.id;
  m.byId.delete(oldId);
  m.byId.set(c.id, c);
  m.arch.diagram = {
    version: 1, basis: '<b onclick="alert(1)">basis</b>',
    description: '<img src=x onerror="alert(2)">',
    flow: [
      { kind: "input", title: '<script>alert("title")</script>', note: '<iframe src="bad">' },
      { component: c.id },
    ],
  };
  const html = render(m);
  assert.ok(hasClass(html, "diagram-flow"));
  assert.ok(html.includes(escapeHTML(c.name)));
  assert.ok(html.includes(escapeHTML(c.baseline.label)));
  assert.ok(html.includes(`data-comp="${escapeHTML(c.id)}"`));
  assert.ok(html.includes(escapeHTML(m.arch.diagram.flow[0].title)));
  assert.ok(html.includes(escapeHTML(m.arch.diagram.description)));
  assert.doesNotMatch(html, /<(script|img|iframe)\b/i);
  assert.doesNotMatch(html, /\sdata-injected="yes"/);
});

test("unknown or malicious presentation tones cannot create CSS classes or HTML attributes", () => {
  const m = model();
  m.arch.diagram = { version: 1, flow: [{ component: "embedding", tone: "attention" }] };
  assert.ok(hasClass(render(m), "schematic-tone-attention"), "documented tones remain available");
  for (const tone of ["unrecognized-tone", "attention is-hidden", 'attention" onload="alert(1)', { name: "attention" }]) {
    m.arch.diagram.flow[0].tone = tone;
    const html = render(m);
    assert.ok(hasClass(html, "diagram-flow"), "unknown optional styling does not discard valid topology");
    assert.ok(hasClass(html, "schematic-tone-neutral"), "unknown tones use a neutral presentation");
    assert.equal(hasClass(html, "is-hidden"), false);
    assert.doesNotMatch(html, /onload=|unrecognized-tone|schematic-tone-\[object/);
  }
});

test("invalid graph schemas and unknown component references fall back without dropping data", () => {
  const cases = [
    { version: 2, flow: [{ component: "embedding" }] },
    { version: 1, flow: [] },
    { version: 1, flow: "embedding" },
    { version: 1, flow: [null] },
    { version: 1, flow: [{ component: "missing_component" }] },
    { version: 1, flow: [{ kind: "unsupported", title: "Mystery" }] },
    { version: 1, flow: [{ kind: "repeat", children: [] }] },
  ];
  for (const diagram of cases) {
    const m = model();
    m.arch.diagram = diagram;
    const html = render(m);
    assertFallback(html);
    for (const c of m.comps) assert.ok(html.includes(escapeHTML(c.baseline.label)));
  }
});

test("over-nested and cyclic graphs are bounded and safely recover to component cards", () => {
  const m = model();
  let node = { component: "embedding" };
  for (let i = 0; i < 30; i++) node = { kind: "repeat", title: "Nested", children: [node] };
  m.arch.diagram = { version: 1, flow: [node] };
  assertFallback(render(m));
  const cycle = { kind: "repeat", title: "Cycle", children: [] };
  cycle.children.push(cycle);
  m.arch.diagram = { version: 1, flow: [cycle] };
  assertFallback(render(m));
  m.arch.diagram = { version: 1, flow: Array.from({ length: 200 }, () => ({ component: "embedding" })) };
  assertFallback(render(m));
});


test("unmapped experiment categories are not invented baseline components", () => {
  const m = model();
  const inferred = { id: "experiment_only", name: "Experiment-only category", group: "components", baseline: { label: "baseline" } };
  m.comps.push(inferred);
  m.byId.set(inferred.id, inferred);
  m.groups.push({id: "components", name: "Components", comps: [inferred]});
  m.arch.diagram = topology();
  const html = render(m);
  assert.doesNotMatch(html, /Experiment-only category/);
  m.arch.diagram.flow = [{component: inferred.id}];
  assertFallback(render(m));
});

test("the schematic keeps forward execution order while positioning inputs below outputs", () => {
  const m = model();
  m.arch.diagram = topology();
  const html = render(m);
  assert.match(html, /data-flow-direction=["']bottom-up["']/);
  const groups = schematicGroups(html);
  const component = (id) => groups.find((g) => g.buttons.some((b) => b["data-comp"] === id));
  const path = [
    groups.find((g) => g.attrs["data-schematic-kind"] === "input"),
    component("embedding"), component("attention"), component("ffn"), component("head"),
    groups.find((g) => g.attrs["data-schematic-kind"] === "output"),
  ];
  const ys = path.map(yOf);
  for (let i = 1; i < ys.length; i++) {
    assert.ok(ys[i - 1] > ys[i], "each successive model operation is above its input");
  }
  assert.ok(hasClass(html, "diagram-repeat"), "the shared Transformer block is visibly repeated");
  assert.match(html, /data-flow-edge=["']sequence["']/, "recorded sequence has explicit connectors");
});

test("normalization stays before attention and feed-forward in the bottom-up repeated block", () => {
  const m = model();
  m.arch.diagram = topology();
  const groups = schematicGroups(render(m));
  const attention = groups.find((g) => g.buttons.some((b) => b["data-comp"] === "attention"));
  const ffn = groups.find((g) => g.buttons.some((b) => b["data-comp"] === "ffn"));
  const norms = groups.filter((g) => g.attrs["data-schematic-kind"] === "operation").map(yOf).sort((a, b) => b - a);
  assert.equal(norms.length, 2);
  assert.ok(norms[0] > yOf(attention), "attention's RMSNorm is below attention");
  assert.ok(yOf(attention) > norms[1], "the FFN normalization occurs after attention");
  assert.ok(norms[1] > yOf(ffn), "FFN's RMSNorm is below feed-forward");
});

test("residual bypasses and adds appear only when explicitly present in the recorded graph", () => {
  const m = model();
  m.arch.diagram = topology();
  const countClass = (html, cls) => [...html.matchAll(/<[\w:-]+\b[^>]*>/g)]
    .filter((match) => (attributes(match[0]).class || "").split(/\s+/).includes(cls)).length;
  const html = render(m);
  assert.equal(countClass(html, "schematic-bypass"), 2, "both residual branches bypass their model operation");
  assert.equal(countClass(html, "schematic-add"), 2, "each residual branch has an explicit add");
  m.arch.diagram.flow = [{ component: "embedding" }, { component: "head" }];
  const plain = render(m);
  assert.equal(countClass(plain, "schematic-bypass"), 0);
  assert.equal(countClass(plain, "schematic-add"), 0);
});

test("every interactive schematic component is a named native button accessible by keyboard", () => {
  const m = model();
  m.arch.diagram = topology();
  const groups = schematicGroups(render(m));
  for (const id of ["embedding", "blocks", "attention", "ffn", "head"]) {
    const button = groups.flatMap((group) => group.buttons).find((b) => b["data-comp"] === id);
    assert.ok(button, `${id} is a native button rather than a click-only SVG element`);
    assert.equal(button.type, "button");
    assert.notEqual(button.tabindex, "-1", `${id} remains in keyboard focus order`);
    assert.equal(button["data-sel"], `c:${id}`);
    assert.ok(button["aria-label"]?.includes(escapeHTML(m.byId.get(id).name)), `${id} has an accessible name`);
    assert.ok(button["aria-label"]?.includes(escapeHTML(m.byId.get(id).baseline.label)), `${id} names its value`);
  }
});
