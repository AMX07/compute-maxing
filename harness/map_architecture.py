"""
Keep a workspace's architecture.json up to date: the component map that the web UI's Baseline, Evolution and Current
views draw (schema: webui/README.md, "architecture.json").

    python3 harness/map_architecture.py --workspace workspaces/fixed             # map whatever is new
    python3 harness/map_architecture.py --workspace workspaces/fixed --dry-run   # print the merged map, write nothing

It reads the baseline train.py, results.tsv, each runs/<id>/idea.json and change.diff, and memory/ (state.json,
playbook.json, ideas.jsonl). The first run asks the LLM for the baseline's components, once. After that, experiments
missing from "changes" are mapped in one small call per experiment (up to BATCH per call when catching up): the
component each one changed, the hypotheses it tested, and the component each queued idea would change.
step.record() runs this after every experiment, for both designs. Calls go to Crusoe through harness.llm, on the
reflector's model (config.json "mapper" overrides it). The prompt is harness/prompts/mapper.md.

Merging only adds, so a hand-written map keeps every entry:
  - components, groups, changes{n} and next{idea} get only missing keys; tried[] only grows;
  - an existing hypothesis keeps its text, its evidence only grows, and its status changes only along with new
    evidence. Hypotheses this script adds are marked "auto": true, and only those may be reworded later.
A merge that would drop or rewrite an existing entry is refused, and the file is left as it was.
"""

import argparse
import copy
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from harness import llm  # noqa: E402
from harness import workspace as wsp  # noqa: E402

PROMPT = os.path.join(ROOT, "harness", "prompts", "mapper.md")   # a project's own prompts/mapper.md wins (main)
METRIC = "val_loss"              # the project's metric name, from its project.json (main)
STATUSES = ("validated", "invalidated", "inconclusive", "untested")
ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,23}$")
BATCH = 4            # experiments per call when catching up on several
MAX_IDEAS = 12       # queued ideas shown per call
MAX_NEW_HYPS = 2     # new hypotheses accepted per call
DIFF_CHARS = 2500


def log(msg):
    print(f"[map {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read(path):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return ""


def load_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def clip(s, n):
    s = re.sub(r"\s+", " ", str(s or "")).strip()
    return s if len(s) <= n else s[:n - 1].rstrip() + "…"


def git(ws, *args):
    r = subprocess.run(["git", "-C", ws, *args], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else ""


# ---------------------------------------------------------------------------
# Reading the workspace
# ---------------------------------------------------------------------------

def to_float(s):
    try:
        v = float(s)
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


def experiments(ws):
    """One dict per results.tsv row (n = row index, 0 = the baseline): idea, outcome, metrics, and the best kept
    experiment before it (`ref`), for notes like "steps 2719 → 1525"."""
    state = load_json(os.path.join(ws, "memory", "state.json"), {}) or {}
    hist = {h.get("n"): h for h in state.get("history", []) if isinstance(h, dict)}
    lines = read(os.path.join(ws, "results.tsv")).splitlines()
    cols = [c.strip() for c in lines[0].split("\t")] if lines else []
    has_run = "run_id" in cols                     # upstream autoresearch's results.tsv has no run_id column
    width = 6 if has_run else 5
    rows = [l.split("\t", width - 1) for l in lines[1:]]
    exps, best = [], None
    for n, parts in enumerate(r for r in rows if len(r) >= width):   # numbered like step.read_results()
        if has_run:
            commit, val, _mem, status, run_id, desc = parts[:6]
        else:
            (commit, val, _mem, status, desc), run_id = parts[:5], "-"
        run = os.path.join(ws, "runs", run_id) if run_id not in ("", "-") else None
        idea = (load_json(os.path.join(run, "idea.json"), {}) if run else {}) or {}
        outcome, h = idea.get("outcome") or {}, hist.get(n, {})
        e = {"n": n, "status": status, "val_loss": to_float(val), "description": desc, "commit": commit, "run": run,
             "title": idea.get("title") or h.get("title") or desc, "category": idea.get("category") or h.get("category"),
             "rationale": idea.get("rationale"), "predicted": (idea.get("prediction") or {}).get("delta_pct", h.get("prediction_pct")),
             "check": outcome.get("prediction_check") or h.get("prediction_check"), "error": outcome.get("error_tail"),
             "idea_id": idea.get("idea_id") or h.get("idea_id"),
             "metrics": (load_json(os.path.join(run, "metrics.json"), {}) if run else {}) or {}, "ref": best,
             "delta": outcome.get("delta_pct", h.get("actual_pct"))}
        if e["delta"] is None and best and e["val_loss"]:
            e["delta"] = (e["val_loss"] - best["val_loss"]) / best["val_loss"] * 100
        if status == "keep" and e["val_loss"] and (best is None or e["val_loss"] < best["val_loss"]):
            best = e
        exps.append(e)
    return exps


def load_ideas(ws):
    """memory/ideas.jsonl folded by id: a later line with the same id updates the idea."""
    bank = {}
    for line in read(os.path.join(ws, "memory", "ideas.jsonl")).splitlines():
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("id"), str):
            bank.setdefault(obj["id"], {}).update(obj)
    return list(bank.values())


def baseline_src(ws, exps):
    """The train.py the baseline ran: its run's copy, else the workspace's train.py at the baseline commit, else the
    external image template (only for workspaces in this repo)."""
    b = exps[0] if exps else {}
    if b.get("run") and os.path.exists(os.path.join(b["run"], "train.py")):
        return read(os.path.join(b["run"], "train.py"))
    src = git(ws, "show", f"{b['commit']}:./train.py") if b.get("commit") else ""
    template = read(os.path.join(wsp.IMAGE_PROJECT, "train.py")) if os.path.dirname(ws) == os.path.join(ROOT, "workspaces") else ""
    return src or template or read(os.path.join(ws, "train.py"))


def diff_of(ws, e):
    text = read(os.path.join(e["run"], "change.diff")) if e["run"] else ""
    if not text and e["commit"]:
        text = git(ws, "show", "--format=", e["commit"], "--", "train.py")
    body = "\n".join(l for l in text.splitlines() if not l.startswith(("diff --git", "index ", "--- ", "+++ ")))
    return body if len(body) <= DIFF_CHARS else body[:DIFF_CHARS] + "\n... (truncated)"


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def run_numbers(e):
    m, r = e["metrics"], (e["ref"] or {}).get("metrics", {})
    out = []
    for key, name, unit in (("num_steps", "steps", ""), ("num_params_M", "params", "M"), ("peak_memory_gb", "memory", " GB"),
                            ("peak_vram_mb", "memory", " MB")):
        if m.get(key):  # fake runs report 0
            out.append(f"{name} {m[key]}{unit}" + (f" (best before: {r[key]}{unit})" if r.get(key) else ""))
    return ", ".join(out)


def experiment_text(ws, e):
    lines = [f"### Experiment {e['n']}: {e['title']}" + (f" [{e['category']}]" if e["category"] else "") + f", {e['status']}"]
    if e["description"] and e["description"] != e["title"]:
        lines.append(f"Description: {e['description']}")
    if e["val_loss"]:
        res = f"{METRIC} {e['val_loss']:.6f}"
        if e["delta"] is not None:
            res += f" ({e['delta']:+.2f}% vs the best before" + (f", {e['ref']['val_loss']:.6f}" if e["ref"] else "") + ")"
    else:
        res = f"crashed, no {METRIC}"
    if isinstance(e["predicted"], (int, float)):
        res += f"; predicted {e['predicted']:+.2f}%"
    lines.append(f"Result: {res}")
    if run_numbers(e):
        lines.append(f"Run: {run_numbers(e)}")
    if e["rationale"]:
        lines.append(f"Rationale: {clip(e['rationale'], 300)}")
    if e["check"]:
        lines.append(f"Reflector: {clip(e['check'], 300)}")
    if e["error"]:
        lines.append(f"Crash: {clip(e['error'][-400:], 400)}")
    if e["idea_id"]:
        lines.append(f"Implements queued idea {e['idea_id']}")
    lines.append(f"Diff:\n```diff\n{diff_of(ws, e) or '(not available)'}\n```")
    return "\n".join(lines)


def components_prompt(ws, exps):
    b = exps[0]
    m = b["metrics"]
    facts = [f"{METRIC} {b['val_loss']}" if b["val_loss"] else None,
             f"{m['num_params_M']}M params" if m.get("num_params_M") else None,
             f"{m['num_steps']} steps" if m.get("num_steps") else None,
             f"{m['training_seconds']:.0f} s of training" if m.get("training_seconds") else None,
             f"peak memory {m['peak_memory_gb']} GB" if m.get("peak_memory_gb") else None]
    facts = ", ".join(f for f in facts if f)
    return (f"## Task A: components\n\nBaseline run: {facts or 'no metrics recorded'}.\n\n"
            f"## Baseline train.py\n\n```python\n{baseline_src(ws, exps)}\n```\n\nReply with the Task A JSON.")


def current_labels(arch, exps, upto):
    """Each component's label after the kept changes before experiment `upto` (what the UI calls "before")."""
    cur = {c["id"]: (c.get("baseline") or {}).get("label", "") for c in arch.get("components", [])}
    for e in exps[1:upto]:
        ch = (arch.get("changes") or {}).get(str(e["n"])) or {}
        if e["status"] == "keep" and ch.get("component") in cur:
            cur[ch["component"]] = ch.get("after") or ch.get("label") or cur[ch["component"]]
    return cur


def hyp_key(h):
    return h.get("id") or h.get("text")


def queued(arch, bank):
    tried = set(arch.get("tried") or [])
    return [i for i in bank if (i.get("status") or "new") == "new" and i["id"] not in tried][:MAX_IDEAS]


def update_prompt(ws, arch, exps, batch, ideas, lessons):
    cur = current_labels(arch, exps, batch[0]["n"] if batch else len(exps))
    comps = "\n".join(f"- {c['id']} ({c.get('group', '-')}) {c.get('name', c['id'])}: {cur.get(c['id'], '')}"
                      for c in arch.get("components", []))
    hyps = "\n".join(f"- {hyp_key(h)} [{h.get('status', 'untested')}; exp {', '.join(map(str, h.get('evidence') or [])) or '-'}] "
                     f"{h.get('text', '')}" for h in arch.get("hypotheses") or [] if isinstance(h, dict)) or "(none yet)"
    nxt = arch.get("next") or {}
    idea_lines = "\n".join(f"- {i['id']} [{i.get('category', '-')}]" + (f" (placed: {nxt[i['id']]})" if i["id"] in nxt else "")
                           + f" {clip(i.get('title'), 60)}: {clip(i.get('change'), 160)}" for i in ideas) or "(none)"
    parts = ["## Task B: map new experiments",
             f"## Components (id (group) name: its label now, after every kept change so far)\n\n{comps}",
             f"## Hypotheses so far\n\n{hyps}"]
    if lessons:
        parts.append("## Lessons the reflector just wrote\n\n" + "\n".join(f"- {clip(l.get('text'), 300)}" for l in lessons))
    if batch:
        parts.append("## New experiments, in order (a kept one changes its component's label for the ones after it)\n\n"
                     + "\n\n".join(experiment_text(ws, e) for e in batch))
    else:
        parts.append("## New experiments\n\nNone: only place the queued ideas, and leave changes, hypotheses and tried empty.")
    parts.append(f"## Queued ideas\n\n{idea_lines}")
    parts.append("Reply with the Task B JSON.")
    return "\n\n".join(parts)


def ask(user, tag, max_tokens):
    messages = [{"role": "system", "content": read(PROMPT)}, {"role": "user", "content": user}]
    model = llm.role_model("mapper") or llm.role_model("reflector")
    reply = llm.chat(messages, "mapper", tag, temperature=0.2, max_tokens=max_tokens, timeout=180, model=model)
    return llm.extract_json(reply) or {}


# ---------------------------------------------------------------------------
# Merging (pure: arch, reply -> arch, so a reply can be re-applied to a file a human edited meanwhile)
# ---------------------------------------------------------------------------

def add_group(arch, gid, name=None, flow=False):
    groups = arch.setdefault("groups", [])
    if gid not in {g.get("id") for g in groups}:
        groups.append({"id": gid, "name": clip(name or gid.replace("_", " ").capitalize(), 30), **({"flow": True} if flow else {})})


def is_blank(v):
    return str(v or "").strip().lower() in ("", "-", "—", "none", "n/a", "null", "0")


def add_component(arch, c):
    """Add one component if it's well formed and new. Returns its id, or None."""
    if not isinstance(c, dict) or not ID_RE.match(str(c.get("id", ""))):
        return None
    comps = arch.setdefault("components", [])
    base = c.get("baseline") if isinstance(c.get("baseline"), dict) else {}
    if c["id"] in {x.get("id") for x in comps} or not str(base.get("label", "")).strip():
        return None
    gids = [g.get("id") for g in arch.get("groups", [])]
    group = c.get("group") if ID_RE.match(str(c.get("group", ""))) else (gids[-1] if gids else "model")
    add_group(arch, group)
    baseline = {k: clip(base[k], n) for k, n in (("label", 48), ("detail", 320), ("cost", 64), ("params", 16))
                if str(base.get(k) or "").strip() and not (k == "params" and is_blank(base[k]))}
    comps.append({"id": c["id"], "group": group, "name": clip(c.get("name") or c["id"], 32), "baseline": baseline})
    return c["id"]


def merge_components(arch, rep, metric):
    """Task A reply -> name, summary, metric, groups and components, each only if missing."""
    arch, out = copy.deepcopy(arch), {}
    for k in ("name", "summary"):
        v = arch.get(k) or clip(rep.get(k), 80 if k == "name" else 400)
        if v:
            out[k] = v
    out["metric"] = arch.get("metric") or metric
    out.update({k: v for k, v in arch.items() if k not in out})
    for g in rep.get("groups") or []:
        if isinstance(g, dict) and ID_RE.match(str(g.get("id", ""))):
            add_group(out, g["id"], g.get("name"), bool(g.get("flow")))
    for c in rep.get("components") or []:
        add_component(out, c)
    if out.get("groups"):  # drop groups the LLM named but put nothing in, unless they were already there
        had = {g.get("id") for g in arch.get("groups") or []}
        used = {c.get("group") for c in out.get("components", [])}
        out["groups"] = [g for g in out["groups"] if g["id"] in used or g["id"] in had]
    return out


def clean_change(c, comp_ids):
    if not isinstance(c, dict) or c.get("component") not in comp_ids or not str(c.get("label") or "").strip():
        return None
    out = {"component": c["component"]}
    for k, n in (("label", 40), ("after", 64), ("cost", 80), ("note", 240)):
        v = clip(c.get(k), n)
        if not is_blank(v):
            out[k] = v
    return out


def next_hyp_id(hyps):
    nums = [int(m.group(1)) for h in hyps if isinstance(h, dict) and (m := re.match(r"^H(\d+)$", str(h.get("id", ""))))]
    return f"H{max(nums, default=0) + 1}"


def merge_update(arch, rep, ns, idea_ids, all_ns):
    """Task B reply -> new changes for experiments `ns`, hypothesis updates backed by them, next for `idea_ids`."""
    arch = copy.deepcopy(arch)
    for c in (rep.get("new_components") or [])[:1]:
        add_component(arch, c)
    comp_ids = {c.get("id") for c in arch.get("components", [])}
    changes = arch.setdefault("changes", {})
    for k, c in (rep.get("changes") or {}).items():
        k = str(k).strip()
        if k.isdigit() and int(k) in ns and k not in changes:
            ch = clean_change(c, comp_ids)
            if ch:
                changes[k] = ch
            else:
                log(f"exp {k}: unusable change {json.dumps(c)[:200]}; it will be retried next time")

    hyps = arch.setdefault("hypotheses", [])
    by_key = {hyp_key(h): h for h in hyps if isinstance(h, dict)}
    added = 0
    for h in rep.get("hypotheses") or []:
        if not isinstance(h, dict):
            continue
        status = h.get("status") if h.get("status") in STATUSES else None
        ev = sorted({int(x) for x in h.get("evidence") or [] if str(x).strip().isdigit()})
        old = by_key.get(h.get("id"))
        if old is not None:
            grown = [x for x in ev if x in ns and x not in (old.get("evidence") or [])]
            if not grown:
                continue  # an experiment is judged once, when it is mapped
            old["evidence"] = sorted(set(old.get("evidence") or []) | set(grown))
            if status:
                old["status"] = status
            if old.get("auto") and str(h.get("text") or "").strip():
                old["text"] = clip(h["text"], 220)
        elif str(h.get("text") or "").strip() and status and added < MAX_NEW_HYPS:
            text = clip(h["text"], 220)
            ev = [x for x in ev if x in all_ns]
            if not set(ev) & ns or text.lower() in {str(x.get("text", "")).lower() for x in hyps if isinstance(x, dict)}:
                continue
            new = {"id": next_hyp_id(hyps), "status": status, "evidence": ev, "text": text, "auto": True}
            hyps.append(new)
            by_key[new["id"]] = new
            added += 1

    nxt = arch.setdefault("next", {})
    for iid, comp in (rep.get("next") or {}).items():
        if iid in idea_ids and iid not in nxt and comp in comp_ids:
            nxt[iid] = comp
    if ns:
        tried = arch.setdefault("tried", [])
        tried += [i for i in rep.get("tried") or [] if i in idea_ids and i not in tried]
    return arch


def mark_tried(arch, exps, bank):
    """Ideas the bank marks as tried, or that an experiment says it implemented. No LLM needed."""
    known = {i["id"] for i in bank}
    ran = [i["id"] for i in bank if (i.get("status") or "new") != "new"] + [e["idea_id"] for e in exps if e["idea_id"] in known]
    tried = list(arch.get("tried") or [])
    tried += [i for i in dict.fromkeys(ran) if i not in tried]
    if tried != (arch.get("tried") or []):
        arch = {**arch, "tried": tried}
    return arch


def problems(old, new):
    """Ways `new` drops or rewrites something `old` had. Empty means the merge only added."""
    out = []
    for k, v in old.items():
        nv = new.get(k)
        if k not in new:
            out.append(f"drops {k}")
        elif k == "hypotheses" and isinstance(v, list) and isinstance(nv, list):
            nh = {hyp_key(h): h for h in nv if isinstance(h, dict)}
            for h in v:
                m = nh.get(hyp_key(h)) if isinstance(h, dict) else None
                if not isinstance(h, dict):
                    continue
                if m is None:
                    out.append(f"drops hypothesis {hyp_key(h)}")
                    continue
                ev, mev = set(h.get("evidence") or []), set(m.get("evidence") or [])
                if not ev <= mev:
                    out.append(f"drops evidence from {hyp_key(h)}")
                if m.get("status") != h.get("status") and ev == mev:
                    out.append(f"changes {hyp_key(h)}'s status without new evidence")
                if not h.get("auto") and m.get("text") != h.get("text"):
                    out.append(f"rewrites {hyp_key(h)}")
        elif k == "tried" and isinstance(v, list) and isinstance(nv, list):
            if not set(v) <= set(nv):
                out.append("drops tried ideas")
        elif k in ("components", "groups") and isinstance(v, list) and isinstance(nv, list):
            nid = {x.get("id"): x for x in nv if isinstance(x, dict)}
            out += [f"rewrites {k[:-1]} {x.get('id')}" for x in v if isinstance(x, dict) and nid.get(x.get("id")) != x]
        elif isinstance(v, dict) and isinstance(nv, dict):
            out += [f"rewrites {k}.{kk}" for kk, vv in v.items() if nv.get(kk) != vv]
        elif nv != v:
            out.append(f"rewrites {k}")
    return out


def dump(arch):
    """One line per group, component, change and hypothesis, so a diff shows exactly what was added."""
    lines, items = [], list(arch.items())
    for i, (k, v) in enumerate(items):
        end = "," if i < len(items) - 1 else ""
        if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
            body = ",\n".join("    " + json.dumps(x, ensure_ascii=False) for x in v)
            lines.append(f"  {json.dumps(k)}: [\n{body}\n  ]{end}")
        elif isinstance(v, dict) and v and all(isinstance(x, dict) for x in v.values()):
            body = ",\n".join(f"    {json.dumps(kk)}: {json.dumps(x, ensure_ascii=False)}" for kk, x in v.items())
            lines.append(f"  {json.dumps(k)}: {{\n{body}\n  }}{end}")
        else:
            lines.append(f"  {json.dumps(k)}: {json.dumps(v, ensure_ascii=False)}{end}")
    return "{\n" + "\n".join(lines) + "\n}\n"


# ---------------------------------------------------------------------------
# One update
# ---------------------------------------------------------------------------

def update(ws, dry_run=False):
    path = os.path.join(ws, "architecture.json")
    exps = experiments(ws)
    if not exps:
        log("no results yet; nothing to map")
        return 0
    disk = load_json(path, {}) if os.path.exists(path) else {}
    if not isinstance(disk, dict) or (os.path.exists(path) and load_json(path) is None):
        log(f"{path} is not a JSON object; leaving it alone")
        return 1
    bank = load_ideas(ws)
    lessons = (load_json(os.path.join(ws, "memory", "playbook.json"), {}) or {}).get("lessons") or []
    all_ns = {e["n"] for e in exps if e["n"] > 0}
    steps = [lambda a: mark_tried(a, exps, bank)]   # replies, as merges to re-apply if the file changes meanwhile
    arch = steps[0](disk)

    if not arch.get("components"):
        log("no components yet: asking for the baseline's components (once)")
        try:
            rep = ask(components_prompt(ws, exps), "exp000", 8192)
        except RuntimeError as e:
            log(f"{e}; will try again next time")
            return 1
        steps.append(lambda a, rep=rep: merge_components(a, rep, {"name": METRIC, "goal": "min"}))
        arch = steps[-1](arch)
        if not arch.get("components"):
            log("the reply had no usable components; will try again next time")
            return 1

    todo = [e for e in exps if e["n"] > 0 and str(e["n"]) not in (arch.get("changes") or {})]
    batches = [todo[i:i + BATCH] for i in range(0, len(todo), BATCH)]
    ideas = queued(arch, bank)
    if not batches and any(i["id"] not in (arch.get("next") or {}) for i in ideas):
        batches = [[]]  # only new ideas to place
    for batch in batches:
        ns = {e["n"] for e in batch}
        ideas = queued(arch, bank)
        idea_ids = {i["id"] for i in ideas}
        fresh = [l for l in lessons if l.get("updated") in ns or l.get("since") in ns]
        tag = f"exp{max(ns):03d}" if ns else f"exp{len(exps) - 1:03d}"
        log(f"mapping {'experiments ' + ', '.join(map(str, sorted(ns))) if ns else 'queued ideas'}")
        try:
            rep = ask(update_prompt(ws, arch, exps, batch, ideas, fresh), tag, 4096)
        except RuntimeError as e:  # keep what earlier batches mapped; the rest is retried next time
            log(f"{e}; the rest will be retried next time")
            break
        steps.append(lambda a, rep=rep, ns=ns, idea_ids=idea_ids: merge_update(a, rep, ns, idea_ids, all_ns))
        arch = steps[-1](arch)

    if dry_run:
        print(dump(arch), end="")
        return 0
    now = load_json(path, {}) if os.path.exists(path) else {}
    if now != disk:  # edited while we waited on the LLM: apply the same replies to the new version
        log("architecture.json changed meanwhile; merging into the new version")
        arch = now
        for f in steps:
            arch = f(arch)
        disk = now
    if arch == disk:
        log("nothing new to map")
        return 0
    bad = problems(disk, arch)
    if bad:
        log(f"refusing to write: the merge {'; '.join(bad[:5])}")
        return 1
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(dump(arch))
    os.replace(tmp, path)
    added = [f"{len(arch.get(k) or []) - len(disk.get(k) or [])} {k}" for k in ("components", "changes", "hypotheses", "next", "tried")
             if len(arch.get(k) or []) != len(disk.get(k) or [])]
    log(f"architecture.json updated ({', '.join(added) or 'hypotheses revised'})")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Keep a workspace's architecture.json (the web UI's component map) up to date.")
    ap.add_argument("--workspace", required=True, help="the project folder: train.py, results.tsv, runs/, memory/")
    ap.add_argument("--dry-run", action="store_true", help="print the merged map instead of writing it")
    args = ap.parse_args()
    ws = os.path.abspath(args.workspace)
    if not os.path.isdir(ws):
        raise SystemExit(f"{ws} is not a directory")
    llm.TRACE_DIR = os.path.join(ws, "traces")   # this workspace's traces, whatever HARNESS_WORKSPACE says
    global PROMPT, METRIC
    METRIC = wsp.load_profile(ws)["metric"]
    if os.path.exists(os.path.join(ws, "prompts", "mapper.md")):
        PROMPT = os.path.join(ws, "prompts", "mapper.md")
    return update(ws, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
