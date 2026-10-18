"""
Server for the autoresearch UI. Standard library only.

A project is a directory an autoresearch loop works in: train.py, results.tsv, runs/<run_id>/, and optionally
memory/ (playbook.json, ideas.jsonl), traces/status.json, and architecture.json (the component map the
architecture views draw; without it, experiments are grouped by their idea category).

    python3 webui/server.py                                # registers workspaces/*, http://localhost:8080
    python3 webui/server.py --root path/to/project         # also registers that project
    python3 webui/server.py --port 9000 --host 0.0.0.0

Routes:
    /                             the UI (webui/index.html, app.js, style.css)
    /api/projects                 GET recent projects; POST {"path": ...} adds one
    /api/p/<id>                   one project's state: experiments, architecture, lessons, ideas, live status
    /p/<id>/runs/<run_id>/<file>  samples.png, trajectory.png, train.py, change.diff, idea.json, metrics.json
    /api/annotations              GET or POST the design-review notes from /?annotate (webui/annotations.json)
"""

import argparse
import hashlib
import json
import math
import mimetypes
import os
import re
import subprocess
import sys
import time
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

sys.path.insert(0, os.path.dirname(__file__))
import training
import evaluation
import research

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = {"/": "index.html", "/index.html": "index.html", "/app.js": "app.js", "/style.css": "style.css",
          "/annotate.js": "annotate.js"}
ANNOTATIONS = os.path.join(HERE, "annotations.json")  # design-review notes from ?annotate mode
REGISTRY = os.path.expanduser("~/.autoresearch/projects.json")  # recent projects: [{path, added_at, opened_at}]

# The agent harness that runs projects under <repo>/workspaces/. Projects elsewhere are view-only.
REPO = os.path.dirname(HERE)
HARNESS = os.path.join(REPO, "harness")
PROMPTS = os.path.join(HARNESS, "prompts")
STOP_FILE = os.path.join(HARNESS, "STOP")
DESIGNS = {
    "fixed": {"name": "Custom pipeline", "about": "Roles fixed in code: planner → guards → train → fixer on crash → "
              "reflector → playbook, with a background researcher filling the idea bank.",
              "roles": [("planner", "Proposes one experiment and writes the edit to train.py"),
                        ("fixer", "Repairs an edit that crashed"),
                        ("reflector", "Checks the prediction against the result and updates the playbook"),
                        ("researcher", "Runs in the background: searches the web and fills the idea bank")]},
    "agentic": {"name": "Model-driven", "about": "One orchestrator picks an action each turn: spawn sub-agents "
                "(choosing their role, task, context and model), search, read files, update its notes, or run.",
                "roles": [("orchestrator", "Decides the next action, up to 8 turns and 4 sub-agents per experiment"),
                          ("subagent", "A role the orchestrator invents for one task; runs in parallel")]},
}
TRACE_RE = re.compile(r"^exp(\d+)(?:-([a-z]\d+))?-([a-z_]+)-(\d+)\.json$")
LAUNCH_LOCK = threading.Lock()
RUNNERS = {}  # workspace path -> Popen of harness/run.py started from the UI
RUN_FILES = {"samples.png", "trajectory.png", "train.py", "change.diff", "idea.json", "metrics.json"}
RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
HEADER = ["commit", "val_loss", "memory_gb", "status", "run_id", "description"]
LIVE_LOG_SECONDS = 90  # run.log counts as "training now" if written within this many seconds


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def to_float(s):
    try:
        v = float(s)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def read_results(root):
    """Parse results.tsv into a list of dicts, in experiment order. Columns come from the header, so both this
    harness's layout and upstream autoresearch's (commit, val_bpb, memory_gb, status, description) work: the second
    column is the metric, whatever its name, and is returned as "val_loss". Tolerates a missing header or short rows."""
    try:
        with open(os.path.join(root, "results.tsv")) as f:
            raw = f.read().splitlines()
    except OSError:
        return []
    rows, cols = [], HEADER
    for line in raw:
        if not line.strip():
            continue
        if line.split("\t")[0].strip().lower() == "commit":
            cols = [c.strip().lower() for c in line.split("\t")]
            cols = ["commit", "val_loss"] + cols[2:]
            continue
        parts = line.split("\t", len(cols) - 1)
        parts += [""] * (len(cols) - len(parts))
        row = dict(zip(cols, (p.strip() for p in parts)))
        for k in HEADER:
            row.setdefault(k, "")
        row["status"] = row["status"].lower() or "unknown"
        row["val_loss"] = to_float(row["val_loss"])
        if row["status"] == "crash":
            row["val_loss"] = None
        row["memory_gb"] = to_float(row["memory_gb"])
        if row["run_id"] in ("", "-"):
            row["run_id"] = None
        rows.append(row)
    return rows


def run_started(run_id, run_dir, metrics):
    """Epoch seconds the run was saved: metrics.created_at, else the run_id timestamp, else the dir mtime."""
    for text, fmt in (((metrics or {}).get("created_at"), "%Y-%m-%dT%H:%M:%S"), (run_id, "%Y%m%d-%H%M%S")):
        try:
            return datetime.strptime(str(text)[:19], fmt).timestamp()
        except (TypeError, ValueError):
            pass
    try:
        return os.path.getmtime(run_dir)
    except OSError:
        return None


def results_metric(root):
    """The metric's name: architecture.json, else the second column of results.tsv's header, else val_loss."""
    name = ((read_json(os.path.join(root, "architecture.json")) or {}).get("metric") or {}).get("name")
    if isinstance(name, str) and name:
        return name
    research_config = (read_json(os.path.join(root, "project.json")) or {}).get("research") or {}
    if research_config.get("enabled") and research_config.get("metric"):
        return research_config["metric"]
    try:
        with open(os.path.join(root, "results.tsv")) as f:
            head = f.readline().rstrip("\n").split("\t")
        if head[0].strip().lower() == "commit" and len(head) > 1:
            return head[1].strip()
    except OSError:
        pass
    return "val_loss"


def results_goal(root):
    metric = (read_json(os.path.join(root, "architecture.json")) or {}).get("metric") or {}
    if metric.get("goal") in ("min", "max"):
        return metric["goal"]
    config = (read_json(os.path.join(root, "project.json")) or {}).get("research") or {}
    return config.get("goal", "min") if config.get("enabled") else "min"


HASH_RE = re.compile(r"^[0-9a-f]{7,40}$")
GIT_DIFFS = {}  # (root, commit) -> {"add", "del"} or None; commits never change, so this never goes stale


def git_diff(root, commit, stats_only=True):
    """train.py's change in a commit of the project's own repo (upstream autoresearch keeps no run folders, but its
    kept experiments are commits). Returns line counts, or the diff text with stats_only=False; None if unknown."""
    if not commit or not HASH_RE.match(commit):
        return None
    key = (root, commit)
    if stats_only and key in GIT_DIFFS:
        return GIT_DIFFS[key]
    args = ["git", "-C", root, "show", "--format=", "--numstat" if stats_only else "--patch", commit, "--", "train.py"]
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode:
        out = None
    elif not stats_only:
        return r.stdout
    else:
        nums = r.stdout.split()
        out = {"add": int(nums[0]), "del": int(nums[1])} if len(nums) >= 2 and nums[0].isdigit() else None
    GIT_DIFFS[key] = out
    return out


def diff_stats(path):
    """Lines added and removed in a unified diff, or None if there is no diff."""
    try:
        with open(path, errors="replace") as f:
            lines = f.read().splitlines()
    except OSError:
        return None
    add = sum(1 for l in lines if l.startswith("+") and not l.startswith("+++"))
    rem = sum(1 for l in lines if l.startswith("-") and not l.startswith("---"))
    return {"add": add, "del": rem}


def run_info(root, run_id, prefix=""):
    """Everything the UI needs from runs/<run_id>/. Image URLs carry the file mtime so the browser refetches on change."""
    run_dir = os.path.join(root, "runs", run_id)
    if not os.path.isdir(run_dir):
        return None
    metrics = read_json(os.path.join(run_dir, "metrics.json"))
    files = {}
    for name in RUN_FILES:
        path = os.path.join(run_dir, name)
        try:
            files[name] = f"{prefix}/runs/{run_id}/{name}?v={int(os.path.getmtime(path))}"
        except OSError:
            pass
    idea = read_json(os.path.join(run_dir, "idea.json")) if "idea.json" in files else None
    if idea is not None and not isinstance(idea, dict):
        idea = None
    return {"run_id": run_id, "metrics": metrics, "files": files, "idea": idea,
            "diff": diff_stats(os.path.join(run_dir, "change.diff")),
            "started_at": run_started(run_id, run_dir, metrics)}


def live_log(root):
    """Last progress line of run.log if it is being written right now (training is in progress)."""
    path = os.path.join(root, "run.log")
    try:
        mtime = os.path.getmtime(path)
        if time.time() - mtime > LIVE_LOG_SECONDS:
            return None
        with open(path, "rb") as f:
            f.seek(max(0, os.path.getsize(path) - 4096))
            tail = f.read().decode("utf-8", "replace")
    except OSError:
        return None
    segments = [s.strip() for s in re.split(r"[\r\n]", tail) if s.strip()]
    step = next((s for s in reversed(segments) if s.startswith("step ")), None)
    out = {"updated_at": mtime, "line": segments[-1][:200] if segments else ""}
    if step:
        m = re.search(r"\(([\d.]+)%\)", step)
        out["pct"] = to_float(m.group(1)) if m else None
        m = re.search(r"remaining:\s*([\d.]+)s", step)
        out["remaining_s"] = to_float(m.group(1)) if m else None
        m = re.search(r"loss:\s*([\d.]+)", step)
        out["train_loss"] = to_float(m.group(1)) if m else None
        out["finished"] = segments[-1] != step  # anything printed after the step line means training ended
    return out


def build_state(root, prefix=""):
    rows = read_results(root)
    arch = read_json(os.path.join(root, "architecture.json"))
    goal = results_goal(root)
    try:
        run_ids = sorted(d for d in os.listdir(os.path.join(root, "runs"))
                         if RUN_ID_RE.match(d) and os.path.isdir(os.path.join(root, "runs", d)))
    except OSError:
        run_ids = []

    baseline = rows[0]["val_loss"] if rows else None
    best = None        # running best over kept rows within the current comparison series
    best_idx = None
    experiments = []
    previous_group = None
    for i, row in enumerate(rows):
        group = row.get("comparison_group") or None
        reset = i > 0 and group != previous_group
        if reset:
            best, best_idx = None, None
        previous_group = group
        info = run_info(root, row["run_id"], prefix) if row["run_id"] and RUN_ID_RE.match(row["run_id"]) else None
        v = row["val_loss"]
        exp = {
            "idx": i, "commit": row["commit"] or None, "status": row["status"], "val_loss": v,
            "comparison_reset": reset, "comparison_group": group,
            "memory_gb": row["memory_gb"], "run_id": row["run_id"], "description": row["description"],
            "best_before": best,
            "delta_baseline": (v - baseline) if v is not None and baseline is not None and i > 0
                              and group == (rows[0].get("comparison_group") or None) else None,
            "delta_best": (v - best) if v is not None and best is not None else None,
        }
        exp.update({k: info[k] for k in ("metrics", "files", "idea", "diff", "started_at")} if info else
                   {"metrics": None, "files": {}, "idea": None, "diff": None, "started_at": None})
        if "change.diff" not in exp["files"] and i > 0:
            stats = git_diff(root, row["commit"])
            if stats:
                exp["files"] = {**exp["files"], "change.diff": f"{prefix}/diff/{row['commit']}"}
                exp["diff"] = stats
        if row["status"] in ("keep", "retime") and v is not None and (best is None or (v > best if goal == "max" else v < best)):
            best, best_idx = v, i
        exp["best_after"] = best
        experiments.append(exp)

    # Run dirs not in results.tsv and newer than every logged run: saved, but the agent has not logged them yet.
    logged = {r["run_id"] for r in rows if r["run_id"]}
    newest_logged = max(logged) if logged else ""
    pending = []
    for rid in run_ids:
        if rid in logged or rid <= newest_logged:
            continue
        info = run_info(root, rid, prefix)
        if info:
            info["stage"] = "logging" if "samples.png" in info["files"] else "sampling"
            pending.append(info)

    counts = {}
    for e in experiments:
        counts[e["status"]] = counts.get(e["status"], 0) + 1
    prompt = next((e["metrics"].get("prompt") for e in list(reversed(experiments)) + pending
                   if e.get("metrics") and e["metrics"].get("prompt")), None)
    # A workspace keeps traces/ and memory/ at its top level; the older layout kept them under harness/.
    def first(*parts_list):
        for parts in parts_list:
            path = os.path.join(root, *parts)
            if os.path.exists(path):
                return path
        return os.path.join(root, *parts_list[0])
    status = read_json(first(("traces", "status.json"), ("harness", "traces", "status.json")))
    if not isinstance(status, dict) or time.time() - status.get("since", 0) > 20 * 60:
        status = None  # stale: the loop isn't running
    book = read_json(first(("memory", "playbook.json"), ("harness", "memory", "playbook.json")))
    lessons = book.get("lessons", []) if isinstance(book, dict) else []
    try:
        with open(os.path.join(root, "memory", "notes.md")) as f:
            notes = f.read()
    except OSError:
        notes = ""
    design = (read_json(os.path.join(root, "workspace.json")) or {}).get("design")
    return {
        "root": os.path.abspath(root),
        "name": project_name(root),
        "metric": results_metric(root), "goal": goal,
        "evaluation": evaluation.describe(root, results_metric(root), arch),
        "server_time": time.time(),
        "harness": {"status": status, "lessons": lessons, "notes": notes, "design": design},
        "architecture": arch if isinstance(arch, dict) else None,
        "ideas": read_jsonl(first(("memory", "ideas.jsonl"), ("harness", "memory", "ideas.jsonl"))),
        "prompt": prompt,
        "experiments": experiments,
        "pending": pending,
        "live": live_log(root),
        "summary": {"baseline_val_loss": baseline, "best_val_loss": best, "best_idx": best_idx,
                    "counts": counts, "total": len(experiments)},
    }


def read_jsonl(path):
    out = []
    try:
        with open(path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if isinstance(d, dict):
                    out.append(d)
    except OSError:
        pass
    return out


def project_name(root):
    """architecture.json "name", else workspace.json "name", else the directory name."""
    for fname in ("architecture.json", "workspace.json"):
        name = (read_json(os.path.join(root, fname)) or {}).get("name") if fname else None
        if isinstance(name, str) and name.strip():
            return name.strip()
    return os.path.basename(os.path.normpath(root))


def project_id(path):
    return hashlib.sha1(path.encode()).hexdigest()[:10]


def load_registry():
    data = read_json(REGISTRY)
    return [p for p in data if isinstance(p, dict) and isinstance(p.get("path"), str)] if isinstance(data, list) else []


def register(path, opened=False):
    """Add a project directory to the recent list (and mark it opened). Returns its entry."""
    path = os.path.abspath(os.path.expanduser(path))
    items = load_registry()
    entry = next((p for p in items if p["path"] == path), None)
    if entry is None:
        entry = {"path": path, "added_at": time.time()}
        items.append(entry)
    if opened:
        entry["opened_at"] = time.time()
    os.makedirs(os.path.dirname(REGISTRY), exist_ok=True)
    tmp = REGISTRY + ".tmp"
    with open(tmp, "w") as f:
        json.dump(items, f, indent=2)
    os.replace(tmp, REGISTRY)
    return entry


def is_project(path):
    return os.path.isdir(path) and any(os.path.exists(os.path.join(path, f)) for f in ("results.tsv", "train.py", "runs"))


def projects():
    """id -> registry entry, for the registered directories that still exist."""
    return {project_id(p["path"]): p for p in load_registry() if os.path.isdir(p["path"])}


def project_card(pid, entry):
    """The landing page's view of a project: name, path, best-so-far curve, and when it last changed."""
    root = entry["path"]
    rows = read_results(root)
    goal = results_goal(root)
    best, curve = None, []
    previous_group = None
    for r in rows:
        group = r.get("comparison_group") or None
        if group != previous_group:
            best = None
        previous_group = group
        if r["status"] in ("keep", "retime") and r["val_loss"] is not None and (best is None or (r["val_loss"] > best if goal == "max" else r["val_loss"] < best)):
            best = r["val_loss"]
        curve.append(best)
    mtimes = []
    for f in ("results.tsv", "run.log", "train.py"):
        try:
            mtimes.append(os.path.getmtime(os.path.join(root, f)))
        except OSError:
            pass
    status = read_json(os.path.join(root, "traces", "status.json"))
    live = isinstance(status, dict) and status.get("phase") not in (None, "stopped") and \
        time.time() - status.get("since", 0) < 20 * 60
    return {"id": pid, "name": project_name(root), "path": root, "total": len(rows),
            "baseline": rows[0]["val_loss"] if rows else None, "best": best, "curve": curve, "goal": goal,
            "updated_at": max(mtimes) if mtimes else None, "opened_at": entry.get("opened_at"),
            "has_map": os.path.exists(os.path.join(root, "architecture.json")), "live": bool(live)}


def in_repo(root):
    return os.path.dirname(os.path.abspath(root)) == os.path.join(REPO, "workspaces")


def managed(root):
    """True if this repo's harness can run the project: one of its workspaces, or an outside project that has been
    set up for it (workspace.json with a design, and usually a project.json saying how to train and score it)."""
    if (read_json(os.path.join(root, "architecture.json")) or {}).get("history_only"):
        return False
    design = (read_json(os.path.join(root, "workspace.json")) or {}).get("design")
    return os.path.isdir(HARNESS) and (in_repo(root) or design in DESIGNS)


def tail(path, n=60):
    try:
        with open(path, "rb") as f:
            f.seek(max(0, os.path.getsize(path) - 16384))
            return f.read().decode("utf-8", "replace").splitlines()[-n:]
    except OSError:
        return []


def trace_summary(path):
    m = TRACE_RE.match(os.path.basename(path))
    d = read_json(path)
    if not m or not isinstance(d, dict):
        return None
    reply = d.get("reply") or ""
    title = None
    found = re.search(r'"(?:title|action)"\s*:\s*"([^"]{1,120})"', reply)
    if found:
        title = found.group(1)
    usage = d.get("usage") or {}
    return {"file": os.path.basename(path), "exp": int(m.group(1)), "step": m.group(2), "role": m.group(3),
            "ts": int(m.group(4)), "model": d.get("model"), "seconds": d.get("seconds"),
            "tokens": usage.get("total_tokens") if isinstance(usage, dict) else None, "error": d.get("error"),
            "title": title, "preview": re.sub(r"\s+", " ", reply)[:220], "has_reasoning": bool(d.get("reasoning"))}


def agents_state(root):
    """Design, roles and models, the live status, the loop log, and the most recent LLM calls."""
    design = (read_json(os.path.join(root, "workspace.json")) or {}).get("design")
    models = (read_json(os.path.join(HARNESS, "config.json")) or {}).get("models", {})
    traces_dir = os.path.join(root, "traces")
    try:
        files = sorted((f for f in os.listdir(traces_dir) if TRACE_RE.match(f)),
                       key=lambda f: int(TRACE_RE.match(f).group(4)), reverse=True)[:120]
    except OSError:
        files = []
    calls = [c for c in (trace_summary(os.path.join(traces_dir, f)) for f in files) if c]
    status = read_json(os.path.join(traces_dir, "status.json"))
    proc = RUNNERS.get(os.path.abspath(root))
    designs = []
    for key, d in DESIGNS.items():
        ws = os.path.join(REPO, "workspaces", key)
        roles = []
        for role, about in d["roles"]:
            prompt = os.path.join(PROMPTS, role + ".md")
            tiers = (read_json(os.path.join(HARNESS, "config.json")) or {}).get("subagent_models") or {}
            model = models.get(role) or (" or ".join(m.split("/")[-1] for m in tiers.values()) if role == "subagent" and tiers
                                         else "default (CRUSOE_MODEL)")
            roles.append({"role": role, "about": about, "model": model,
                          "prompt": role if os.path.exists(prompt) else None})
        designs.append({"id": key, "name": d["name"], "about": d["about"], "roles": roles,
                        "workspace": ws, "exists": os.path.isdir(ws),
                        "project": project_id(ws) if os.path.isdir(ws) else None})
    return {"research": research.state(root), "training": training.state(root), "execution_note": ("Use Start training above to reproduce the current code on a configured machine. "
                               "The autonomous research loop below is not configured for this multi-file trainer."
                               if (read_json(os.path.join(root, "architecture.json")) or {}).get("history_only")
                               else (read_json(os.path.join(root, "architecture.json")) or {}).get("execution_note")),
            "managed": managed(root), "design": design, "designs": designs, "status": status,
            "server_time": time.time(), "running": bool(proc and proc.poll() is None),
            "stop_requested": os.path.exists(STOP_FILE), "calls": calls,
            "log": tail(os.path.join(traces_dir, "loop.log"))}


def start_loop(root, design, until=None, max_iters=0):
    """Start harness/run.py for a design. A design runs in its own workspace, so another design starts a new one
    (in this repo's workspaces/; an outside project runs only the design it was set up with)."""
    own = (read_json(os.path.join(root, "workspace.json")) or {}).get("design")
    if own != design and not in_repo(root):
        raise ValueError(f"this project is set up for the {own} design; clone it again to run {design}")
    ws = os.path.abspath(root) if own == design else os.path.join(REPO, "workspaces", design)
    if RUNNERS.get(ws) and RUNNERS[ws].poll() is None:
        raise ValueError("already running")
    if research.state(ws)["active"]:
        raise ValueError("Stop this project's agentic research before starting the legacy loop.")
    if any(job["status"] in training.ACTIVE for job in training.state(ws)["jobs"]):
        raise ValueError("Stop this project's standalone training job before starting the research loop.")
    cmd = [sys.executable, os.path.join(HARNESS, "run.py"), "--design", design, "--workspace", ws]
    if until:
        if not re.match(r"^\d{1,2}:\d{2}$", until):
            raise ValueError("until must be HH:MM")
        cmd += ["--until", until]
    if max_iters:
        cmd += ["--max-iters", str(int(max_iters))]
    log = open(os.path.join(HARNESS, f"runner-{design}.log"), "a")
    RUNNERS[ws] = subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    return ws


class Handler(BaseHTTPRequestHandler):
    root = "."
    server_version = "autoresearch-webui/1"

    def log_message(self, fmt, *args):
        pass  # quiet: the UI polls every few seconds

    def send(self, code, body, ctype, cache="no-cache"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def not_found(self):
        self.send(404, b"not found\n", "text/plain; charset=utf-8")

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        try:
            if path in STATIC:
                return self.serve_file(os.path.join(HERE, STATIC[path]), "no-store")
            parts = path.strip("/").split("/")
            if path == "/api/projects":
                cards = [project_card(pid, e) for pid, e in projects().items()]
                cards.sort(key=lambda c: -(c["opened_at"] or c["updated_at"] or 0))
                return self.json(cards)
            if len(parts) == 3 and parts[:2] == ["api", "p"]:
                entry = projects().get(parts[2])
                if not entry:
                    return self.not_found()
                if "open=1" in (urlparse(self.path).query or ""):
                    register(entry["path"], opened=True)
                return self.json(build_state(entry["path"], f"/p/{parts[2]}"))
            if len(parts) == 4 and parts[:2] == ["api", "p"] and parts[3] == "agents":
                entry = projects().get(parts[2])
                return self.json(agents_state(entry["path"])) if entry else self.not_found()
            if len(parts) == 5 and parts[:2] == ["api", "p"] and parts[3] == "training-log":
                entry = projects().get(parts[2])
                if not entry or not re.fullmatch(r"\d{8}-\d{6}-[0-9a-f]{8}", parts[4]):
                    return self.not_found()
                return self.serve_file(os.path.join(entry["path"], "training", "runs", parts[4], "run.log"), "no-store")
            if len(parts) == 5 and parts[:2] == ["api", "p"] and parts[3] == "trace" and TRACE_RE.match(parts[4]):
                entry = projects().get(parts[2])
                d = read_json(os.path.join(entry["path"], "traces", parts[4])) if entry else None
                return self.json(d) if isinstance(d, dict) else self.not_found()
            if len(parts) == 3 and parts[:2] == ["api", "prompts"] and re.match(r"^[a-z_]+$", parts[2]):
                try:
                    with open(os.path.join(PROMPTS, parts[2] + ".md")) as f:
                        return self.json({"role": parts[2], "text": f.read()})
                except OSError:
                    return self.not_found()
            if len(parts) == 4 and parts[0] == "p" and parts[2] == "diff" and HASH_RE.match(parts[3]):
                entry = projects().get(parts[1])
                text = git_diff(entry["path"], parts[3], stats_only=False) if entry else None
                if text is None:
                    return self.not_found()
                return self.send(200, text.encode(), "text/plain; charset=utf-8", "max-age=3600")
            if len(parts) == 5 and parts[0] == "p" and parts[2] == "runs" and RUN_ID_RE.match(parts[3]) \
                    and parts[4] in RUN_FILES:
                entry = projects().get(parts[1])
                if not entry:
                    return self.not_found()
                # URLs carry ?v=<mtime>, so a changed file gets a new URL and these can be cached.
                return self.serve_file(os.path.join(entry["path"], "runs", parts[3], parts[4]), "max-age=3600")
            if path == "/api/annotations":
                if not os.path.exists(ANNOTATIONS):
                    return self.send(200, b"[]", "application/json; charset=utf-8", "no-store")
                return self.serve_file(ANNOTATIONS, "no-store")
            if path == "/favicon.ico":
                return self.send(204, b"", "image/x-icon")
            self.not_found()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def json(self, obj, code=200):
        self.send(code, json.dumps(obj, allow_nan=False).encode(), "application/json; charset=utf-8", "no-store")

    def do_POST(self):
        """POST /api/projects {"path"} adds a project. POST /api/annotations replaces the notes (a JSON array)."""
        path = urlparse(self.path).path
        parts = path.strip("/").split("/")
        action = parts[3] if len(parts) == 4 and parts[:2] == ["api", "p"] else None
        prompt = parts[2] if len(parts) == 3 and parts[:2] == ["api", "prompts"] else None
        if path not in ("/api/annotations", "/api/projects") and action not in ("run", "stop", "training-save", "training-check", "training-start", "training-stop", "research-start", "research-stop") and \
                not (prompt and os.path.exists(os.path.join(PROMPTS, prompt + ".md"))):
            return self.not_found()
        if action and action.startswith(("training-", "research-")):
            # Execution endpoints are only available from the local UI, with same-origin JSON requests.
            host = self.headers.get("Host", "")
            origin = self.headers.get("Origin")
            hostname = urlparse("http://" + host).hostname
            if self.client_address[0] not in ("127.0.0.1", "::1") or hostname not in ("localhost", "127.0.0.1", "::1") \
                    or (origin and urlparse(origin).netloc != host) \
                    or self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
                return self.json({"error": "Execution controls require a same-origin request to the local Turing UI."}, 403)
        length = int(self.headers.get("Content-Length") or 0)
        if not 0 < length <= 1 << 20:
            return self.send(413, b"bad length\n", "text/plain; charset=utf-8")
        try:
            notes = json.loads(self.rfile.read(length))
        except ValueError:
            notes = None
        if prompt:
            text = (notes or {}).get("text") if isinstance(notes, dict) else None
            if not isinstance(text, str) or not text.strip():
                return self.json({"error": "empty prompt"}, 400)
            with open(os.path.join(PROMPTS, prompt + ".md"), "w") as f:
                f.write(text)
            return self.json({"ok": True})
        if action and action.startswith(("training-", "research-")):
            entry = projects().get(parts[2])
            if not entry or not isinstance(notes, dict):
                return self.json({"error": "Project not found or invalid settings."}, 400)
            root = entry["path"]
            try:
                if action == "research-start":
                    with LAUNCH_LOCK:
                        if RUNNERS.get(root) and RUNNERS[root].poll() is None:
                            raise ValueError("Stop this project's legacy research loop first.")
                        result = research.start(root, notes)
                elif action == "research-stop":
                    result = research.stop(root, notes.get("run_id"))
                elif action == "training-save":
                    result = {"config": training.save(root, notes.get("config"))}
                elif action == "training-check":
                    result = training.check(notes.get("config"))
                elif action == "training-start":
                    with LAUNCH_LOCK:
                        if (RUNNERS.get(root) and RUNNERS[root].poll() is None) or research.state(root)["active"]:
                            raise ValueError("Stop this project's research loop before launching a standalone training run.")
                        result = training.start(root, notes.get("config"))
                else:
                    result = training.stop(root, notes.get("job_id"))
                return self.json(result)
            except (ValueError, OSError) as e:
                return self.json({"error": str(e)}, 400)
        if action:
            entry = projects().get(parts[2])
            if not entry or not managed(entry["path"]):
                return self.json({"error": "this project isn't run by this repo's harness"}, 400)
            body = notes if isinstance(notes, dict) else {}
            if action == "stop":
                with open(STOP_FILE, "w") as f:
                    f.write("stop requested from the UI\n")
                proc = RUNNERS.get(entry["path"])
                if body.get("now") and proc and proc.poll() is None:
                    proc.terminate()
                return self.json({"ok": True})
            if body.get("design") not in DESIGNS:
                return self.json({"error": "design must be fixed or agentic"}, 400)
            try:
                with LAUNCH_LOCK:
                    ws = start_loop(entry["path"], body["design"], body.get("until"), body.get("max_iters") or 0)
            except ValueError as e:
                return self.json({"error": str(e)}, 400)
            register(ws, opened=True)
            return self.json({"id": project_id(ws)})
        if path == "/api/projects":
            target = os.path.abspath(os.path.expanduser(str((notes or {}).get("path") or "")))
            if not isinstance(notes, dict) or not is_project(target):
                return self.json({"error": "not a project directory (needs train.py, results.tsv or runs/)"}, 400)
            register(target, opened=True)
            return self.json({"id": project_id(target)})
        if not isinstance(notes, list):
            return self.send(400, b"expected a JSON array\n", "text/plain; charset=utf-8")
        tmp = ANNOTATIONS + ".tmp"
        with open(tmp, "w") as f:
            json.dump(notes, f, indent=2)
            f.write("\n")
        os.replace(tmp, ANNOTATIONS)
        self.send(200, b'{"ok": true}', "application/json; charset=utf-8", "no-store")

    def serve_file(self, path, cache):
        try:
            with open(path, "rb") as f:
                body = f.read()
        except OSError:
            return self.not_found()
        ctype = {".py": "text/plain", ".diff": "text/plain", ".js": "text/javascript", ".json": "application/json"}.get(
            os.path.splitext(path)[1]) or mimetypes.guess_type(path)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype == "application/json":
            ctype += "; charset=utf-8"
        self.send(200, body, ctype, cache)


def main():
    ap = argparse.ArgumentParser(description="Serve the autoresearch presentation UI.")
    ap.add_argument("--root", action="append", default=[],
                    help="a project directory to add to the recent list (repeatable); workspaces/* are added too")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to reach it from another device")
    args = ap.parse_args()

    found = []
    ws = os.path.join(os.path.dirname(HERE), "workspaces")
    if os.path.isdir(ws):
        found = [os.path.join(ws, d) for d in sorted(os.listdir(ws)) if is_project(os.path.join(ws, d))]
    for root in found + args.root:
        if not is_project(os.path.abspath(root)):
            raise SystemExit(f"--root {root} is not a project directory (needs train.py, results.tsv or runs/)")
        register(root)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.daemon_threads = True
    shown = "localhost" if args.host in ("127.0.0.1", "0.0.0.0") else args.host
    print(f"Projects: {', '.join(p['path'] for p in load_registry()) or 'none yet'}\n"
          f"  http://{shown}:{args.port}  (Ctrl-C to stop)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
