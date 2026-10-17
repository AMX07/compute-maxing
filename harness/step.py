"""
One iteration of the research loop: plan -> apply -> run -> decide -> reflect -> log.

    python3 -m harness.step          # one experiment (the first call runs the unmodified baseline)

Every iteration is a fresh process and all state lives in files, so a crash costs one iteration and edits to the
harness take effect on the next one:
    harness/memory/state.json     best commit and val_loss, history of ideas with predictions and outcomes
    harness/memory/playbook.json  lessons learned (rendered to playbook.md), maintained by the reflector
    results.tsv, runs/<run_id>/   the shared data contract (idea.json and change.diff are written here)
Keep/discard is decided by this code from val_loss, never by the model. Git history is forward-only: a discarded
experiment is reverted with a new commit, so nothing is lost and other uncommitted work is never touched.
"""

import ast
import builtins
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time

from harness import llm
from harness import workspace as wsp

try:  # the idea bank is optional: a broken researcher must never stop an experiment
    from harness import researcher
except Exception:  # noqa: BLE001
    researcher = None

ROOT = llm.ROOT
WS = wsp.WS                      # this design's workspace: train.py, results.tsv, runs/, memory/, traces/
MEM = wsp.MEM
STATE = os.path.join(MEM, "state.json")
PLAYBOOK = os.path.join(MEM, "playbook.json")
PROMPTS = os.path.join(ROOT, "harness", "prompts")
TRAIN = os.path.join(WS, "train.py")
RESULTS = os.path.join(WS, "results.tsv")
RUN_LOG = os.path.join(WS, "run.log")
RUNS = os.path.join(WS, "runs")
TRAIN_REL, MEM_REL = wsp.rel(TRAIN), wsp.rel(MEM)   # for git
HEADER = "commit\tval_loss\tmemory_gb\tstatus\trun_id\tdescription\n"

FROZEN = ["prepare.py", "inference.py"]
RUN_TIMEOUT = 6 * 60             # kill a run after this long (program.md)
TRAIN_SLACK = 100                # seconds allowed on top of TIME_BUDGET for startup, compile steps and eval
SAMPLE_LIMIT = 120               # seconds allowed for the 16 benchmark images
MIN_CAPTION_DROPOUT = 0.1
MAX_LESSONS = 25
STUCK_AFTER = 4                  # discards in a row before the planner is told to switch category


def log(msg):
    print(f"[harness {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def set_status(phase, n, title=None, **extra):
    """harness/traces/status.json tells the web UI what the loop is doing right now."""
    try:
        os.makedirs(wsp.TRACES, exist_ok=True)
        save_json(os.path.join(wsp.TRACES, "status.json"),
                  {"phase": phase, "n": n, "title": title, "since": time.time(), **extra})
    except OSError:
        pass


def git(*args, check=True):
    for _ in range(20):  # another process (the researcher, a human) may hold the index lock for a moment
        r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
        if r.returncode == 0 or "index.lock" not in r.stderr:
            break
        time.sleep(0.5)
    if check and r.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout.strip()


def commit(message, paths):
    git("add", "--", *paths)
    if not git("diff", "--cached", "--name-only", "--", *paths):
        return git("rev-parse", "HEAD")
    git("commit", "-q", "-m", message, "--", *paths)
    return git("rev-parse", "HEAD")


def secret_values():
    """Values that must never leave this machine: the API keys from .env/the environment."""
    vals = []
    try:
        vals.append(llm.config()["CRUSOE_API_KEY"])
    except SystemExit:
        pass
    llm.load_env()
    vals += [os.environ.get(k, "") for k in ("YDC_API_KEY",)]
    return [v for v in vals if len(v) >= 8]


def push_safely():
    """Push main to origin so GitHub shows the experiment history live. Refuses if the outgoing commits touch .env
    or any git-ignored path, or contain a secret value. Never raises and never force-pushes."""
    try:
        if not git("remote", check=False):
            return
        upstream = git("rev-parse", "--verify", "-q", "origin/main", check=False) or None
        rng = f"{upstream}..HEAD" if upstream else "HEAD"
        files = [f for f in git("log", rng, "--name-only", "--format=", check=False).splitlines() if f.strip()]
        if not files and upstream:
            return  # nothing new
        bad = [f for f in set(files) if os.path.basename(f) == ".env" or
               subprocess.run(["git", "check-ignore", "--no-index", "-q", f], cwd=ROOT).returncode == 0]
        if bad:
            log(f"push refused: outgoing commits touch ignored/secret paths: {sorted(bad)}")
            return
        patch = git("log", rng, "-p", "--format=", check=False)
        if any(v in patch for v in secret_values()):
            log("push refused: outgoing commits contain a secret value")
            return
        r = subprocess.run(["git", "push", "-q", "origin", "main"], cwd=ROOT, capture_output=True, text=True, timeout=90)
        log("pushed to origin/main" if r.returncode == 0 else f"push failed (will retry next experiment): {r.stderr.strip()[:200]}")
    except Exception as e:  # noqa: BLE001  a push problem must never cost an experiment
        log(f"push skipped: {e}")


def sha(path):
    with open(os.path.join(WS, path), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def read(path):
    with open(path) as f:
        return f.read()


def write(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def load_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write(path, json.dumps(obj, indent=2) + "\n")


def time_budget():
    m = re.search(r"^TIME_BUDGET\s*=\s*(\d+)", read(os.path.join(WS, "prepare.py")), re.M)
    return int(m.group(1)) if m else 120


# ---------------------------------------------------------------------------
# results.tsv
# ---------------------------------------------------------------------------

def read_results():
    if not os.path.exists(RESULTS):
        return []
    rows = []
    for line in read(RESULTS).splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) >= 6:
            rows.append(dict(zip(["commit", "val_loss", "memory_gb", "status", "run_id", "description"], parts)))
    return rows


def append_result(commit_hash, val_loss, memory_gb, status, run_id, description):
    if not os.path.exists(RESULTS):
        write(RESULTS, HEADER)
    desc = re.sub(r"\s+", " ", description).strip()[:120]
    with open(RESULTS, "a") as f:
        f.write(f"{commit_hash[:7]}\t{val_loss or 0:.6f}\t{memory_gb or 0:.1f}\t{status}\t{run_id or '-'}\t{desc}\n")


# ---------------------------------------------------------------------------
# State and playbook
# ---------------------------------------------------------------------------

def load_state():
    state = load_json(STATE, None)
    if state:
        return state
    rows = read_results()
    kept = [float(r["val_loss"]) for r in rows if r["status"] == "keep" and float(r["val_loss"]) > 0]
    return {"best_commit": git("rev-parse", "HEAD"), "best_val_loss": min(kept) if kept else None,
            "consecutive_discards": 0, "frozen": {p: sha(p) for p in FROZEN}, "history": []}


def render_playbook(lessons):
    if not lessons:
        return "(empty: no lessons yet)"
    return "\n".join(f"- [{l['id']}] {l['text']}" for l in lessons)


def apply_playbook_ops(ops, exp_n):
    book = load_json(PLAYBOOK, {"next_id": 1, "lessons": []})
    by_id = {l["id"]: l for l in book["lessons"]}
    for op in (ops or [])[:3]:
        kind, text = op.get("op"), str(op.get("text", "")).strip()[:300]
        if kind == "add" and text:
            book["lessons"].append({"id": f"L{book['next_id']}", "text": text, "since": exp_n, "updated": exp_n})
            book["next_id"] += 1
        elif kind == "update" and op.get("id") in by_id and text:
            by_id[op["id"]].update(text=text, updated=exp_n)
        elif kind == "remove" and op.get("id") in by_id:
            book["lessons"] = [l for l in book["lessons"] if l["id"] != op["id"]]
    # Bounded memory: drop the lessons that have gone longest without being confirmed or revised.
    book["lessons"].sort(key=lambda l: l["updated"], reverse=True)
    book["lessons"] = sorted(book["lessons"][:MAX_LESSONS], key=lambda l: int(l["id"][1:]))
    save_json(PLAYBOOK, book)
    write(os.path.join(MEM, "playbook.md"), "# Playbook\n\nLessons the loop has learned, maintained by the reflector.\n\n"
          + render_playbook(book["lessons"]) + "\n")
    return book["lessons"]


# ---------------------------------------------------------------------------
# Guards: the evaluator lives outside the loop
# ---------------------------------------------------------------------------

def check_frozen(state):
    changed = [p for p, h in state["frozen"].items() if sha(p) != h]
    if changed:
        raise SystemExit(f"Refusing to run: frozen file(s) changed since the loop started: {', '.join(changed)}. "
                         "Restore them, or delete harness/memory/state.json to accept the new versions deliberately.")


MLX_CACHE = os.path.join(ROOT, "harness", "traces", "mlx_api_cache.json")
EXTRA_GLOBALS = {"__file__", "__name__", "__doc__", "__spec__", "__builtins__", "__annotations__"}


def undefined_names(tree):
    """Names that are read somewhere but never bound anywhere in the file (a missing import, a typo).
    Deliberately loose about scope, so it never flags valid code; it only catches names that exist nowhere."""
    if any(isinstance(n, ast.ImportFrom) and any(a.name == "*" for a in n.names) for n in ast.walk(tree)):
        return []
    bound = set(dir(builtins)) | EXTRA_GLOBALS
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            bound |= {(a.asname or a.name).split(".")[0] for a in node.names}
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bound.update(node.names)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            bound.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            bound.add(node.rest)
    used = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id not in bound:
            used.setdefault(node.id, node.lineno)
    return sorted(used.items(), key=lambda kv: kv[1])


def mlx_chains(tree):
    """Dotted MLX paths the code relies on, e.g. 'mlx.core.fast.scaled_dot_product_attention'."""
    aliases, chains = {}, set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] == "mlx":
                    aliases[a.asname or "mlx"] = a.name if a.asname else "mlx"
        elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "mlx":
            chains |= {f"{node.module}.{a.name}" for a in node.names if a.name != "*"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            parts, cur = [], node
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name) and cur.id in aliases:
                chains.add(".".join([aliases[cur.id]] + parts[::-1]))
    return chains


def missing_mlx_apis(tree):
    """MLX functions/classes used by train.py that don't exist in the installed MLX (a hallucinated API).
    Resolved once per path in the project's venv and cached; if MLX can't be inspected, nothing is flagged."""
    chains = mlx_chains(tree)
    cache = load_json(MLX_CACHE, {})
    unknown = sorted(c for c in chains if c not in cache)
    python = os.path.join(ROOT, ".venv", "bin", "python")
    if unknown and os.path.exists(python):
        probe = ("import importlib, json, sys\nout = {}\nfor chain in json.loads(sys.argv[1]):\n"
                 "    parts, obj, rest = chain.split('.'), None, []\n"
                 "    for i in range(len(parts), 0, -1):\n"
                 "        try:\n            obj, rest = importlib.import_module('.'.join(parts[:i])), parts[i:]\n            break\n"
                 "        except Exception:\n            continue\n"
                 "    ok = obj is not None\n    for p in rest:\n        if not hasattr(obj, p):\n            ok = False\n            break\n"
                 "        obj = getattr(obj, p)\n    out[chain] = ok\nprint(json.dumps(out))\n")
        try:
            r = subprocess.run([python, "-c", probe, json.dumps(unknown)], cwd=ROOT, capture_output=True, text=True, timeout=60)
            cache.update(json.loads(r.stdout.strip().splitlines()[-1]))
            save_json(MLX_CACHE, cache)
        except Exception as e:  # noqa: BLE001  can't inspect MLX: don't block the experiment
            log(f"MLX API check skipped: {e}")
            return []
    return sorted(c for c in chains if cache.get(c) is False)


def validate(src):
    """Problems that disqualify an edited train.py before it runs."""
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [f"SyntaxError at line {e.lineno}: {e.msg}"]
    problems = []
    for name, line in undefined_names(tree):
        problems.append(f"`{name}` is used (line {line}) but never defined or imported")
    for chain in missing_mlx_apis(tree):
        problems.append(f"`{chain}` does not exist in the installed MLX")
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in ("prepare", "inference"):
            imported |= {a.asname or a.name for a in node.names}
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in imported:
            problems.append(f"redefines `{node.name}`, which is imported from prepare/inference")
        for t in targets:
            for n in ast.walk(t):
                if isinstance(n, ast.Name) and n.id in imported:
                    problems.append(f"reassigns `{n.id}`, which is imported from prepare/inference")
                if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in ("prepare", "inference"):
                    problems.append(f"assigns to `{n.value.id}.{n.attr}`")
        if isinstance(node, ast.Call):
            for a in list(node.args) + [k.value for k in node.keywords]:
                if isinstance(a, ast.Constant) and a.value in ("test", "val"):
                    problems.append("reads the held-out split, which only the evaluation may use")
    m = re.search(r"^CAPTION_DROPOUT\s*=\s*([0-9.eE+-]+)", src, re.M)
    if not m or float(m.group(1)) < MIN_CAPTION_DROPOUT:
        problems.append(f"CAPTION_DROPOUT must stay >= {MIN_CAPTION_DROPOUT} (lowering it games val_loss and breaks guidance)")
    if "evaluate_loss(model" not in src or "save_run(" not in src:
        problems.append("the final evaluate_loss(model) and save_run(...) calls must stay")
    return sorted(set(problems))


def apply_edits(src, edits):
    """Apply SEARCH/REPLACE blocks. Each SEARCH must match exactly once (trailing whitespace is ignored)."""
    if not edits:
        raise ValueError("no SEARCH/REPLACE blocks found in the reply")
    for i, (search, replace) in enumerate(edits, 1):
        if not search.strip():
            raise ValueError(f"edit {i}: empty SEARCH block")
        count = src.count(search)
        if count == 1:
            src = src.replace(search, replace, 1)
            continue
        if count > 1:
            raise ValueError(f"edit {i}: SEARCH matches {count} places; include more surrounding lines")
        # Fallback: match line by line ignoring trailing whitespace.
        lines, want = src.split("\n"), [l.rstrip() for l in search.split("\n")]
        hits = [j for j in range(len(lines) - len(want) + 1) if [l.rstrip() for l in lines[j:j + len(want)]] == want]
        if len(hits) != 1:
            first = search.strip().splitlines()[0][:80]
            raise ValueError(f"edit {i}: SEARCH text not found in train.py (starts with: {first!r})")
        j = hits[0]
        src = "\n".join(lines[:j] + replace.split("\n") + lines[j + len(want):])
    return src


# ---------------------------------------------------------------------------
# Running an experiment
# ---------------------------------------------------------------------------

def birthtime(path):
    st = os.stat(path)
    return getattr(st, "st_birthtime", st.st_mtime)


FAKE_TRAIN = os.environ.get("HARNESS_FAKE_TRAIN") == "1"


class FakeRun:
    """No-GPU stand-in for tests (HARNESS_FAKE_TRAIN=1). Writes a run folder and run.log the way train.py does,
    with a val_loss derived from the code, so every other step of the pipeline runs for real."""
    def __init__(self, out):
        src = read(TRAIN)
        try:
            ast.parse(src)
        except SyntaxError:
            out.write("SyntaxError (fake run)\n")
            self.returncode = 1
            return
        h = int(hashlib.sha256(src.encode()).hexdigest()[:8], 16)
        base = (load_json(STATE, {}) or {}).get("best_val_loss") or 0.1838
        val = round(base * (0.985 + (h % 400) / 10000), 6)        # between -1.5% and +2.5% of the best
        run_id = time.strftime("%Y%m%d-%H%M%S")
        run_dir = os.path.join(RUNS, run_id)
        while os.path.exists(run_dir):
            run_dir += "x"
        os.makedirs(run_dir)
        metrics = {"run_id": os.path.basename(run_dir), "commit": git("rev-parse", "--short=7", "HEAD", check=False),
                   "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "val_loss": val, "training_seconds": 0.0,
                   "total_seconds": 0.0, "peak_memory_gb": 0.0, "num_steps": 0, "num_params_M": 0.0, "fake": True}
        save_json(os.path.join(run_dir, "metrics.json"), metrics)
        write(os.path.join(run_dir, "train.py"), src)
        out.write(f"FAKE RUN (no GPU)\n---\nval_loss:         {val:.6f}\npeak_memory_gb:   0.0\nnum_steps:        0\n"
                  f"run_dir:          {run_dir}\n")
        self.returncode = 0


def run_training():
    log("training (uv run train.py > run.log)" + (" [FAKE, no GPU]" if FAKE_TRAIN else ""))
    start = time.time()
    timed_out = False
    with open(RUN_LOG, "w") as out:
        if FAKE_TRAIN:
            proc = FakeRun(out)
        else:
            proc = subprocess.Popen(["uv", "run", "train.py"], cwd=WS, stdout=out, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            try:
                proc.wait(timeout=RUN_TIMEOUT)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
    text = read(RUN_LOG) if os.path.exists(RUN_LOG) else ""

    def field(name):
        m = re.search(rf"^{name}:\s*(\S+)", text, re.M)
        return m.group(1).replace(",", "") if m else None

    run_dir = field("run_dir")
    if not run_dir or not os.path.isdir(run_dir):
        fresh = [os.path.join(RUNS, d) for d in os.listdir(RUNS)] if os.path.isdir(RUNS) else []
        fresh = [d for d in fresh if os.path.isdir(d) and birthtime(d) >= start - 1]
        run_dir = max(fresh, key=birthtime) if fresh else None
    metrics = load_json(os.path.join(run_dir, "metrics.json"), {}) if run_dir else {}
    val_loss = field("val_loss") or metrics.get("val_loss")
    res = {
        "returncode": proc.returncode, "timed_out": timed_out, "seconds": round(time.time() - start, 1),
        "run_dir": run_dir, "run_id": os.path.basename(run_dir) if run_dir else None, "metrics": metrics,
        "val_loss": float(val_loss) if val_loss else None,
        "memory_gb": float(field("peak_memory_gb") or metrics.get("peak_memory_gb") or 0),
        "train_wall": round(birthtime(run_dir) - start, 1) if run_dir else None,
        "tail": "\n".join(re.split(r"[\r\n]", text)[-60:])[-6000:],
    }
    log(f"run finished in {res['seconds']}s: val_loss={res['val_loss']} rc={proc.returncode}"
        f"{' TIMEOUT' if timed_out else ''} run_dir={res['run_id']}")
    return res


def violations(res, budget):
    out = []
    if res["train_wall"] and res["train_wall"] > budget + TRAIN_SLACK:
        out.append(f"training took {res['train_wall']:.0f}s of wall clock, over the {budget}s budget + {TRAIN_SLACK}s slack")
    ss = res["metrics"].get("sample_seconds")
    if ss is not None and ss > SAMPLE_LIMIT:
        out.append(f"sampling took {ss}s, over the {SAMPLE_LIMIT}s limit")
    return out


# ---------------------------------------------------------------------------
# LLM roles
# ---------------------------------------------------------------------------

def experiment_log(state):
    lines = ["#  | status  | val_loss | vs best before | predicted | category | description"]
    hist = {h["n"]: h for h in state["history"]}
    for i, r in enumerate(read_results()):
        h = hist.get(i, {})
        actual = f"{h['actual_pct']:+.2f}%" if h.get("actual_pct") is not None else "-"
        pred = f"{h['prediction_pct']:+.2f}%" if isinstance(h.get("prediction_pct"), (int, float)) else "-"
        lines.append(f"{i:<2} | {r['status']:<7} | {r['val_loss']} | {actual} | {pred} | {h.get('category', '-')} | {r['description']}")
    return "\n".join(lines)


def idea_bank_text(limit=6):
    """Untried cards from the background researcher (harness/memory/ideas.jsonl), or None."""
    try:
        ideas = researcher.untried()[:limit] if researcher else []
    except Exception as e:  # noqa: BLE001
        log(f"idea bank unreadable: {e}")
        return None
    if not ideas:
        return None
    lines = [f"- {c.get('id')} [{c.get('category')}, mlx_risk {c.get('mlx_risk')}] {c.get('title')}: {c.get('change')} "
             f"Why: {c.get('mechanism')} Sources: {', '.join(c.get('sources') or []) or 'model knowledge only (weaker)'}"
             for c in ideas]
    return ("## Idea bank (from a background literature search; suggestions only, check them against train.py)\n\n"
            + "\n".join(lines) + '\n\nYou may implement one of these (set "idea_id" in your JSON) or propose your own '
            '("idea_id": null).')


def mark_idea(meta, n, status, val_loss):
    iid = meta.get("idea_id")
    if not iid or not researcher:
        return
    try:
        if iid in {c.get("id") for c in researcher.load_bank()}:
            researcher.mark(iid, status="tried", exp=n, outcome=status, val_loss=val_loss)
    except Exception as e:  # noqa: BLE001
        log(f"could not mark idea {iid}: {e}")


def planner_messages(state, src, lessons, n):
    parts = [
        f"## Research brief (program.md, written for coding agents; the harness now does the git/run/log steps)\n\n"
        f"{read(os.path.join(WS, 'program.md'))}",
        f"## Playbook: lessons so far\n\n{render_playbook(lessons)}",
        idea_bank_text(),
        f"## Experiment log (results.tsv; 'vs best before' is the change against the best kept val_loss at that time)\n\n"
        f"{experiment_log(state)}\n\nCurrent best val_loss: {state['best_val_loss']}",
    ]
    recent = state["history"][-STUCK_AFTER:]
    if state["consecutive_discards"] >= STUCK_AFTER:
        cats = sorted({h.get("category", "?") for h in recent})
        parts.append(f"## Note\n\nThe last {state['consecutive_discards']} experiments were all discarded or crashed "
                     f"(categories: {', '.join(cats)}). Pick a different category or a bolder idea this time.")
    parts = [p for p in parts if p]
    parts.append(f"## Current train.py (the best version so far)\n\n```python\n{src}\n```")
    parts.append(f"Propose experiment #{n}. Follow the reply format exactly.")
    return [{"role": "system", "content": read(os.path.join(PROMPTS, "planner.md"))},
            {"role": "user", "content": "\n\n".join(parts)}]


def propose(state, src, lessons, n):
    """Planner call. A reply is retried (up to 2 times) if the edit can't be applied or breaks a rule."""
    messages = planner_messages(state, src, lessons, n)
    for attempt in range(3):
        reply = llm.chat(messages, "planner", f"exp{n:03d}", temperature=0.7)
        meta = llm.extract_json(reply) or {}
        try:
            new_src = apply_edits(src, llm.extract_edits(reply))
            problems = validate(new_src)
            if new_src == src:
                problems.append("the edit does not change train.py")
        except ValueError as e:
            problems = [str(e)]
        if not meta.get("description"):
            problems.append("the JSON block with title/description/prediction is missing")
        if not meta.get("self_check"):
            problems.append('the JSON needs "self_check": concrete values your code produces at key points')
        if not problems:
            return meta, new_src
        log(f"planner reply rejected: {problems}")
        messages += [{"role": "assistant", "content": reply},
                     {"role": "user", "content": "Your reply could not be used:\n- " + "\n- ".join(problems)
                      + "\n\nReply again in the same format, against the same current train.py."}]
    return None, None


def fix(src, meta, res, n):
    messages = [{"role": "system", "content": read(os.path.join(PROMPTS, "fixer.md"))},
                {"role": "user", "content": f"## The idea\n\n{json.dumps(meta, indent=1)}\n\n## Crash output (end of run.log)\n\n"
                                            f"```\n{res['tail']}\n```\n\n## Current train.py (with the idea applied)\n\n```python\n{src}\n```"}]
    reply = llm.chat(messages, "fixer", f"exp{n:03d}", temperature=0.2)
    info = llm.extract_json(reply) or {}
    if info.get("give_up"):
        log(f"fixer gave up: {info.get('reason')}")
        return None, info
    try:
        new_src = apply_edits(src, llm.extract_edits(reply))
    except ValueError as e:
        log(f"fixer edit rejected: {e}")
        return None, info
    problems = validate(new_src)
    if problems:
        log(f"fixer edit rejected: {problems}")
        return None, info
    return new_src, info


def reflect(meta, outcome, lessons, n, diff=""):
    messages = [{"role": "system", "content": read(os.path.join(PROMPTS, "reflector.md"))},
                {"role": "user", "content": f"## Playbook\n\n{render_playbook(lessons)}\n\n## Experiment #{n}\n\n"
                                            f"Idea and prediction:\n{json.dumps(meta, indent=1)}\n\nThe code change that actually ran:\n"
                                            f"```diff\n{diff[:6000]}\n```\n\nOutcome:\n{json.dumps(outcome, indent=1)}"}]
    reply = llm.chat(messages, "reflector", f"exp{n:03d}", temperature=0.3, max_tokens=4096)
    return llm.extract_json(reply) or {}


# ---------------------------------------------------------------------------
# The iteration
# ---------------------------------------------------------------------------

def finish_artifacts(res, meta, outcome, parent_commit, exp_commit):
    if not res.get("run_dir"):
        return
    idea = {k: meta.get(k) for k in ("title", "category", "rationale", "prediction", "self_check", "sources", "idea_id",
                                     "agents", "actions") if meta.get(k)}
    idea["outcome"] = outcome
    save_json(os.path.join(res["run_dir"], "idea.json"), idea)
    diff = git("diff", parent_commit, exp_commit, "--", TRAIN_REL, check=False)
    if diff:
        write(os.path.join(res["run_dir"], "change.diff"), diff + "\n")


def run_baseline(state, budget):
    log("no results yet: running the unmodified train.py as the baseline")
    # A new workspace's files must be in git before anything can be restored from best_commit.
    files = [wsp.rel(os.path.join(WS, f)) for f in ("train.py", "prepare.py", "inference.py", "program.md", "workspace.json")
             if os.path.exists(os.path.join(WS, f))]
    commit(f"workspace {os.path.basename(WS)}: baseline files\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>", files)
    state["frozen"] = {p: sha(p) for p in FROZEN}
    set_status("training", 0, "Baseline")
    res = run_training()
    head = git("rev-parse", "HEAD")
    if res["val_loss"] is None:
        log("baseline crashed; fix train.py by hand before starting the loop. End of run.log:\n" + res["tail"][-2000:])
        return 1
    append_result(head, res["val_loss"], res["memory_gb"], "keep", res["run_id"], "baseline")
    meta = {"title": "Baseline", "category": "baseline", "rationale": "The unmodified train.py."}
    outcome = {"status": "keep", "val_loss": res["val_loss"], "num_steps": res["metrics"].get("num_steps"),
               "train_wall_s": res["train_wall"]}
    finish_artifacts(res, meta, outcome, head, head)
    state.update(best_commit=head, best_val_loss=res["val_loss"], consecutive_discards=0)
    state["history"].append({"n": 0, "title": "Baseline", "category": "baseline", "status": "keep",
                             "val_loss": res["val_loss"], "run_id": res["run_id"], "commit": head[:7]})
    save_json(STATE, state)
    apply_playbook_ops([{"op": "add", "text": f"Baseline: val_loss {res['val_loss']:.4f} in "
                         f"{res['metrics'].get('num_steps', '?')} steps at {res['metrics'].get('num_params_M', '?')}M params (exp 0)"}], 0)
    commit("harness: baseline recorded\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>", [MEM_REL])
    push_safely()
    log(f"baseline val_loss {res['val_loss']:.6f}")
    return 0


def execute(state, meta, new_src, n, fixer=None):
    """Shared by every design: commit the edit, train, optionally repair a crash, decide keep/discard in code,
    revert on discard, and log results.tsv, idea.json and change.diff. `fixer(new_src, res)` returns
    (fixed_src or None, note)."""
    budget = time_budget()
    desc = re.sub(r"\s+", " ", str(meta.get("description", meta.get("title", "experiment")))).strip()[:100]
    write(TRAIN, new_src)
    parent = state["best_commit"]
    exp_commit = commit(f"exp {n}: {desc}", [TRAIN_REL])

    set_status("training", n, meta.get("title"), prediction=meta.get("prediction"))
    res = run_training()
    fix_note = None
    if res["val_loss"] is None and fixer:
        log("crashed; asking the fixer for one repair")
        fixed_src, fix_note = fixer(new_src, res)
        if fixed_src:
            write(TRAIN, fixed_src)
            exp_commit = commit(f"exp {n} fix: {fix_note or 'repair crash'}"[:200], [TRAIN_REL])
            set_status("training", n, meta.get("title"), note="retry after fix")
            res = run_training()

    best_before = state["best_val_loss"]
    problems = violations(res, budget) if res["val_loss"] is not None else []
    if res["val_loss"] is None:
        status = "crash"
    elif problems:
        status = "discard"
    elif res["val_loss"] < best_before:
        status = "keep"
    else:
        status = "discard"
    actual_pct = (res["val_loss"] - best_before) / best_before * 100 if res["val_loss"] is not None else None
    log(f"decision: {status}" + (f" ({actual_pct:+.2f}% vs best)" if actual_pct is not None else "") +
        (f"; rule violations: {problems}" if problems else ""))

    if status == "keep":
        state.update(best_commit=exp_commit, best_val_loss=res["val_loss"], consecutive_discards=0)
    else:
        git("checkout", parent, "--", TRAIN_REL)
        commit(f"revert exp {n} ({status})", [TRAIN_REL])
        state["consecutive_discards"] += 1

    pred = meta.get("prediction") or {}
    outcome = {"status": status, "val_loss": res["val_loss"], "best_before": best_before,
               "delta_pct": round(actual_pct, 3) if actual_pct is not None else None,
               "predicted_delta_pct": pred.get("delta_pct"), "num_steps": res["metrics"].get("num_steps"),
               "num_params_M": res["metrics"].get("num_params_M"), "peak_memory_gb": res["memory_gb"],
               "train_wall_s": res["train_wall"], "sample_seconds": res["metrics"].get("sample_seconds"),
               "rule_violations": problems, "fix": fix_note}
    if status == "crash":
        outcome["error_tail"] = res["tail"][-1500:]
    finish_artifacts(res, meta, outcome, parent, exp_commit)
    append_result(exp_commit, res["val_loss"], res["memory_gb"], status, res["run_id"], desc)
    return {"res": res, "status": status, "outcome": outcome, "parent": parent, "exp_commit": exp_commit,
            "actual_pct": actual_pct}


def record(state, meta, ex, n, **extra):
    """Append to the history, save state, commit this design's memory, push."""
    res, status, outcome = ex["res"], ex["status"], ex["outcome"]
    pred = meta.get("prediction") or {}
    state["history"].append({"n": n, "title": meta.get("title"), "category": meta.get("category"), "status": status,
                             "idea_id": meta.get("idea_id"),
                             "val_loss": res["val_loss"], "prediction_pct": pred.get("delta_pct"),
                             "actual_pct": round(ex["actual_pct"], 3) if ex["actual_pct"] is not None else None,
                             "run_id": res["run_id"], "commit": ex["exp_commit"][:7],
                             "prediction_check": outcome.get("prediction_check"), **extra})
    state["last_outcome"] = {"n": n, "title": meta.get("title"), **{k: v for k, v in outcome.items() if v is not None}}
    save_json(STATE, state)
    commit(f"harness: exp {n} {status}" + (f" val_loss {res['val_loss']:.6f}" if res["val_loss"] else "")
           + "\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>", [MEM_REL])
    push_safely()
    set_status("idle", n + 1)


def begin():
    """Common start of an iteration: load state, check the evaluator, reset train.py to the best version.
    Returns (state, n, src), or runs the baseline and returns None."""
    os.makedirs(MEM, exist_ok=True)
    state = load_state()
    check_frozen(state)
    n = len(read_results())
    if n == 0 or state["best_val_loss"] is None:
        run_baseline(state, time_budget())
        return None
    git("checkout", state["best_commit"], "--", TRAIN_REL)
    return state, n, read(TRAIN)


def main():
    """The fixed design: planner -> guards -> run -> fixer on crash -> decide -> reflector -> playbook."""
    started = begin()
    if started is None:
        return 0 if read_results() else 1
    state, n, src = started
    lessons = load_json(PLAYBOOK, {"lessons": []})["lessons"]

    log(f"experiment #{n}: planning (best val_loss {state['best_val_loss']:.6f})")
    set_status("planning", n)
    meta, new_src = propose(state, src, lessons, n)
    if meta is None:
        log("planner failed 3 times; skipping this iteration")
        return 2
    log(f"idea: {meta.get('title')} [{meta.get('category')}] predicted {meta.get('prediction', {}).get('delta_pct')}%")

    def fixer(edited_src, res):
        try:
            fixed_src, info = fix(edited_src, meta, res, n)
        except RuntimeError as e:
            fixed_src, info = None, {"error": str(e)}
        return fixed_src, info.get("fix") or info.get("reason")

    ex = execute(state, meta, new_src, n, fixer=fixer)
    mark_idea(meta, n, ex["status"], ex["res"]["val_loss"])

    set_status("reflecting", n, meta.get("title"))
    try:
        diff = git("diff", ex["parent"], ex["exp_commit"], "--", TRAIN_REL, check=False)
        refl = reflect(meta, ex["outcome"], lessons, n, diff)
        apply_playbook_ops(refl.get("ops"), n)
        ex["outcome"]["prediction_check"] = refl.get("prediction_check")
        if ex["res"].get("run_dir"):
            finish_artifacts(ex["res"], meta, ex["outcome"], ex["parent"], ex["exp_commit"])
    except Exception as e:  # reflection is useful but never worth losing an experiment over
        log(f"reflector failed: {e}")

    record(state, meta, ex, n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
