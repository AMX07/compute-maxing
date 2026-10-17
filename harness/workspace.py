"""
Each agent design runs in its own workspace, so no task file, memory or LLM context is shared between designs:

    workspaces/<name>/
        workspace.json            {"design": "fixed" | "agentic", "created_at": ...}
        train.py                  the file the agents edit (starts as the root train.py, the baseline)
        prepare.py, inference.py  frozen copies of the root files (data, evaluation, benchmark images)
        program.md                the research brief; the agentic design gets an extra section
        results.tsv, runs/, run.log
        memory/                   state.json, playbook / notes, ideas.jsonl, agents/ (committed)
        traces/                   every LLM call and the live status file (git-ignored)

harness/run.py sets HARNESS_WORKSPACE for every process it starts; modules read it once at import.
"""

import datetime
import json
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DESIGNS = ("fixed", "agentic")
TEMPLATE = ["train.py", "prepare.py", "inference.py", "program.md"]
ADDENDA = {"agentic": os.path.join(ROOT, "harness", "designs", "agentic_program.md")}


def resolve(path=None):
    path = path or os.environ.get("HARNESS_WORKSPACE") or os.path.join(ROOT, "workspaces", "fixed")
    return os.path.abspath(path if os.path.isabs(path) else os.path.join(ROOT, path))


WS = resolve()
MEM = os.path.join(WS, "memory")
TRACES = os.path.join(WS, "traces")


def rel(path):
    """Path relative to the git root, for git commands."""
    return os.path.relpath(path, ROOT)


def design_of(ws):
    try:
        with open(os.path.join(ws, "workspace.json")) as f:
            return json.load(f).get("design")
    except (OSError, ValueError):
        return None


def create(ws, design):
    """Set up a new workspace from the root template (the baseline train.py and the frozen files)."""
    if design not in DESIGNS:
        raise SystemExit(f"unknown design {design!r}; choose one of {', '.join(DESIGNS)}")
    os.makedirs(os.path.join(ws, "memory"), exist_ok=True)
    os.makedirs(os.path.join(ws, "traces"), exist_ok=True)
    for name in TEMPLATE:
        shutil.copyfile(os.path.join(ROOT, name), os.path.join(ws, name))
    addendum = ADDENDA.get(design)
    if addendum:
        with open(addendum) as f, open(os.path.join(ws, "program.md"), "a") as out:
            out.write("\n\n" + f.read())
    with open(os.path.join(ws, "workspace.json"), "w") as f:
        json.dump({"design": design, "created_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                   "template": "root train.py (the baseline)"}, f, indent=2)
        f.write("\n")
