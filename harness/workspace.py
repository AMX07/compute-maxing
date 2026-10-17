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

A workspace can also be a project outside this repo, such as a clone of karpathy/autoresearch. Then:
    project.json                  how to train and score it (see PROFILE_DEFAULTS); missing keys use the defaults
    prompts/<role>.md             optional per-project prompts, in place of harness/prompts/<role>.md
and git commands run in that project's own repository.
"""

import datetime
import json
import os
import shutil
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DESIGNS = ("fixed", "agentic")
TEMPLATE = ["train.py", "prepare.py", "inference.py", "program.md"]
ADDENDA = {"agentic": os.path.join(ROOT, "harness", "designs", "agentic_program.md")}

# What a project needs the harness to know. The defaults are this repo's image project, so its workspaces need no
# project.json. Values in <workspace>/project.json override them key by key.
PROFILE_DEFAULTS = {
    "metric": "val_loss",                  # train.py prints "<metric>: <value>"; lower is better
    "memory_field": "peak_memory_gb",      # printed field with peak memory, and the factor that turns it into GB
    "memory_scale": 1.0,
    "frozen": ["prepare.py", "inference.py"],   # files the loop must never change (data, evaluation)
    "required": ["evaluate_loss(model", "save_run("],   # snippets an edited train.py must keep
    "min_values": {"CAPTION_DROPOUT": 0.1},     # train.py constants that must stay at or above these values
    "readable": ["train.py", "prepare.py", "inference.py", "program.md", "run.log", "results.tsv", "memory/notes.md"],
    "train_command": "uv run train.py",
    "remote": None,                        # {"host", "dir", "sync": ["train.py"]}: train over ssh on a GPU box
    "run_timeout": 360,                    # seconds before a run is killed
    "train_slack": 100,                    # seconds allowed on top of TIME_BUDGET for startup, compile and eval
    "push": "main",                        # branch to push after each experiment, or null to never push
    "fake_base": 0.1838,                   # HARNESS_FAKE_TRAIN: the metric of the fake baseline
}


def resolve(path=None):
    path = path or os.environ.get("HARNESS_WORKSPACE") or os.path.join(ROOT, "workspaces", "fixed")
    return os.path.abspath(path if os.path.isabs(path) else os.path.join(ROOT, path))


def load_profile(ws):
    profile = dict(PROFILE_DEFAULTS)
    try:
        with open(os.path.join(ws, "project.json")) as f:
            profile.update(json.load(f))
    except (OSError, ValueError):
        pass
    return profile


def git_root(ws):
    """The repository that holds the workspace: this repo for workspaces/*, the project's own for an outside one."""
    try:
        r = subprocess.run(["git", "-C", ws, "rev-parse", "--show-toplevel"], capture_output=True, text=True, timeout=10)
        if r.returncode == 0 and r.stdout.strip():
            return os.path.realpath(r.stdout.strip())
    except (OSError, subprocess.TimeoutExpired):
        pass
    return ROOT


WS = resolve()
MEM = os.path.join(WS, "memory")
TRACES = os.path.join(WS, "traces")
PROFILE = load_profile(WS)
GIT_ROOT = git_root(WS) if os.path.isdir(WS) else ROOT
METRIC = PROFILE["metric"]


def rel(path):
    """Path relative to the workspace's git root, for git commands."""
    return os.path.relpath(os.path.realpath(path), GIT_ROOT)


def prompt_path(role):
    """The project's own prompt for a role if it has one, else the harness's."""
    own = os.path.join(WS, "prompts", role + ".md")
    return own if os.path.exists(own) else os.path.join(ROOT, "harness", "prompts", role + ".md")


def design_of(ws):
    try:
        with open(os.path.join(ws, "workspace.json")) as f:
            return json.load(f).get("design")
    except (OSError, ValueError):
        return None


def create(ws, design):
    """Set up a new workspace from the root template (the baseline train.py and the frozen files). A folder that
    already has a train.py is an existing project (e.g. a clone of karpathy/autoresearch): it is adopted as it is,
    and only workspace.json, memory/ and traces/ are added."""
    if design not in DESIGNS:
        raise SystemExit(f"unknown design {design!r}; choose one of {', '.join(DESIGNS)}")
    adopt = os.path.exists(os.path.join(ws, "train.py"))
    os.makedirs(os.path.join(ws, "memory"), exist_ok=True)
    os.makedirs(os.path.join(ws, "traces"), exist_ok=True)
    if not adopt:
        for name in TEMPLATE:
            shutil.copyfile(os.path.join(ROOT, name), os.path.join(ws, name))
    # The design's addendum to program.md: the project's own prompts/<design>_program.md, else (for this repo's
    # template only) harness/designs/<design>_program.md, which describes the image model.
    own = os.path.join(ws, "prompts", f"{design}_program.md")
    addendum = own if os.path.exists(own) else (None if adopt else ADDENDA.get(design))
    if addendum:
        with open(addendum) as f, open(os.path.join(ws, "program.md"), "a") as out:
            out.write("\n\n" + f.read())
    if adopt:  # keep the project's own git status clean: run output and traces are not part of its history
        exclude = os.path.join(git_root(ws), ".git", "info", "exclude")
        try:
            have = open(exclude).read() if os.path.exists(exclude) else ""
            extra = [p for p in ("traces/", "runs/", "run.log") if p not in have.split()]
            if extra and os.path.isdir(os.path.dirname(exclude)):
                with open(exclude, "a") as f:
                    f.write("\n# autoresearch harness\n" + "\n".join(extra) + "\n")
        except OSError:
            pass
    with open(os.path.join(ws, "workspace.json"), "w") as f:
        json.dump({"design": design, "created_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                   "template": "the project's own train.py" if adopt else "root train.py (the baseline)"}, f, indent=2)
        f.write("\n")
