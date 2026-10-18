"""
Run the research loop with the agent design of your choice. Each design runs in its own workspace
(workspaces/<design>/: its own train.py, results, runs, memory and traces), so the two never share files or context.

    python3 harness/run.py --design fixed --until 16:15     # planner -> fixer -> reflector, plus a background researcher
    python3 harness/run.py --design agentic --until 16:15   # one orchestrator decides when to spawn agents, search, read
    python3 harness/run.py                                  # asks which design to use
    python3 harness/run.py --design agentic --max-iters 2   # a fixed number of experiments
    touch harness/STOP                                      # finish the current experiment, then stop
    HARNESS_FAKE_TRAIN=1 python3 harness/run.py ...         # no GPU: a fake run replaces training (for tests)

A new workspace starts from the external image template (the baseline train.py and the frozen prepare.py/inference.py).
"""

import argparse
import datetime
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from harness import workspace as wsp  # noqa: E402

STOP = os.path.join(ROOT, "harness", "STOP")
ENTRY = {"fixed": "harness.step", "agentic": "harness.agentic"}
ABOUT = {
    "fixed": "fixed roles in code: planner -> guards -> run -> fixer on crash -> reflector -> playbook, "
             "with a background You.com researcher filling an idea bank",
    "agentic": "model-driven: an orchestrator decides each turn whether to spawn sub-agents (it picks role, task, "
               "context and model), search, read files, update its notes, or run the experiment",
}


def log(msg):
    print(f"[loop {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def choose_design(arg):
    if arg:
        return arg
    if not sys.stdin.isatty():
        raise SystemExit("Choose a design with --design fixed or --design agentic.")
    print("Which agent design should run?")
    for i, d in enumerate(wsp.DESIGNS, 1):
        print(f"  {i}. {d}: {ABOUT[d]}")
    while True:
        ans = input("Design [1/2]: ").strip().lower()
        if ans in ("1", "2"):
            return wsp.DESIGNS[int(ans) - 1]
        if ans in wsp.DESIGNS:
            return ans


def main():
    ap = argparse.ArgumentParser(description="Run the autoresearch loop.")
    ap.add_argument("--design", choices=wsp.DESIGNS, help="fixed (roles in code) or agentic (the model decides)")
    ap.add_argument("--workspace", help="default: workspaces/<design>")
    ap.add_argument("--until", help="local time HH:MM after which no new experiment starts")
    ap.add_argument("--max-iters", type=int, default=0, help="stop after this many experiments (0 = no limit)")
    ap.add_argument("--no-researcher", action="store_true", help="fixed design: don't start the background researcher")
    args = ap.parse_args()

    design = choose_design(args.design)
    ws = wsp.resolve(args.workspace or os.path.join("workspaces", design))
    shown = os.path.relpath(ws, ROOT) if ws.startswith(ROOT + os.sep) else ws
    if not os.path.exists(os.path.join(ws, "workspace.json")):
        adopt = os.path.exists(os.path.join(ws, "train.py"))
        wsp.create(ws, design)
        log(f"adopted the existing project {shown} (its files are unchanged)" if adopt
            else f"created workspace {shown} from the baseline template")
    elif wsp.design_of(ws) != design:
        raise SystemExit(f"{shown} belongs to the {wsp.design_of(ws)!r} design; pick another --workspace.")
    env = {**os.environ, "HARNESS_WORKSPACE": ws, "HARNESS_DESIGN": design}
    log(f"design: {design} ({ABOUT[design]})")
    log(f"workspace: {shown}" + ("  [FAKE TRAINING: no GPU]" if env.get("HARNESS_FAKE_TRAIN") == "1" else ""))

    deadline = None
    if args.until:
        h, m = map(int, args.until.split(":"))
        deadline = datetime.datetime.now().replace(hour=h, minute=m, second=0, microsecond=0)
    if os.path.exists(STOP):
        os.remove(STOP)

    researcher = None
    if design == "fixed" and not args.no_researcher:
        out = open(os.path.join(ws, "traces", "researcher-loop.log"), "a")
        researcher = subprocess.Popen([sys.executable, "-u", "-m", "harness.researcher", "--loop", "--max-searches", "60"],
                                      cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT)
        log(f"background researcher started (pid {researcher.pid})")

    done = failures = 0
    try:
        while True:
            if os.path.exists(STOP):
                log("harness/STOP found; stopping")
                break
            if deadline and datetime.datetime.now() >= deadline:
                log(f"reached --until {args.until}; stopping")
                break
            if args.max_iters and done >= args.max_iters:
                log(f"finished {done} experiment(s)")
                break
            rc = subprocess.run([sys.executable, "-m", ENTRY[design]], cwd=ROOT, env=env).returncode
            done += 1
            if rc == 0:
                failures = 0
                continue
            failures += 1
            log(f"iteration exited with {rc} ({failures} failure(s) in a row)")
            if failures >= 5:
                log("5 failed iterations in a row; stopping so a human can look")
                break
            time.sleep(20)
    finally:
        if researcher and researcher.poll() is None:
            researcher.terminate()
            log("background researcher stopped")


if __name__ == "__main__":
    main()
