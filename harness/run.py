"""
Run the research loop: one fresh `python3 -m harness.step` process per experiment.

    python3 harness/run.py                   # until stopped
    python3 harness/run.py --until 16:15     # don't start a new experiment after 4:15 PM local time
    python3 harness/run.py --max-iters 1     # a single experiment
    touch harness/STOP                       # finish the current experiment, then stop

All state is in files (see harness/step.py), so the loop can be stopped and restarted at any point.
"""

import argparse
import datetime
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STOP = os.path.join(ROOT, "harness", "STOP")


def log(msg):
    print(f"[loop {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Run the autoresearch loop.")
    ap.add_argument("--until", help="local time HH:MM after which no new experiment starts")
    ap.add_argument("--max-iters", type=int, default=0, help="stop after this many experiments (0 = no limit)")
    args = ap.parse_args()

    deadline = None
    if args.until:
        h, m = map(int, args.until.split(":"))
        deadline = datetime.datetime.now().replace(hour=h, minute=m, second=0, microsecond=0)
    if os.path.exists(STOP):
        os.remove(STOP)

    done = failures = 0
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
        rc = subprocess.run([sys.executable, "-m", "harness.step"], cwd=ROOT).returncode
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


if __name__ == "__main__":
    main()
