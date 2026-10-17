"""
Presentation server for the autoresearch diffusion runs. Standard library only.

    python3 webui/server.py                       # project root (parent of webui/), http://localhost:8080
    python3 webui/server.py --root webui/mock     # mock data from make_mock.py
    python3 webui/server.py --port 9000 --host 0.0.0.0

Routes:
    /                      the UI (webui/index.html, app.js, style.css)
    /api/runs              results.tsv joined with runs/*/metrics.json and the optional files
    /runs/<run_id>/<file>  samples.png, trajectory.png, train.py, change.diff, idea.json, metrics.json
"""

import argparse
import json
import math
import mimetypes
import os
import re
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = {"/": "index.html", "/index.html": "index.html", "/app.js": "app.js", "/style.css": "style.css"}
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
    """Parse results.tsv into a list of dicts, in experiment order. Tolerates a missing header or short rows."""
    try:
        with open(os.path.join(root, "results.tsv")) as f:
            raw = f.read().splitlines()
    except OSError:
        return []
    rows = []
    for line in raw:
        if not line.strip():
            continue
        parts = line.split("\t", len(HEADER) - 1)
        if parts[0].strip().lower() == "commit":
            continue
        parts += [""] * (len(HEADER) - len(parts))
        row = dict(zip(HEADER, (p.strip() for p in parts)))
        row["status"] = row["status"].lower() or "unknown"
        row["val_loss"] = to_float(row["val_loss"])
        if row["status"] == "crash" or row["val_loss"] == 0:
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


def run_info(root, run_id):
    """Everything the UI needs from runs/<run_id>/. Image URLs carry the file mtime so the browser refetches on change."""
    run_dir = os.path.join(root, "runs", run_id)
    if not os.path.isdir(run_dir):
        return None
    metrics = read_json(os.path.join(run_dir, "metrics.json"))
    files = {}
    for name in RUN_FILES:
        path = os.path.join(run_dir, name)
        try:
            files[name] = f"/runs/{run_id}/{name}?v={int(os.path.getmtime(path))}"
        except OSError:
            pass
    idea = read_json(os.path.join(run_dir, "idea.json")) if "idea.json" in files else None
    if idea is not None and not isinstance(idea, dict):
        idea = None
    return {"run_id": run_id, "metrics": metrics, "files": files, "idea": idea,
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


def build_state(root):
    rows = read_results(root)
    try:
        run_ids = sorted(d for d in os.listdir(os.path.join(root, "runs"))
                         if RUN_ID_RE.match(d) and os.path.isdir(os.path.join(root, "runs", d)))
    except OSError:
        run_ids = []

    baseline = rows[0]["val_loss"] if rows else None
    best = None        # running best over kept rows (lower is better)
    best_idx = None
    experiments = []
    for i, row in enumerate(rows):
        info = run_info(root, row["run_id"]) if row["run_id"] and RUN_ID_RE.match(row["run_id"]) else None
        v = row["val_loss"]
        exp = {
            "idx": i, "commit": row["commit"] or None, "status": row["status"], "val_loss": v,
            "memory_gb": row["memory_gb"], "run_id": row["run_id"], "description": row["description"],
            "best_before": best,
            "delta_baseline": (v - baseline) if v is not None and baseline is not None and i > 0 else None,
            "delta_best": (v - best) if v is not None and best is not None else None,
        }
        exp.update({k: info[k] for k in ("metrics", "files", "idea", "started_at")} if info else
                   {"metrics": None, "files": {}, "idea": None, "started_at": None})
        if row["status"] == "keep" and v is not None and (best is None or v < best):
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
        info = run_info(root, rid)
        if info:
            info["stage"] = "logging" if "samples.png" in info["files"] else "sampling"
            pending.append(info)

    counts = {}
    for e in experiments:
        counts[e["status"]] = counts.get(e["status"], 0) + 1
    prompt = next((e["metrics"].get("prompt") for e in list(reversed(experiments)) + pending
                   if e.get("metrics") and e["metrics"].get("prompt")), None)
    return {
        "root": os.path.abspath(root),
        "server_time": time.time(),
        "prompt": prompt,
        "experiments": experiments,
        "pending": pending,
        "live": live_log(root),
        "summary": {"baseline_val_loss": baseline, "best_val_loss": best, "best_idx": best_idx,
                    "counts": counts, "total": len(experiments)},
    }


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
            if path == "/api/runs":
                body = json.dumps(build_state(self.root), allow_nan=False).encode()
                return self.send(200, body, "application/json; charset=utf-8", "no-store")
            parts = path.strip("/").split("/")
            if len(parts) == 3 and parts[0] == "runs" and RUN_ID_RE.match(parts[1]) and parts[2] in RUN_FILES:
                # URLs carry ?v=<mtime>, so a changed file gets a new URL and these can be cached.
                return self.serve_file(os.path.join(self.root, "runs", parts[1], parts[2]), "max-age=3600")
            if path == "/favicon.ico":
                return self.send(204, b"", "image/x-icon")
            self.not_found()
        except (BrokenPipeError, ConnectionResetError):
            pass

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
    ap.add_argument("--root", default=os.path.dirname(HERE), help="project root with results.tsv and runs/ (default: parent of webui/)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to reach it from another device")
    args = ap.parse_args()

    Handler.root = os.path.abspath(args.root)
    if not os.path.isdir(Handler.root):
        raise SystemExit(f"--root {Handler.root} is not a directory")
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.daemon_threads = True
    shown = "localhost" if args.host in ("127.0.0.1", "0.0.0.0") else args.host
    print(f"Serving {Handler.root}\n  http://{shown}:{args.port}  (Ctrl-C to stop)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
