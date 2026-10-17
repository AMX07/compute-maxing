"""
Crusoe Managed Inference client: OpenAI-compatible chat completions, standard library only.

Configuration comes from the environment or a .env file at the project root (never committed):
    CRUSOE_API_KEY=...
    CRUSOE_BASE_URL=https://.../v1
    CRUSOE_MODEL=...
Every call is written to harness/traces/ so the loop's decisions can be inspected afterwards.
"""

import json
import os
import re
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACE_DIR = os.path.join(ROOT, "harness", "traces")


def load_env():
    path = os.path.join(ROOT, ".env")
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip().removeprefix("export ").strip()
            os.environ.setdefault(key, value.strip().strip('"').strip("'"))


def config():
    load_env()
    cfg = {k: os.environ.get(k, "").strip() for k in ("CRUSOE_API_KEY", "CRUSOE_BASE_URL", "CRUSOE_MODEL")}
    missing = [k for k, v in cfg.items() if not v]
    if missing:
        raise SystemExit(f"Missing {', '.join(missing)}. Put them in {os.path.join(ROOT, '.env')} (see .env.example).")
    if not cfg["CRUSOE_BASE_URL"].startswith("http") and cfg["CRUSOE_API_KEY"].startswith("http"):
        cfg["CRUSOE_API_KEY"], cfg["CRUSOE_BASE_URL"] = cfg["CRUSOE_BASE_URL"], cfg["CRUSOE_API_KEY"]  # swapped in .env
    if not cfg["CRUSOE_BASE_URL"].startswith("http"):
        raise SystemExit("CRUSOE_BASE_URL in .env must be a URL starting with https:// (value not shown).")
    return cfg


def chat(messages, role, tag, temperature=0.7, max_tokens=8192, timeout=600):
    """Send a chat completion and return the text reply. role/tag only name the trace file."""
    cfg = config()
    url = cfg["CRUSOE_BASE_URL"].rstrip("/") + "/chat/completions"
    body = {"model": cfg["CRUSOE_MODEL"], "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    t0 = time.time()
    last_err = None
    for attempt in range(5):
        req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST", headers={
            "Authorization": f"Bearer {cfg['CRUSOE_API_KEY']}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.load(resp)
            msg = data["choices"][0]["message"]
            text = msg.get("content") or ""
            trace(tag, role, messages, text, data.get("usage"), time.time() - t0, msg.get("reasoning_content"))
            return text
        except urllib.error.HTTPError as e:
            detail = e.read()[:800].decode("utf-8", "replace")
            last_err = f"HTTP {e.code}: {detail}"
            if e.code not in (408, 409, 425, 429, 500, 502, 503, 504):
                break
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError, KeyError, IndexError) as e:
            last_err = f"{type(e).__name__}: {e}"
        time.sleep(min(60, 5 * 2 ** attempt))
    trace(tag, role, messages, None, None, time.time() - t0, None, error=last_err)
    raise RuntimeError(f"Crusoe call failed ({role}): {last_err}")


def trace(tag, role, messages, reply, usage, seconds, reasoning, error=None):
    os.makedirs(TRACE_DIR, exist_ok=True)
    path = os.path.join(TRACE_DIR, f"{tag}-{role}-{int(time.time())}.json")
    with open(path, "w") as f:
        json.dump({"role": role, "model": os.environ.get("CRUSOE_MODEL"), "seconds": round(seconds, 1), "usage": usage,
                   "error": error, "messages": messages, "reasoning": reasoning, "reply": reply}, f, indent=1)


def strip_thinking(text):
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()


def extract_json(text):
    """First JSON object in the reply: a ```json fence if present, else the first parseable {...}."""
    text = strip_thinking(text)
    for m in re.finditer(r"```(?:json)?\s*\n(.*?)```", text, flags=re.S):
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    dec = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                obj, _ = dec.raw_decode(text[i:])
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue
    return None


def extract_edits(text):
    """SEARCH/REPLACE blocks (Aider style) -> [(search, replace)]."""
    edits, mode, search, replace = [], None, [], []
    for line in strip_thinking(text).splitlines():
        marker = line.strip()
        if marker.startswith("<<<<<<<") and "SEARCH" in marker:
            mode, search, replace = "search", [], []
        elif marker == "=======" and mode == "search":
            mode = "replace"
        elif marker.startswith(">>>>>>>") and "REPLACE" in marker and mode == "replace":
            edits.append(("\n".join(search), "\n".join(replace)))
            mode = None
        elif mode == "search":
            search.append(line)
        elif mode == "replace":
            replace.append(line)
    return edits


if __name__ == "__main__":  # smoke test: python3 -m harness.llm
    print(strip_thinking(chat([{"role": "user", "content": "Reply with the single word: ready"}], "smoke", "test",
                              temperature=0, max_tokens=512)))
