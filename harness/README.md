# harness

Our own agent harness for the autoresearch loop. It replaces "point Claude Code or Codex at program.md" with a small, inspectable program, and every LLM call goes to Crusoe Managed Inference.

## Run it

1. Put your Crusoe settings in `.env` at the project root. The file is git-ignored; see `.env.example`.
2. Check the connection with `python3 -m harness.llm`, which should reply "ready".
3. Pick a design and start the loop. Leave out `--design` and it asks you.

```bash
python3 harness/run.py --design fixed --until 16:15
```

```bash
python3 harness/run.py --design agentic --until 16:15
```

`--until` stops starting new experiments after that local time. `touch harness/STOP` finishes the current experiment and then stops. `--max-iters N` runs N experiments. A new workspace's first iteration runs the unmodified baseline `train.py`. There is one GPU, so run one design at a time.

## Two designs, two workspaces

| | `fixed` | `agentic` |
|---|---|---|
| Who decides the workflow | Code: planner, guards, run, fixer on a crash, reflector | The model: an orchestrator picks one action per turn, up to 8 turns and 4 sub-agents per experiment |
| Agents | Fixed roles with fixed prompts | Sub-agents the orchestrator spawns itself, choosing each one's role, task, context and model (V4 Pro, or Nemotron 3.5 Super VL for low-risk work). They run in parallel. |
| Literature | A background researcher fills `memory/ideas.jsonl` | The orchestrator searches (`search_literature`, You.com Search) and reads pages it finds (`read_url`, You.com Contents, research domains only, needs `YDC_API_KEY`) when it decides to |
| Memory | Playbook lessons written by the reflector | `memory/notes.md`, which the orchestrator rewrites itself (`write_notes`) |
| Task brief | `program.md` | `program.md` plus `designs/agentic_program.md`: one GPU so runs are serial, only thinking parallelizes, and the failure modes past models hit here |
| Shared code | Edit guards, training, keep/discard by val_loss, logging, git, push (`step.execute` / `step.record`) | Same |

Each design runs in its own workspace, `workspaces/<design>/`. A workspace holds its own `train.py`, frozen copies of `prepare.py` and `inference.py`, `program.md`, `results.tsv`, `runs/`, `run.log`, `memory/` and `traces/`. The harness only reads and writes inside the workspace that `run.py` passes it. The agentic `read_file` action is confined there too, so no task file, memory or LLM context crosses between designs. New workspaces start from the root `train.py`, which is the baseline. `workspaces/fixed/` holds the first 13 experiments, which ran before workspaces existed.

Point the web UI at a workspace: `python3 webui/server.py --root workspaces/agentic --port 8081`. It defaults to `workspaces/fixed`.

### Testing without the GPU

`HARNESS_FAKE_TRAIN=1` swaps training for a fake run: a run folder and `run.log` with a val_loss derived from the code. Everything else runs for real: the Crusoe calls, guards, git, logging and memory. The isolation test runs both designs in a throwaway clone and checks four things:
- each design's LLM prompts never contain the other workspace's `train.py`;
- each run writes only inside its own workspace;
- every commit touches only that workspace;
- `read_file` can't escape the workspace (`../`, `.env`, absolute paths).

## One iteration of the fixed design (`harness/step.py`)

Each experiment runs in a fresh process, and all state lives in files. A crash costs one iteration, and edits to the prompts or code take effect on the next one.

1. **Plan:** the planner (`prompts/planner.md`) reads `program.md`, the playbook, the experiment log with past predictions, the idea bank and the current best `train.py`. It replies with an idea (title, category, rationale, a predicted val_loss change), a `self_check`, and SEARCH/REPLACE edits. In the `self_check`, the planner states values its code produces at key points, traced by hand (e.g. "lr multiplier at progress 0 / 0.5 / 1.0 = 0 / 1 / 0").
2. **Apply:** the edits must match exactly and pass the guards below. A rejected reply gets up to two retries with the reason. Then `train.py` is committed as `exp N: ...`.
3. **Run:** `uv run train.py > run.log`, killed after 6 minutes. On a crash, the fixer (`prompts/fixer.md`) gets one repair attempt with the end of the log.
4. **Decide:** code decides, not the model. The change is kept if val_loss beats the best so far; otherwise `train.py` is reverted with a new commit. History only moves forward, so discarded ideas stay in git.
5. **Log:** the row is appended to `results.tsv`, and `idea.json` (idea, prediction, outcome) and `change.diff` are written into `runs/<run_id>/` for the web UI.
6. **Reflect:** the reflector (`prompts/reflector.md`) reads the diff that ran, checks it against the planner's `self_check`, and compares the prediction with the outcome. If the code didn't implement the idea, it records a lesson about the bug rather than the idea. It makes at most 3 itemized changes to the playbook (`memory/playbook.json`, also rendered as `playbook.md`), and code merges them.
7. **Map:** `map_architecture.py` adds the experiment to `architecture.json` (below). Both designs do this step, because it lives in the shared `step.record()`.

## Models and thinking (`harness/config.json`)

`models` maps each role to a Crusoe model:
- **DeepSeek V4 Pro:** the planner in both designs (the fixed design's planner, the model-driven design's orchestrator), and the fixer. These make the ML decisions and write the code.
- **Nemotron 3.5 Super VL (reasoning off):** reflector, researcher, and the architecture mapper (`mapper`). This is lower-risk work that the planner checks anyway.
- **Sub-agents:** `subagent_models` defines the tiers. The orchestrator picks "pro" for decisions, code and debugging, and "nemotron" for reading, summarizing and extracting. The old "flash" tier maps to "nemotron".

Why Nemotron there and not elsewhere: on Crusoe (2026-10-17) we replayed real harness calls. With reasoning off, Nemotron 3.5 Super VL matched V4-Flash on speed (1.6–4 s) and on the quality of the reflector's lessons, at a slightly lower input price. With reasoning on, it took 14–43 s. Without reasoning it failed a crash fix that V4 Pro got right, so it stays out of code-writing roles. Nemotron 3.5 Lightning was cheaper but wrote a playbook lesson the data didn't support. `llm.py` turns Nemotron's default reasoning off (`chat_template_kwargs: {"enable_thinking": false}`) unless the role is listed under `thinking`.

DeepSeek V4 only reasons when asked (`chat_template_kwargs: {"thinking": true}`). `thinking` lists the roles that get it, with a 16k-token budget. Measured on our prompts:

| Role | Without thinking | With thinking |
|---|---|---|
| Planner | 3–8 s | 129 s, a valid plan with a more concrete self-check |
| Fixer, replaying the exp 4 EMA crash | 2.6 s, correct diagnosis | 162 s, same diagnosis and a cleaner fix (the EMA update moves into the compiled step) |

Only the fixer has thinking on: it runs on roughly 1 in 13 experiments, and a bad repair wastes a 2-minute run. The planner and orchestrator stay fast, so more experiments fit in the time.

## The evaluator stays outside the loop

These checks are enforced in code, not by asking the model:

- `prepare.py` and `inference.py` are hashed when the loop starts. If either changes, the loop refuses to run.
- An edit is rejected if it redefines or reassigns anything imported from `prepare` (`TIME_BUDGET`, `evaluate_loss`, ...), reads the test split, removes the final `evaluate_loss`/`save_run` calls, or lowers `CAPTION_DROPOUT` below 0.1. A lower dropout improves val_loss but breaks classifier-free guidance.
- An edit is rejected if it reads a name that is never defined or imported anywhere in the file (a missing import, a typo), or uses an MLX function, class or module that doesn't exist in the installed MLX. MLX paths are resolved in the project's venv and cached in `traces/mlx_api_cache.json`.
- A run is discarded if training takes more than the time budget plus 100 s of wall clock, or if sampling takes more than 120 s.

### Why there is no reviewer agent

From experiment 3 to 13, a separate reviewer model checked every diff before it ran. In 17 live reviews it made no confirmed catches and 5 confirmed false flags. Two said "`math` is not imported" when it was, and three called correct MLX async code a race. One of those false flags became a false lesson in the playbook. It also missed the one real bug, a memory leak in the EMA change. It was replaced by two things: the planner's `self_check`, which targets the bug class the reviewer did catch in a replay (an inverted schedule), and the code checks above, which answer "is it defined?" with certainty.

## Files

| Path | What it holds | In git |
|---|---|---|
| `workspaces/<d>/memory/state.json` | Best commit and val_loss, and every idea with its prediction and outcome | yes |
| `workspaces/fixed/memory/playbook.json`, `playbook.md` | Lessons learned (fixed design) | yes |
| `workspaces/agentic/memory/notes.md`, `agents/` | The orchestrator's notes, and every sub-agent's role, task and reply (agentic design) | yes |
| `workspaces/<d>/architecture.json` | The component map the web UI draws, maintained by `map_architecture.py` | yes |
| `workspaces/<d>/traces/` | Every LLM call (prompt, reply, timing, usage), the live status file, and loop logs | no |

## Architecture map (`harness/map_architecture.py`)

`architecture.json` in the workspace is the component map that the web UI's Baseline, Evolution and Current views draw. Its schema is in `webui/README.md`. The harness keeps it up to date: after the baseline and after every experiment, `step.py` runs

```bash
python3 harness/map_architecture.py --workspace workspaces/fixed
```

It reads the baseline `train.py` (the baseline run's copy, else the root `train.py`), `results.tsv`, each run's `idea.json` and `change.diff`, and `memory/` (`state.json`, `playbook.json`, `ideas.jsonl`). The first run asks for the baseline's components, once. Then each experiment not yet in `changes` gets one small call (`prompts/mapper.md`, on the reflector's model) that returns the component it changed (label, after, cost, note), updates to the hypotheses it tested, and the component each new queued idea would change. With nothing new, it makes no call. A failed or timed-out call never costs an experiment: whatever is missing is mapped on the next run. `--dry-run` prints the merged map without writing it.

Merging only adds, so a hand-written map keeps everything a human wrote. Components, groups, `changes` and `next` only get missing keys, and `tried` only grows. An existing hypothesis keeps its text, its evidence only grows, and its status changes only together with new evidence. Hypotheses the script writes are marked `"auto": true`, and only those can be reworded later. If a merge would drop or rewrite an existing entry, the script refuses to write it.

## Researcher (`harness/researcher.py`)

Part of the fixed design: `run.py --design fixed` starts it as a background process (unless `--no-researcher`), and it keeps the workspace's idea bank, `memory/ideas.jsonl`, stocked for the planner. You.com only retrieves (Search, optionally Contents; never the Research endpoint, which runs You.com's own model). Every LLM call goes to Crusoe through `harness.llm` under the `researcher` role.

```bash
HARNESS_WORKSPACE=workspaces/fixed python3 -m harness.researcher --once     # one round
HARNESS_WORKSPACE=workspaces/fixed python3 -m harness.researcher --loop     # a round whenever fewer than 6 ideas are untried
```

`--max-searches N` caps You.com searches per process (default 40), and `--loop` exits when `harness/STOP` exists. With `YDC_API_KEY` in the environment or `.env`, it uses the REST Search API (`include_domains` arxiv.org, openreview.net, github.com, huggingface.co; highlights, ~3k tokens). Without a key, it uses the keyless MCP endpoint (`you-search`, about 100 queries a day). If search fails, Crusoe proposes ideas from its own knowledge, and those cards are marked `"origin": "model-only"` with no sources. Network errors are logged to `traces/researcher.log` and retried on the next round.

One round reads `results.tsv`, `memory/state.json`, `memory/playbook.md` and the best `train.py`'s hyperparameters, asks Crusoe for 2-3 queries aimed at the current bottleneck, searches, and asks Crusoe to distill the results into at most 5 cards. Web content is passed as delimited, untrusted data. Cards are dropped if their sources weren't in the results or if they repeat an earlier idea. Prompts are in `prompts/researcher.md`.

`memory/ideas.jsonl` is append-only, one JSON object per line:

```json
{"id": "I7", "status": "new", "title": "Logit-normal timestep sampling", "category": "other",
 "change": "one concrete change to train.py", "mechanism": "why val_loss should drop in 2 minutes",
 "sources": ["https://arxiv.org/..."], "mlx_risk": "low", "query": "q1 | q2", "created_at": "2026-10-17T14:08:25-07:00"}
```

A later line with the same `id` updates that idea, e.g. `{"id": "I7", "status": "tried", "exp": 9, "outcome": "discard"}`. Readers fold lines by id: `researcher.load_bank()` returns the merged ideas, `researcher.untried()` the ones still `new`, and `researcher.mark(id, **fields)` appends an update.
