# webui

The interface for observing and steering autoresearch projects: pick a project, see its architecture evolve experiment by experiment, and control the agents that run it. It works for any project laid out like autoresearch, and uses the Python standard library and vanilla JS, with no CDN, so it works offline.

## Run it

```bash
uv run --no-project python webui/server.py
```

Then open http://localhost:8080. The server adds every `workspaces/*` project to the recent list (`~/.autoresearch/projects.json`). Add another project with `--root path/to/project` (repeatable) or with **Open folder** on the home page. A project is a folder with `train.py`, `results.tsv` and `runs/`. Upstream autoresearch projects work too, such as a clone of karpathy/autoresearch: its 5-column `results.tsv` is read by header, the metric is named after its column (`val_bpb`), and kept experiments' diffs come from the clone's git commits, since upstream keeps no run folders. `webui/tools/karpathy_map.py` writes the architecture map for karpathy's `exp/H100/mar8` run. Mock data: `uv run --no-project python webui/make_mock.py`, then `uv run --no-project python webui/server.py --root webui/mock`.

## Pages

- **Home** (`#/`): recent projects as icons. Each icon is that project's best-so-far curve.
- **Architecture** (`#/p/<id>/baseline|evolution|current`): the model as components, in three views (keys `1` `3` `4`). **Baseline** is what the loop started from. **Evolution** puts every experiment to the right of the component it changed, in order: green = kept, grey = discarded, red dashed = crashed, purple dashed = queued ideas. **Current** is the architecture after every kept change. The loss chart underneath is linked both ways: hovering a block highlights its point, and clicking a point selects its block. The right panel shows the hypotheses (validated or not, with the experiments that decided them), the queued ideas and the playbook's theories. Select a block to see its change (before → after, cost, params and steps), the hypothesis with predicted vs actual, why it worked or didn't, the code diff, and the samples. Keys: `←`/`→` step through experiments, `Esc` clears the selection, `T` switches light/dark.
- **Eval** (`#/p/<id>/eval`, key `2`): metric definition, baseline/best/latest scores, evaluator documentation and constants from the current checkout, and a table of recorded results. Evaluator code is parsed, never imported or executed. Published histories retain their comparison-series boundaries.
- **Agents** (`#/p/<id>/agents`, key `5`): start a bounded agentic research session, inspect agent definitions, and follow live model, tool, delegation, and experiment events. Standalone training is below the research workspace. The older source-editing harness remains available in an expandable section for managed workspaces.

## Start agentic research

Open **Agents → Start agentic research**. Set an objective, candidate count, and per-experiment timeout. Every session first runs a fresh baseline, then evaluates up to the requested number of candidates. An orchestrator can inspect allowlisted source, delegate analysis to parallel advisors, and propose bounded parameter changes. The trusted evaluator command produces a numeric result; the configured metric direction decides whether a candidate is kept.

Activity polls every second and distinguishes model calls, file reads, delegation, experiments, and lifecycle events. Select an event to inspect recorded tasks, responses, parameters, metrics, or live training output. The model status shows work in progress; it does not invent or stream private reasoning. **Stop research** cancels the session and its evaluator. Sessions and logs survive UI-server restarts; a detected orphan evaluator blocks another launch until it is stopped.

Research is shared across projects through `project.json.research` metadata. Enable an explicit adapter with `source_directory`, an argv `command` containing `{parameters}` and `{output}`, `metric`, `goal` (`min` or `max`), a typed `parameters` schema, `fixed_parameters`, a source `readable` allowlist, and immutable `protocol_files` (evaluator code, tokenizer assets, data, and dependency definitions). The adapter reads a JSON parameters file and writes a JSON object containing the final numeric metric. Models cannot supply commands or rewrite source through this research interface. Projects without an adapter display what setup is missing; standalone training remains available.

The CS336 copy has an adapter for `cs336_basics/pretrain.py`: it tunes model dimensions and optimizer parameters with a fixed 200-step screening protocol, seeds and validation windows. This is distinct from its standalone 2,000-step configuration. Source files stay unchanged. Session parameters, logs, checkpoints and events are stored under `research/`; results also appear in the shared Eval/results view, separated by comparison series. `research/best.json` stores the session's best settings without changing standalone training defaults. The orchestrator and advisors use the existing `harness/config.json` model settings and Crusoe credentials.

## Start training

Open **Agents → Start training**. Choose the project command, the modded-nanogpt preset, or a custom command; select this computer or SSH; then set the checkout directory and time limit. Commands use shell-style quoting but execute as argument lists: pipes, redirects, and shell expansion are not supported.

**Save setup** retains settings per project. **Check setup** verifies the directory, executable and script files; the modded-nanogpt profile also checks eight H100 GPUs, PyTorch 2.10, required Python imports and FineWeb shards. It does not import the trainer or download data or kernels. **Start training** automatically checks the setup and starts one run; a separate Check setup click is optional. Follow live output, open its full log, or use **Stop training**. The time limit includes setup performed by the command. The process and its logs survive restarting the UI server.

For SSH, enter an existing `~/.ssh/config` alias or `user@hostname`, with noninteractive key authentication. The machine must already have `uv`, the checkout, dependencies, and data prepared according to the upstream trainer instructions. Turing does not provision a GPU instance. The modded-nanogpt preset defaults to `~/modded-nanogpt` and `uv run --no-project torchrun --standalone --nproc_per_node=8 train_gpt.py`; adjust these to the prepared environment. The wrapper requests process-group cancellation on stop or SSH disconnect; an unconfirmed remote outcome is shown explicitly and should be checked on the host before restarting.

Settings and jobs are stored under each project's `training/` directory. Training runs are separate from imported leaderboard history and the autonomous research loop. A successful process exit does not establish benchmark acceptance or add a result to the architecture chart. These controls are available only through the local, same-origin UI.

## Evaluation documentation for any project

Eval uses the same layout for every project. Scores and memory come from `results.tsv`; the metric name, direction (`min` or `max`), and optional display unit come from `architecture.json.metric`. The best score considers kept results in the latest comparison series.

Add an optional `evaluation.json` in a project folder to describe its evaluation without changing Turing's UI:

```json
{
  "title": "Validation accuracy",
  "description": "Accuracy on the held-out validation split.",
  "formula": "correct predictions / evaluated examples",
  "facts": [
    {"label": "Validation set", "value": "validation-v2"},
    {"label": "Evaluation size", "value": 10000, "unit": "examples"}
  ],
  "note": "Compare runs using the same validation split and preprocessing."
}
```

Without this file, supported autoresearch evaluators expose their `prepare.py` docstrings and constants through static parsing. Imported benchmark histories show their source summary and comparison limits. Other projects still show recorded metrics and results, with an explicit unavailable-documentation message. Turing never imports or runs evaluator code to build this page.

## architecture.json

Optional, in the project folder. It maps the model's components and tags each experiment with the component it changed. Without it, the Evolution view groups experiments by the planner's idea category. See `workspaces/fixed/architecture.json`:

- `components[]`: `{id, group, name, baseline: {label, detail, cost, params}}`, in diagram order. `groups[]` names the groups; `flow: true` draws them as a forward pass.
- `changes{n}`: `{component, label, after, cost, note}` for experiment n. `after` becomes the component's label once the change is kept.
- `hypotheses[]`: `{text, status: validated|invalidated|inconclusive|untested, evidence: [n, ...]}`. Without it, each experiment's prediction is checked against its result.
- `next{idea_id: component}` places queued ideas from `memory/ideas.jsonl`; `tried[]` hides ideas that already ran.

### Baseline architecture diagrams

Baseline opens in **Diagram** mode: a bottom-to-top schematic with colored operation blocks, directional arrows, residual bypasses, and repeated-block enclosures. Select a component to inspect its details, or switch to **Components** for the original list. The display preference is shared across projects and remembered in the browser. Rendering uses local architecture metadata; it does not make a model call.

Every project uses the same renderer. Without explicit connections, it shows a grouped component map. Published history projects remain category maps: research categories alone cannot establish a model's forward pass.

An optional `diagram` in `architecture.json` declares a **baseline-only** forward pass. Component references pull their names and values from `components[]`; training and other unreferenced settings remain visible beside or below the model. For example:

```json
{
  "diagram": {
    "version": 1,
    "basis": "Source defaults",
    "description": "Describe what this diagram represents and its provenance.",
    "flow": [
      {"kind": "input", "title": "Token IDs"},
      {"component": "embedding", "tone": "embedding"},
      {"kind": "repeat", "component": "blocks", "children": [
        {"kind": "residual", "title": "Attention sublayer", "children": [
          {"kind": "operation", "title": "RMSNorm", "tone": "norm"},
          {"component": "attention", "tone": "attention"}
        ]},
        {"kind": "residual", "title": "Feed-forward sublayer", "children": [
          {"kind": "operation", "title": "RMSNorm", "tone": "norm"},
          {"component": "ffn", "tone": "ffn"}
        ]}
      ]},
      {"component": "head", "tone": "head"},
      {"kind": "output", "title": "Next-token logits"}
    ]
  }
}
```

`flow` and container `children` are ordered by forward execution; the schematic places inputs at the bottom and outputs at the top. `repeat` groups a repeated sequence; its component label supplies the recorded depth. `residual` explicitly adds a bypass and sum around its children. `parallel` displays independent branches. Leaf kinds are `input`, `output`, and `operation`. Nodes may have a `title` and a short explanatory `note` (available on hover). Optional `tone` selects an operation color: `embedding`, `attention`, `ffn`, `norm`, `head`, `input`, or `output`. Colors are declared explicitly; an unspecified tone stays neutral. Component blocks remain keyboard-accessible buttons that open the existing details panel. Optional `source` metadata can record source files, symbols, or a baseline commit. Supply only connections verified from that baseline's code; the UI never infers topology from component names or list order. Invalid or incomplete diagrams fall back to the component map.

Current and Evolution keep their existing recorded-change views. A baseline diagram does not establish the topology of later architectures or apply research-run parameter overrides to source defaults.

## Annotate (design review)

Add `?annotate` to any page, e.g. http://localhost:8080/?annotate#/. Hover a component to see its name and size, use `↑`/`↓` to widen to the parent or narrow back, and click to pin a numbered note. `A` turns picking off so you can use the app normally (to select a block, or switch views), and on again. Notes are saved to `webui/annotations.json` with each component's selector, text, and position, so Claude can read them. **Copy** puts them on the clipboard as Markdown.

## API

- `GET /api/projects`: recent projects. `POST /api/projects {"path"}` adds one.
- `GET /api/p/<id>`: `experiments[]` (results.tsv joined with each run's metrics.json, idea.json and diff size), `architecture`, `ideas`, `harness.lessons`, `harness.status`, `live` (progress parsed from `run.log`), and `summary`.
- `GET /api/p/<id>/agents`: designs, roles and models, status, the loop log and recent LLM calls. `GET /api/p/<id>/trace/<file>` returns one call in full.
- `POST /api/p/<id>/run {"design", "until"}` starts `harness/run.py`. `POST /api/p/<id>/stop` writes `harness/STOP` (add `"now": true` to end the process).
- `POST /api/p/<id>/training-save`, `training-check`, and `training-start` accept `{ "config": { "profile", "target", "host", "directory", "command", "timeout_minutes" } }`. `training-stop` accepts `{ "job_id" }`. Agent state includes settings, recent jobs, and the latest output; `GET /api/p/<id>/training-log/<job_id>` reads a full log.
- `POST /api/p/<id>/research-start` accepts `{ "objective", "max_experiments", "experiment_timeout_minutes" }`; `research-stop` accepts `{ "run_id" }`. Agent state includes `research.config`, `active`, `latest`, `events`, and evaluator progress.
- `GET`/`POST /api/prompts/<role>` reads or writes `harness/prompts/<role>.md`.

Run files are served at `/p/<id>/runs/<run_id>/<file>`, for the whitelisted names only.

## modded-nanogpt

Import the published 8×H100 speedrun history with `uv run --no-project python webui/tools/modded_nanogpt.py ../modded-nanogpt --register`. The generated project charts training seconds, preserves re-timings, and links milestones to logs, PRs, and available source diffs. See [the importer documentation](tools/README.md) for provenance and training limitations.
