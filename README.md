# Turing

Turing runs and visualizes autonomous machine-learning experiments. Agents propose code changes, train models, evaluate results, keep improvements, and record lessons.

- `harness/`: fixed and model-driven agent workflows, Crusoe inference, research retrieval, experiment execution, and architecture mapping.
- `webui/`: local project browser, architecture history, results, prompts, and agent controls.
- `workspaces/`: existing experiments, model snapshots, memory, and run artifacts.

## Run the interface

```sh
uv run --no-project python webui/server.py
```

Open http://localhost:8080. See [the UI documentation](webui/README.md) and [the harness documentation](harness/README.md).

## Image research project

The image-model source, instructions, README, and Python dependency files live outside this repository in `../autoresearch-image/`. Existing experiment workspaces remain here.

New image workspaces use that sibling folder as their baseline. Set `TURING_IMAGE_PROJECT` to use a different template directory. Training in internal image workspaces uses the external project's dependencies through `uv run --project`; it executes the workspace's own `train.py`.

```sh
uv sync --project ../autoresearch-image
uv run --no-project python harness/run.py --design fixed --max-iters 1
```

Configure Crusoe in `.env` as described in [harness/README.md](harness/README.md). Other projects can supply their own `project.json`, training command, and prompts; see [the autoresearch adapter](harness/projects/karpathy-autoresearch/README.md).

## Published research histories

Turing can display [modded-nanogpt’s speedrun history](webui/tools/README.md), including timing records, research categories, source links, and historical code diffs. Import the sibling clone with `uv run --no-project python webui/tools/modded_nanogpt.py ../modded-nanogpt --register`. This view does not start training jobs.

A [CS336 copied-project example](harness/projects/cs336/README.md) includes portable training and research configuration, evaluation documentation, and baseline architecture metadata.
