# webui

The interface for observing and steering autoresearch projects: pick a project, see its architecture evolve experiment by experiment, and control the agents that run it. It works for any project laid out like autoresearch, and uses the Python standard library and vanilla JS, with no CDN, so it works offline.

## Run it

```bash
python3 webui/server.py
```

Then open http://localhost:8080. The server adds every `workspaces/*` project to the recent list (`~/.autoresearch/projects.json`). Add another project with `--root path/to/project` (repeatable) or with **Open folder** on the home page. A project is a folder with `train.py`, `results.tsv` and `runs/`. Upstream autoresearch projects work too, such as a clone of karpathy/autoresearch: its 5-column `results.tsv` is read by header, the metric is named after its column (`val_bpb`), and kept experiments' diffs come from the clone's git commits, since upstream keeps no run folders. `webui/tools/karpathy_map.py` writes the architecture map for karpathy's `exp/H100/mar8` run. Mock data: `python3 webui/make_mock.py`, then `python3 webui/server.py --root webui/mock`.

## Pages

- **Home** (`#/`): recent projects as icons. Each icon is that project's best-so-far curve.
- **Architecture** (`#/p/<id>/baseline|evolution|current`): the model as components, in three views (keys `1` `2` `3`). **Baseline** is what the loop started from. **Evolution** puts every experiment to the right of the component it changed, in order: green = kept, grey = discarded, red dashed = crashed, purple dashed = queued ideas. **Current** is the architecture after every kept change. The loss chart underneath is linked both ways: hovering a block highlights its point, and clicking a point selects its block. The right panel shows the hypotheses (validated or not, with the experiments that decided them), the queued ideas and the playbook's theories. Select a block to see its change (before → after, cost, params and steps), the hypothesis with predicted vs actual, why it worked or didn't, the code diff, and the samples. Keys: `←`/`→` step through experiments, `Esc` clears the selection, `T` switches light/dark.
- **Agents** (`#/p/<id>/agents`): pick the design (custom pipeline or model-driven), start or stop the loop, edit any role's prompt (`harness/prompts/*.md`), and follow every LLM call as it happens: what the agent read, its reasoning and its reply. Only projects under this repo's `workspaces/` can be started from here.

## architecture.json

Optional, in the project folder. It maps the model's components and tags each experiment with the component it changed. Without it, the Evolution view groups experiments by the planner's idea category. See `workspaces/fixed/architecture.json`:

- `components[]`: `{id, group, name, baseline: {label, detail, cost, params}}`, in diagram order. `groups[]` names the groups; `flow: true` draws them as a forward pass.
- `changes{n}`: `{component, label, after, cost, note}` for experiment n. `after` becomes the component's label once the change is kept.
- `hypotheses[]`: `{text, status: validated|invalidated|inconclusive|untested, evidence: [n, ...]}`. Without it, each experiment's prediction is checked against its result.
- `next{idea_id: component}` places queued ideas from `memory/ideas.jsonl`; `tried[]` hides ideas that already ran.

## Annotate (design review)

Add `?annotate` to any page, e.g. http://localhost:8080/?annotate#/. Hover a component to see its name and size, use `↑`/`↓` to widen to the parent or narrow back, and click to pin a numbered note. `A` turns picking off so you can use the app normally (to select a block, or switch views), and on again. Notes are saved to `webui/annotations.json` with each component's selector, text, and position, so Claude can read them. **Copy** puts them on the clipboard as Markdown.

## API

- `GET /api/projects`: recent projects. `POST /api/projects {"path"}` adds one.
- `GET /api/p/<id>`: `experiments[]` (results.tsv joined with each run's metrics.json, idea.json and diff size), `architecture`, `ideas`, `harness.lessons`, `harness.status`, `live` (progress parsed from `run.log`), and `summary`.
- `GET /api/p/<id>/agents`: designs, roles and models, status, the loop log and recent LLM calls. `GET /api/p/<id>/trace/<file>` returns one call in full.
- `POST /api/p/<id>/run {"design", "until"}` starts `harness/run.py`. `POST /api/p/<id>/stop` writes `harness/STOP` (add `"now": true` to end the process).
- `GET`/`POST /api/prompts/<role>` reads or writes `harness/prompts/<role>.md`.

Run files are served at `/p/<id>/runs/<run_id>/<file>`, for the whitelisted names only.
