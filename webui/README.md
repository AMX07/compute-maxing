# webui

The presentation view for the autoresearch diffusion runs. It shows the progress chart, the experiment log, the samples for each run, and a side-by-side comparison of two runs. It uses the Python standard library and vanilla JS, with no CDN, so it works offline.

## Run it

Real data, from the project root (reads `results.tsv` and `runs/`):

```bash
python3 webui/server.py
```

Then open http://localhost:8080. The page polls every 5 seconds, so new results appear while the agent runs.

Mock data, for working on the UI before real runs exist:

```bash
python3 webui/make_mock.py
```

```bash
python3 webui/server.py --root webui/mock
```

Flags: `--root` (default: the parent of `webui/`), `--port` (default `8080`), and `--host` (default `127.0.0.1`; use `0.0.0.0` to open it from another device). `make_mock.py --upto N` writes only the first N experiments. Run it again with a higher N to watch results arrive live. `--no-pending` skips the unlogged run.

## What's on screen

- **Header:** baseline → best val_loss, experiment counts, and a live status. The status shows "Training #N · 34% · 1m 20s left", read from `run.log`, while a run trains. It shows "#N finished, waiting for the agent to log it" once the run folder exists but `results.tsv` has no row for it yet.
- **Progress** (`P`): val_loss per experiment in the style of Karpathy's progress.png. Kept runs are green dots, discarded runs grey dots, crashes a red ×, and the running best a green step line. Runs far above the baseline sit as triangles on the top edge. Below the chart is the experiment log, newest first, with the change against the baseline and against the best result before that run. Clicking a run, in the chart or the log, shows its 16 benchmark samples, the seed-0 denoising strip, the planner's `idea.json` if there is one, and its `change.diff`.
- **Compare** (`C`): two runs' samples and trajectories side by side. It defaults to baseline vs current best and keeps following the best until you pick other runs.

Keys: `↑`/`↓` step through runs, `B` jumps to the best run, `L` follows the latest run again, `T` switches light/dark. Use Cmd +/− to scale everything for the projector.

## API

`GET /api/runs` returns `results.tsv` joined with each `runs/<run_id>/metrics.json`:

- `experiments[]`: one per row, in order, with `status`, `val_loss` (null for crashes), `delta_baseline`, `delta_best` (against the best kept result before this row), `best_after` (the running best), `metrics`, `idea`, and `files` (URLs for samples.png, trajectory.png, change.diff, train.py, ...).
- `pending[]`: run folders newer than every logged run that have no row yet.
- `live`: the progress parsed from `run.log` if it was written in the last 90 s.
- `summary`: baseline, best, best_idx, and counts.

Run files are served at `/runs/<run_id>/<file>`, for the whitelisted names only.
