# autoresearch (image edition)

This is an experiment to have the LLM do its own research on image model architecture.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `oct17`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation, run saving. Do not modify.
   - `inference.py` — fixed benchmark image generation. Do not modify.
   - `train.py` — the file you modify. Model architecture, sampler, optimizer, training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch-image/` contains `train.npz` and `test.npz`. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on the Mac's GPU with MLX. The training script runs for a **fixed time budget of 2 minutes** (wall clock training time, excluding startup/compilation). It then evaluates, saves the run to `runs/<run_id>/`, and renders the benchmark prompt "a photo of a horse" from 16 fixed seeds. You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, text conditioning, timestep sampling, loss weighting, optimizer, hyperparameters, batch size, model size, sampler.

**What you CANNOT do:**
- Modify `prepare.py` or `inference.py`. They are read-only. They contain the fixed evaluation, data loading, tokenizer, benchmark prompt and constants (time budget, image size, etc).
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_loss` function in `prepare.py` is the ground truth metric.
- Break the interface the fixed files rely on. `train.py` must define `ModelConfig`, `Model` and `sample`. `Model(config)(x_t, t, tokens)` returns the velocity `noise - x` for `x_t = (1 - t) * x + t * noise`. `sample(model, tokens, noise)` returns the list of states from noise to image (the last is the final image, values in [-1, 1]). The training code stays inside `main()` so `inference.py` can import a run's `train.py`.

**The goal is simple: get the lowest val_loss.** Since the time budget is fixed, you don't need to worry about training time — it's always 2 minutes. Everything is fair game: change the architecture, the conditioning, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**Memory** is a soft constraint. The GPU shares 128 GB of unified memory with the OS. Some increase is acceptable for meaningful val_loss gains, but it should not blow up dramatically.

**Sampling** must finish within 2 minutes (`sample_seconds`). The keep/discard decision uses val_loss only, and sampler changes (steps, guidance, solver) don't move val_loss, so don't spend runs on sampler-only changes.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 val_loss improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val_loss improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_loss:         0.338795
training_seconds: 120.0
total_seconds:    133.1
peak_memory_gb:   2.4
img_per_sec:      2,990
num_steps:        2814
num_params_M:     11.67
depth:            8
sample_seconds:   0.6
run_dir:          /Users/you/autoresearch-image/runs/20261017-133502
```

The numbers depend on the machine. You can extract the key metric from the log file:

```
grep "^val_loss:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 6 columns:

```
commit	val_loss	memory_gb	status	run_id	description
```

1. git commit hash (short, 7 chars)
2. val_loss achieved (e.g. 0.338795) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 2.4) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. run_id: the last part of `run_dir` (e.g. 20261017-133502) — use `-` for crashes
6. short text description of what this experiment tried

Example:

```
commit	val_loss	memory_gb	status	run_id	description
a1b2c3d	0.338795	2.4	keep	20261017-133502	baseline
b2c3d4e	0.331200	2.4	keep	20261017-133811	logit-normal timestep sampling
c3d4e5f	0.341000	2.6	discard	20261017-134120	SwiGLU MLP
d4e5f6g	0.000000	0.0	crash	-	patch size 1 (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/oct17`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^val_loss:\|^peak_memory_gb:\|^run_dir:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If val_loss improved (lower), you "advance" the branch, keeping the git commit
9. If val_loss is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. Every run, kept or not, stays in `runs/` with its code and images, so nothing is lost when you reset. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~2.5 minutes total (+ a few seconds for startup, eval and sampling). If a run exceeds 6 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers on diffusion and flow-matching transformers, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they are away. If each experiment takes you ~2.5 minutes then you can run approx 24/hour. The user then comes back to experimental results and images, all completed by you while they were gone!
