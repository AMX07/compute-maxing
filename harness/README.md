# harness

Our own agent harness for the autoresearch loop. It replaces "point Claude Code or Codex at program.md" with a small, inspectable program, and every LLM call goes to Crusoe Managed Inference.

## Run it

1. Put your Crusoe settings in `.env` at the project root. The file is git-ignored; see `.env.example`.
2. Check the connection with `python3 -m harness.llm`, which should reply "ready".
3. Start the loop:

```bash
python3 harness/run.py --until 16:15
```

`--until` stops starting new experiments after that local time. `touch harness/STOP` finishes the current experiment and then stops. `--max-iters 1` runs a single experiment. The first iteration runs the unmodified `train.py` as the baseline.

## One iteration (`harness/step.py`)

Each experiment runs in a fresh process, and all state lives in files. A crash costs one iteration, and edits to the prompts or code take effect on the next one.

1. **Plan:** the planner (`prompts/planner.md`) reads `program.md`, the playbook, the experiment log with past predictions, and the current best `train.py`. It replies with an idea (title, category, rationale, a predicted val_loss change) and SEARCH/REPLACE edits.
2. **Review and apply:** the edits must match exactly and pass the guards below. A reviewer (`prompts/reviewer.md`) then checks that the diff actually implements the idea; it looks for inverted schedules, sign errors, no-ops and likely crashes. A rejected reply gets up to two retries with the reason. Then `train.py` is committed as `exp N: ...`. The reviewer was added after experiment 1, whose cosine schedule was inverted and taught the loop a false lesson.
3. **Run:** `uv run train.py > run.log`, killed after 6 minutes. On a crash, the fixer (`prompts/fixer.md`) gets one repair attempt with the end of the log.
4. **Decide:** code decides, not the model. The change is kept if val_loss beats the best so far; otherwise `train.py` is reverted with a new commit. History only moves forward, so discarded ideas stay in git.
5. **Log:** the row is appended to `results.tsv`, and `idea.json` (idea, prediction, outcome) and `change.diff` are written into `runs/<run_id>/` for the web UI.
6. **Reflect:** the reflector (`prompts/reflector.md`) reads the diff that ran and compares the prediction with the outcome. If the code didn't implement the idea, it records a lesson about the bug rather than the idea. It makes at most 3 itemized changes to the playbook (`memory/playbook.json`, also rendered as `playbook.md`), and code merges them.

## The evaluator stays outside the loop

These checks are enforced in code, not by asking the model:

- `prepare.py` and `inference.py` are hashed when the loop starts. If either changes, the loop refuses to run.
- An edit is rejected if it redefines or reassigns anything imported from `prepare` (`TIME_BUDGET`, `evaluate_loss`, ...), reads the test split, removes the final `evaluate_loss`/`save_run` calls, or lowers `CAPTION_DROPOUT` below 0.1. A lower dropout improves val_loss but breaks classifier-free guidance.
- A run is discarded if training takes more than the time budget plus 100 s of wall clock, or if sampling takes more than 120 s.

## Files

| Path | What it holds | In git |
|---|---|---|
| `memory/state.json` | Best commit and val_loss, and every idea with its prediction and outcome | yes |
| `memory/playbook.json`, `playbook.md` | Lessons learned | yes |
| `traces/` | Every LLM call (prompt, reply, timing, usage), the live status file, and loop logs | no |
