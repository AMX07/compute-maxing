# CS336 copied-project example

This metadata connects a separately copied CS336 assignment to Turing's shared Baseline, Eval, Agents, and training views. It targets a CS336 implementation with `assignment1-basics/cs336_basics/pretrain.py` and `gpt.py`, not every version of the course repository. No assignment source, dataset, credentials, checkpoints, or research results are included here.

Use a copy of the course directory and preserve the original. The model implementation must expose `hyperparams`, `main`, and `data_loader` as expected by `webui/research_runner.py`; inspect that adapter before applying this example to a different implementation.

## Setup

1. Copy `project.example.json` to the copied directory's `.turing/project.json`, and copy `architecture.json` and `evaluation.json` alongside it. Do not overwrite an existing project configuration or history without merging its metadata.
2. Replace every `/absolute/path/to/cs336-copy` with the absolute path of your copied course directory, and every `/absolute/path/to/turing` with the absolute path of this Turing checkout. These are literal placeholders, not environment variables. Keep `{parameters}` and `{output}` unchanged; Turing supplies those per experiment.
3. Prepare the copied assignment's environment and data. Reinstall its editable package in the copy so it does not point at the original directory:

   ```sh
   uv sync --project /absolute/path/to/cs336-copy/assignment1-basics --reinstall-package cs336_basics
   ```

   The configured protocol expects `out/tinystories_vocab.pkl`, `out/tinystories_merges.pkl`, `out/tinystories_train.npy`, and `out/tinystories_valid.npy`, plus the listed source and dependency files. Use the assignment's preparation instructions to create them. This example uses the MPS device; choose an available device in `research.fixed_parameters.device` before running elsewhere.
4. Create `.turing/training.env` inside the copy containing `WANDB_MODE=offline`. No W&B credential is required. The standalone training command reads this local file; the research adapter also sets offline mode itself.
5. Configure Crusoe using Turing's root `.env` as described in `harness/README.md`, then open the copied metadata directory:

   ```sh
   uv run --no-project python /absolute/path/to/turing/webui/server.py --root /absolute/path/to/cs336-copy/.turing
   ```

In Agents, the standalone launch directory should be the copied `assignment1-basics` directory. Use **Check setup** before **Start training**, or start a bounded session with **Start agentic research**.

## What the example measures

The architecture and evaluation descriptions record this implementation's source defaults: a decoder-only transformer with four pre-norm blocks, width 512, 16 attention heads, SwiGLU width 1,344, and a vocabulary of 10,000. The diagram's connections were verified against `transformer_lm` and `transformer_block` in `gpt.py`. Validate these facts against your copied source before using the map.

Standalone defaults use 2,000 training steps and batch size 200. Research uses a separate 200-step screening protocol: batch size 32, context 256, seed 42, ten validation batches every 50 steps, and fixed train/validation sampling windows. Model dimensions and optimizer settings are bounded by `project.example.json`; final validation loss is minimized within each session's own comparison series.

Research writes parameters, metrics, checkpoints, and activity under `.turing/research/`, and keeps the best settings in `research/best.json`. It does not rewrite the model source or apply those settings to standalone defaults. Standalone outputs remain in the copied assignment. The original course directory is outside the configured source, data, and output paths.
