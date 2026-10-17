# autoresearch (image edition)

The idea, borrowed from [karpathy/autoresearch](https://github.com/karpathy/autoresearch): give an AI agent a small but real text-to-image training setup and let it experiment autonomously. It modifies the code, trains for 2 minutes, checks if the result improved, keeps or discards, and repeats. After every run the same prompt, "a photo of a horse", is rendered from the same 16 seeds, so you can watch the images get better as the architecture does. Everything trains and runs on one MacBook (M5 Max) with [MLX](https://github.com/ml-explore/mlx). As in upstream, you don't touch the Python files like you normally would as a researcher. Instead you program `program.md`, the instructions for the research agents.

## How it works

The repo is deliberately kept small and only really has four files that matter:

- **`prepare.py`** — fixed constants, one-time data prep (downloads CIFAR-10, captions it), and runtime utilities (byte-level tokenizer, dataloader, evaluation, run saving). Not modified.
- **`train.py`** — the single file the agent edits. Contains the full model, a text-conditioned diffusion transformer (DiT) with a small text encoder, plus rectified-flow training, AdamW, and the sampler. Everything is fair game: architecture, conditioning, noise schedule, optimizer, hyperparameters, batch size, sampler. **This file is edited and iterated on by the agent**.
- **`inference.py`** — renders the benchmark prompt for a finished run, from the run's own copy of `train.py` and its weights. Not modified.
- **`program.md`** — baseline instructions for one agent. **This file is edited and iterated on by the human**.

By design, training runs for a **fixed 2-minute time budget** (wall clock, excluding startup/compilation). The metric is **val_loss**: the rectified-flow velocity MSE on 1,024 held-out CIFAR-10 test images at 8 fixed timesteps with fixed noise. Lower is better. Any architecture can be compared fairly, as long as the model predicts velocity.

Every run is saved to `runs/<run_id>/`, whether it was kept or not:

```
model.safetensors   weights
config.json         model config
train.py            exact code (architecture) that produced the weights
metrics.json        val_loss, steps, memory, timings
samples.png         16 images for the benchmark prompt, fixed seeds, 4x upscaled
trajectory.png      seed 0 going from noise to image
change.diff         train.py vs the previous commit (when git is available)
```

## Quick start

**Requirements:** An Apple Silicon Mac, Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install dependencies
uv sync

# 2. Download CIFAR-10 and build captions (one-time, ~1 min)
uv run prepare.py

# 3. Manually run a single training experiment (~2.5 min, includes eval and sampling)
uv run train.py

# 4. Re-render the benchmark prompt for any run
uv run inference.py runs/<run_id>
```

## Running the agent

`program.md` works with any coding agent, as in upstream: point it at `program.md` and say "let's kick off a new experiment". Our own harness with dedicated search and planning agents is next.

## Web UI

`python3 webui/server.py` serves a presentation view at http://localhost:8080: the progress chart, the ideas that were tried, and each run's images side by side.

## Project structure

```
prepare.py      — constants, data prep + runtime utilities (do not modify)
train.py        — model, sampler, optimizer, training loop (agent modifies this)
inference.py    — benchmark image generation (do not modify)
program.md      — agent instructions
pyproject.toml  — dependencies
runs/           — one folder per experiment (untracked)
results.tsv     — experiment log (untracked)
webui/          — presentation UI
```

## Design choices

- **Single file to modify.** The agent only touches `train.py`. This keeps the scope manageable and diffs reviewable.
- **Fixed time budget.** Training always runs for exactly 2 minutes, so runs are comparable whatever the agent changes (model size, batch size, patch size, architecture). About 24 experiments an hour.
- **One fixed prompt.** Same prompt and seeds every run, so differences in the images come from the model, not the sampling noise.
- **Small data.** CIFAR-10 at 32x32 is the image-world equivalent of TinyStories: low entropy, so a model trained for two minutes produces recognisable images, and architecture changes show up in both val_loss and the pictures.
- **Self-contained.** MLX, NumPy and Pillow. One Mac, one file, one metric.

## Differences from karpathy/autoresearch

| | karpathy/autoresearch | this repo |
|---|---|---|
| Model | GPT language model (nanochat) | Text-conditioned diffusion transformer (DiT, adaLN-Zero) with a byte-level text encoder |
| Data | Text shards, BPE tokenizer | CIFAR-10 32x32, captioned from class names with templates ("a blurry photo of a puppy"), byte-level tokenizer |
| Objective | Next-token prediction | Rectified flow (velocity prediction) with classifier-free guidance |
| Hardware | One NVIDIA GPU (H100), PyTorch + CUDA | One MacBook M5 Max (128 GB unified memory), MLX |
| Metric | val_bpb | val_loss: velocity MSE on fixed held-out images, timesteps and noise |
| Time budget | 5 minutes | 2 minutes, so a hackathon afternoon fits ~60 experiments |
| Evaluation output | One number | One number plus 16 images of a fixed prompt ("a photo of a horse") and a denoising trajectory, at most 2 minutes to generate |
| Files | `prepare.py`, `train.py`, `program.md` | Adds `inference.py` (fixed) and `webui/` |
| Artifacts | Kept commits on a branch | Every run saved in `runs/<run_id>/` with weights, the exact `train.py`, images and metrics, even if discarded |
| `train.py` layout | Flat script | Same layout, but the training loop is inside `main()` so `inference.py` can import a run's model and sampler |
| `results.tsv` | 5 columns | 6 columns: adds `run_id`, linking each row to its images |
| Agent harness | Claude Code / Codex reading `program.md` | Our own harness with dedicated agents for search, planning, etc. (in progress) |

## License

MIT. The concept, loop and `program.md` structure are adapted from [karpathy/autoresearch](https://github.com/karpathy/autoresearch) (MIT). The MLX code here is new.
