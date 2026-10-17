"""
Benchmark image generation for a finished run. Fixed, not modified by the agent.
Loads runs/<run_id>/ (weights + the exact train.py that produced them), renders EVAL_PROMPT
from EVAL_SEEDS fixed noise seeds, and saves samples.png and trajectory.png next to the checkpoint.

Usage: uv run inference.py [run_dir]   (default: latest run)
"""

import os
import sys
import json
import time
import importlib.util

import numpy as np
import mlx.core as mx
from PIL import Image

from prepare import IMAGE_SIZE, EVAL_PROMPT, EVAL_SEEDS, INFERENCE_TIME_LIMIT, tokenize, update_metrics, latest_run_dir

GRID_COLS = 4           # samples.png is a 4x4 grid
UPSCALE = 4             # nearest-neighbour upscale so 32x32 images are visible
TRAJECTORY_FRAMES = 8   # frames from noise to image in trajectory.png (seed 0)


def load_run(run_dir):
    """Import the run's own train.py snapshot and load its weights. Returns (model, sample_fn)."""
    name = "run_" + os.path.basename(os.path.normpath(run_dir)).replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, os.path.join(run_dir, "train.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with open(os.path.join(run_dir, "config.json")) as f:
        config = module.ModelConfig(**json.load(f))
    model = module.Model(config)
    model.load_weights(os.path.join(run_dir, "model.safetensors"))
    model.eval()
    return model, module.sample


def to_uint8(x):
    return np.array(mx.round((mx.clip(x.astype(mx.float32), -1, 1) + 1) * 127.5)).astype(np.uint8)


def tile(images, cols, pad=2):
    """(N, H, W, 3) uint8 -> RGBA grid with transparent gaps, upscaled with nearest neighbour."""
    n, h, w, _ = images.shape
    rows = (n + cols - 1) // cols
    canvas = np.zeros((rows * (h + pad) - pad, cols * (w + pad) - pad, 4), dtype=np.uint8)
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        y, x = r * (h + pad), c * (w + pad)
        canvas[y : y + h, x : x + w, :3] = img
        canvas[y : y + h, x : x + w, 3] = 255
    image = Image.fromarray(canvas, "RGBA")
    return image.resize((image.width * UPSCALE, image.height * UPSCALE), Image.NEAREST)


def generate(run_dir, model=None, sample_fn=None):
    """Render the benchmark prompt for a run. Returns sample_seconds."""
    if model is None:
        model, sample_fn = load_run(run_dir)
    tokens = tokenize([EVAL_PROMPT] * EVAL_SEEDS)
    noise = mx.stack([mx.random.normal((IMAGE_SIZE, IMAGE_SIZE, 3), key=mx.random.key(s)) for s in range(EVAL_SEEDS)])
    mx.random.seed(0)  # for stochastic samplers
    t0 = time.time()
    states = sample_fn(model, tokens, noise)
    mx.eval(states)
    sample_seconds = time.time() - t0

    tile(to_uint8(states[-1]), GRID_COLS).save(os.path.join(run_dir, "samples.png"))
    frames = np.linspace(0, len(states) - 1, TRAJECTORY_FRAMES).round().astype(int)
    tile(np.stack([to_uint8(states[i][0]) for i in frames]), TRAJECTORY_FRAMES).save(os.path.join(run_dir, "trajectory.png"))

    over_limit = sample_seconds > INFERENCE_TIME_LIMIT
    update_metrics(run_dir, prompt=EVAL_PROMPT, sample_seconds=round(sample_seconds, 1), sample_over_limit=over_limit)
    if over_limit:
        print(f"WARNING: sampling took {sample_seconds:.1f}s, over the {INFERENCE_TIME_LIMIT}s limit")
    return sample_seconds


if __name__ == "__main__":
    run_dir = sys.argv[1] if len(sys.argv) > 1 else latest_run_dir()
    sample_seconds = generate(run_dir)
    print(f"sample_seconds:   {sample_seconds:.1f}")
    print(f"samples:          {os.path.join(run_dir, 'samples.png')}")
