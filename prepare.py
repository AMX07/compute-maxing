"""
One-time data prep and fixed runtime utilities for autoresearch (image edition).
Downloads CIFAR-10, captions it, and provides the tokenizer, dataloader,
evaluation and run saving. Not modified by the agent.

Usage: uv run prepare.py
"""

import os
import sys
import json
import time
import shutil
import tarfile
import subprocess
import urllib.request

import numpy as np
import mlx.core as mx

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

IMAGE_SIZE = 32             # images are (IMAGE_SIZE, IMAGE_SIZE, 3), NHWC, values in [-1, 1]
TIME_BUDGET = 120           # training time budget in seconds (wall clock, excluding compilation)
TEXT_LEN = 48               # prompt length in tokens (BOS + UTF-8 bytes, padded)
EVAL_IMAGES = 1024          # held-out images used by evaluate_loss
EVAL_TIMESTEPS = 8          # fixed t values per held-out image in evaluate_loss
EVAL_PROMPT = "a photo of a horse"  # the benchmark prompt, rendered after every run
EVAL_SEEDS = 16             # benchmark images per run (noise seeds 0..15)
INFERENCE_TIME_LIMIT = 120  # seconds allowed to generate the benchmark images

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(ROOT, "runs")
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch-image")
CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
CLASS_WORDS = {
    "airplane": ["airplane", "plane", "jet"],
    "automobile": ["car", "automobile"],
    "bird": ["bird"],
    "cat": ["cat", "kitten"],
    "deer": ["deer"],
    "dog": ["dog", "puppy"],
    "frog": ["frog"],
    "horse": ["horse"],
    "ship": ["ship", "boat"],
    "truck": ["truck"],
}
TEMPLATES = [
    "a photo of {}", "a blurry photo of {}", "a close-up photo of {}", "a low resolution photo of {}",
    "a picture of {}", "an image of {}", "a cropped photo of {}", "{}",
]

# ---------------------------------------------------------------------------
# Tokenizer (byte-level)
# ---------------------------------------------------------------------------

PAD = 256
BOS = 257
VOCAB_SIZE = 258

def tokenize(texts):
    """List of strings -> int32 array (len(texts), TEXT_LEN): BOS, UTF-8 bytes, then PAD. "" is the null prompt."""
    out = np.full((len(texts), TEXT_LEN), PAD, dtype=np.int32)
    for i, text in enumerate(texts):
        b = list(text.encode("utf-8"))[: TEXT_LEN - 1]
        out[i, 0] = BOS
        out[i, 1 : 1 + len(b)] = b
    return mx.array(out)

def _caption_table():
    """All captions, grouped by class: (texts, start index per class, count per class)."""
    texts, starts, counts = [], [], []
    for cls in CLASSES:
        starts.append(len(texts))
        for word in CLASS_WORDS[cls]:
            article = "an" if word[0] in "aeiou" else "a"
            texts.extend(t.format(f"{article} {word}") for t in TEMPLATES)
        counts.append(len(texts) - starts[-1])
    return texts, np.array(starts), np.array(counts)

CAPTIONS, _CAPTION_START, _CAPTION_COUNT = _caption_table()
_CAPTION_TOKENS = np.array(tokenize(CAPTIONS))

def caption_tokens(labels, rng):
    """A random caption for each label -> int32 numpy array (len(labels), TEXT_LEN)."""
    idx = _CAPTION_START[labels] + (rng.random(len(labels)) * _CAPTION_COUNT[labels]).astype(np.int64)
    return _CAPTION_TOKENS[idx]

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def download_data():
    """Download CIFAR-10 (binary version, ~163MB) and store train/test as uint8 NHWC arrays."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    if all(os.path.exists(os.path.join(CACHE_DIR, f"{s}.npz")) for s in ("train", "test")):
        print(f"Data: already prepared at {CACHE_DIR}")
        return
    tar_path = os.path.join(CACHE_DIR, "cifar-10-binary.tar.gz")
    if not os.path.exists(tar_path):
        print(f"Data: downloading {CIFAR_URL}")
        urllib.request.urlretrieve(CIFAR_URL, tar_path + ".tmp")
        os.rename(tar_path + ".tmp", tar_path)
    splits = {"train": [f"data_batch_{i}.bin" for i in range(1, 6)], "test": ["test_batch.bin"]}
    with tarfile.open(tar_path) as tar:
        for split, names in splits.items():
            raw = np.concatenate([
                np.frombuffer(tar.extractfile(f"cifar-10-batches-bin/{n}").read(), dtype=np.uint8).reshape(-1, 3073)
                for n in names
            ])
            labels = raw[:, 0].astype(np.int64)
            images = raw[:, 1:].reshape(-1, 3, IMAGE_SIZE, IMAGE_SIZE).transpose(0, 2, 3, 1)
            np.savez(os.path.join(CACHE_DIR, f"{split}.npz"), images=images, labels=labels)
            print(f"Data: {split} {images.shape}")

def load_split(split):
    data = np.load(os.path.join(CACHE_DIR, f"{split}.npz"))
    return data["images"], data["labels"]

def make_dataloader(batch_size, split="train", seed=0):
    """Infinite iterator of (images, tokens, epoch). images: float32 (B, 32, 32, 3) in [-1, 1]. Train split is h-flipped at random."""
    images, labels = load_split(split)
    rng = np.random.default_rng(seed)
    epoch = 1
    while True:
        perm = rng.permutation(len(images))
        for i in range(0, len(perm) - batch_size + 1, batch_size):
            idx = perm[i : i + batch_size]
            x = images[idx]
            if split == "train":
                flip = rng.random(batch_size) < 0.5
                x = np.where(flip[:, None, None, None], x[:, :, ::-1], x)
            x = x.astype(np.float32) / 127.5 - 1.0
            yield mx.array(x), mx.array(caption_tokens(labels[idx], rng)), epoch
        epoch += 1

# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE - this is the fixed metric)
# ---------------------------------------------------------------------------

def evaluate_loss(model, batch_size=256):
    """
    Rectified-flow velocity MSE on held-out images. Lower is better.
    Convention: x_t = (1 - t) * x + t * noise, target v = noise - x, t in (0, 1), t=1 is pure noise.
    model(x_t, t, tokens) must return v with the same shape as x_t. Images, captions, noise and t are fixed.
    """
    images, labels = load_split("test")
    images, labels = images[:EVAL_IMAGES], labels[:EVAL_IMAGES]
    tokens = caption_tokens(labels, np.random.default_rng(1234))
    ts = [(j + 0.5) / EVAL_TIMESTEPS for j in range(EVAL_TIMESTEPS)]
    total, count = 0.0, 0
    for i in range(0, EVAL_IMAGES, batch_size):
        x = mx.array(images[i : i + batch_size].astype(np.float32) / 127.5 - 1.0)
        tok = mx.array(tokens[i : i + batch_size])
        for j, t in enumerate(ts):
            noise = mx.random.normal(x.shape, key=mx.random.key(1_000_003 * j + i))
            xt = (1 - t) * x + t * noise
            v = model(xt, mx.full((x.shape[0],), t, dtype=mx.float32), tok).astype(mx.float32)
            loss = mx.mean((v - (noise - x)) ** 2)
            total += loss.item() * x.shape[0]
            count += x.shape[0]
    return total / count

# ---------------------------------------------------------------------------
# Runs: every experiment is saved to runs/<run_id>/
# ---------------------------------------------------------------------------

def _git(*args):
    try:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True)
        return out.stdout
    except Exception:
        return None

def save_run(model, config, metrics):
    """Save weights, config, metrics, an exact copy of the running train.py and its diff to runs/<run_id>/."""
    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(RUNS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)
    model.save_weights(os.path.join(run_dir, "model.safetensors"))
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    shutil.copy(os.path.abspath(sys.modules["__main__"].__file__), os.path.join(run_dir, "train.py"))
    diff = _git("diff", "HEAD~1", "--", "train.py")
    if diff:
        with open(os.path.join(run_dir, "change.diff"), "w") as f:
            f.write(diff)
    commit = (_git("rev-parse", "--short=7", "HEAD") or "").strip() or None
    metrics = {"run_id": run_id, "commit": commit, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"), **metrics}
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return run_dir

def update_metrics(run_dir, **kwargs):
    path = os.path.join(run_dir, "metrics.json")
    with open(path) as f:
        metrics = json.load(f)
    metrics.update(kwargs)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)

def latest_run_dir():
    runs = sorted(d for d in os.listdir(RUNS_DIR) if os.path.exists(os.path.join(RUNS_DIR, d, "model.safetensors")))
    return os.path.join(RUNS_DIR, runs[-1])

if __name__ == "__main__":
    download_data()
    images, labels = load_split("train")
    print(f"Captions: {len(CAPTIONS)} (e.g. {CAPTIONS[0]!r}, {CAPTIONS[-1]!r})")
    print(f"Done. Train images: {len(images):,}. Benchmark prompt: {EVAL_PROMPT!r}")
