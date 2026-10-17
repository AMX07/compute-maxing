"""
Build a fake project root at webui/mock/ so the web UI can be developed before real runs exist.

    python3 webui/make_mock.py            # 12 logged experiments + 1 pending run
    python3 webui/make_mock.py --upto 6   # only the first 6 logged (re-run with a higher number to simulate progress)

Writes webui/mock/results.tsv, webui/mock/runs/<run_id>/ and webui/mock/run.log.
Standard library only: PNGs are written with zlib + struct.
"""

import argparse
import difflib
import json
import math
import os
import random
import shutil
import struct
import time
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
MOCK = os.path.join(HERE, "mock")

SIZE, PAD, UPSCALE, COLS, FRAMES = 32, 2, 4, 4, 8  # same layout as inference.py

# ---------------------------------------------------------------------------
# A fake train.py; each experiment edits it so change.diff and train.py snapshots look real.
# ---------------------------------------------------------------------------

BASE_TRAIN = '''"""
Autoresearch image-model pretraining script (mock snapshot written by webui/make_mock.py).
A text-conditioned diffusion transformer (DiT) trained with rectified flow on CIFAR-10.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from prepare import IMAGE_SIZE, TIME_BUDGET, tokenize, make_dataloader, evaluate_loss, save_run

@dataclass
class ModelConfig:
    image_size: int = 32
    patch_size: int = 2
    n_layer: int = 6
    n_head: int = 4
    n_embd: int = 256
    qk_norm: bool = False

BATCH_SIZE = 256
LR = 3e-4
WARMUP_STEPS = 0
WEIGHT_DECAY = 0.0
CFG_DROPOUT = 0.1        # probability of dropping the caption during training

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.n_embd, affine=False)
        self.attn = Attention(cfg)
        self.norm2 = nn.LayerNorm(cfg.n_embd, affine=False)
        self.mlp = MLP(cfg)
        self.ada = nn.Linear(cfg.n_embd, 6 * cfg.n_embd)

    def __call__(self, x, c):
        s1, b1, g1, s2, b2, g2 = mx.split(self.ada(c)[:, None], 6, axis=-1)
        h = self.norm1(x) * (1 + s1) + b1
        x = x + g1 * self.attn(h)
        h = self.norm2(x) * (1 + s2) + b2
        return x + g2 * self.mlp(h)

def sample_t(batch):
    return mx.random.uniform(shape=(batch,))

def loss_fn(model, x, t, tok):
    noise = mx.random.normal(x.shape)
    xt = (1 - t[:, None, None, None]) * x + t[:, None, None, None] * noise
    v = model(xt, t, tok)
    return mx.mean((v - (noise - x)) ** 2)
'''

# (description, status, val_loss, [(old, new), ...], idea or None)
EXPERIMENTS = [
    ("baseline: DiT depth 6, width 256, patch 2, AdamW lr 3e-4", "keep", 0.412306, [], None),
    ("raise lr to 1e-3 with 200 warmup steps", "keep", 0.398127,
     [("LR = 3e-4", "LR = 1e-3"), ("WARMUP_STEPS = 0", "WARMUP_STEPS = 200")], None),
    ("patch size 4: 4x fewer tokens, more steps in the budget", "discard", 0.431044,
     [("patch_size: int = 2", "patch_size: int = 4")], None),
    ("logit-normal timestep sampling (SD3)", "keep", 0.387412,
     [("    return mx.random.uniform(shape=(batch,))",
       "    # logit-normal: spend more steps on mid-range noise levels\n    return mx.sigmoid(mx.random.normal(shape=(batch,)))")],
     {"title": "Logit-normal timestep sampling",
      "rationale": "SD3 found that sampling t from a logit-normal instead of uniform puts more training on the "
                   "middle noise levels, where the velocity target is hardest, and lowers loss at equal compute.",
      "sources": ["https://arxiv.org/abs/2403.03206"]}),
    ("replace AdaLN with an additive time embedding", "discard", 0.405231,
     [("        h = self.norm1(x) * (1 + s1) + b1", "        h = self.norm1(x) + c[:, None]"),
      ("        h = self.norm2(x) * (1 + s2) + b2", "        h = self.norm2(x) + c[:, None]")], None),
    ("fused attention via mx.fast.scaled_dot_product_attention in bf16", "crash", None,
     [("        x = x + g1 * self.attn(h)", "        x = x + g1 * self.attn(h.astype(mx.bfloat16))")], None),
    ("zero-init AdaLN gates so every block starts as the identity", "keep", 0.379245,
     [("        self.ada = nn.Linear(cfg.n_embd, 6 * cfg.n_embd)",
       "        self.ada = nn.Linear(cfg.n_embd, 6 * cfg.n_embd)\n"
       "        self.ada.weight = mx.zeros_like(self.ada.weight)\n"
       "        self.ada.bias = mx.zeros_like(self.ada.bias)")],
     {"title": "adaLN-Zero initialisation",
      "rationale": "DiT's best block variant zero-initialises the modulation so each residual block starts as the "
                   "identity. It trained faster than in-context or cross-attention conditioning.",
      "sources": ["https://arxiv.org/abs/2212.09748"]}),
    ("weight decay 0.1 on linear layers", "discard", 0.380810,
     [("WEIGHT_DECAY = 0.0", "WEIGHT_DECAY = 0.1")], None),
    ("min-SNR loss weighting, gamma 5", "keep", 0.365518,
     [("    return mx.mean((v - (noise - x)) ** 2)",
       "    snr = ((1 - t) / mx.maximum(t, 1e-4)) ** 2\n"
       "    w = mx.minimum(snr, 5.0) / (snr + 1)\n"
       "    return mx.mean(w[:, None, None, None] * (v - (noise - x)) ** 2)")],
     {"title": "Min-SNR-gamma loss weighting",
      "rationale": "Timesteps compete for gradient. Clamping the per-timestep weight at SNR 5 stops the easy, "
                   "low-noise steps from dominating and converged 3.4x faster in the paper.",
      "sources": ["https://arxiv.org/abs/2303.09556", "https://github.com/TiankaiHang/Min-SNR-Diffusion-Training"]}),
    ("double depth to 12 layers", "discard", 0.442019,
     [("n_layer: int = 6", "n_layer: int = 12")], None),
    ("batch size 128: twice the optimizer steps in the budget", "keep", 0.356104,
     [("BATCH_SIZE = 256", "BATCH_SIZE = 128")], None),
    ("caption dropout 0.1 to 0.2 for stronger guidance", "discard", 0.359033,
     [("CFG_DROPOUT = 0.1 ", "CFG_DROPOUT = 0.2 ")], None),
]

PENDING = ("QK-norm in attention", [("qk_norm: bool = False", "qk_norm: bool = True")],
           {"title": "QK-norm",
            "rationale": "Normalising queries and keys before the dot product keeps attention logits bounded, which "
                         "lets the higher learning rate train stably and may allow raising it again.",
            "sources": ["https://arxiv.org/abs/2302.05442"]})


def apply_edits(text, edits):
    for old, new in edits:
        assert old in text, old
        text = text.replace(old, new, 1)
    return text


def unified(old, new):
    lines = difflib.unified_diff(old.splitlines(True), new.splitlines(True), "a/train.py", "b/train.py", n=3)
    return "diff --git a/train.py b/train.py\n" + "".join(lines)


# ---------------------------------------------------------------------------
# Images: a procedural "horse in a field" per seed, blurred and noised by how bad val_loss is.
# ---------------------------------------------------------------------------

def scene(seed):
    rng = random.Random(seed * 7919 + 13)
    sky_top = [rng.uniform(0.35, 0.6), rng.uniform(0.55, 0.75), rng.uniform(0.8, 0.95)]
    sky_bot = [min(1, c + 0.25) for c in sky_top]
    ground_y = rng.randint(19, 24)
    grass = [rng.uniform(0.25, 0.45), rng.uniform(0.45, 0.6), rng.uniform(0.15, 0.25)]
    coat = rng.choice([(0.45, 0.27, 0.13), (0.25, 0.15, 0.08), (0.62, 0.42, 0.22), (0.85, 0.82, 0.76), (0.12, 0.1, 0.09)])
    face = rng.choice([-1, 1])
    cx, cy = rng.uniform(13, 19), ground_y - rng.uniform(7, 9)
    rx, ry = rng.uniform(6, 7.5), rng.uniform(2.6, 3.4)
    hx, hy = cx + face * (rx + 1.5), cy - ry - 2.5
    px = []
    for y in range(SIZE):
        row = []
        for x in range(SIZE):
            if y < ground_y:
                f = y / ground_y
                c = [a * (1 - f) + b * f for a, b in zip(sky_top, sky_bot)]
            else:
                c = [g * (1 - 0.02 * (y - ground_y)) for g in grass]
            body = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1
            head = ((x - hx) / 2.0) ** 2 + ((y - hy) / 1.5) ** 2 <= 1
            neck = abs((x - (cx + face * rx * 0.75)) - face * (cy - y) * 0.55) < 1.3 and hy <= y <= cy
            leg = cy < y <= ground_y + 1 and any(abs(x - (cx + o * rx)) < 0.7 for o in (-0.7, -0.45, 0.45, 0.7))
            if body or head or neck or leg:
                shade = 0.85 if (leg or y > cy + 1) else 1.0
                c = [k * shade for k in coat]
            row.append(c)
        px.append(row)
    return px


def degrade(img, badness, seed):
    """badness 0 = clean, 1 = mostly noise."""
    rng = random.Random(seed * 104729 + int(badness * 1e4))
    blur = 1 + int(badness * 3)
    out = []
    for y in range(SIZE):
        row = []
        for x in range(SIZE):
            acc = [0.0, 0.0, 0.0]
            n = 0
            for dy in range(-blur + 1, blur):
                for dx in range(-blur + 1, blur):
                    yy, xx = min(SIZE - 1, max(0, y + dy)), min(SIZE - 1, max(0, x + dx))
                    for k in range(3):
                        acc[k] += img[yy][xx][k]
                    n += 1
            row.append([a / n + rng.gauss(0, 0.35 * badness) for a in acc])
        out.append(row)
    return out


def noisy_mix(img, t, seed):
    """Rectified-flow interpolation between the image (t=0) and pure noise (t=1), in [-1, 1] space."""
    rng = random.Random(seed)
    return [[[((1 - t) * (2 * c - 1) + t * rng.gauss(0, 1) + 1) / 2 for c in p] for p in row] for row in img]


def write_png(path, width, height, rows_rgba):
    raw = b"".join(b"\x00" + bytes(r) for r in rows_rgba)
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(png)


def tile(images, cols, path):
    """Same layout as inference.tile: transparent 2px gaps, nearest-neighbour upscale."""
    n = len(images)
    rows = (n + cols - 1) // cols
    w, h = cols * (SIZE + PAD) - PAD, rows * (SIZE + PAD) - PAD
    canvas = [[(0, 0, 0, 0)] * w for _ in range(h)]
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        oy, ox = r * (SIZE + PAD), c * (SIZE + PAD)
        for y in range(SIZE):
            for x in range(SIZE):
                canvas[oy + y][ox + x] = tuple(int(max(0, min(1, v)) * 255 + 0.5) for v in img[y][x]) + (255,)
    out = []
    for row in canvas:
        line = bytearray()
        for p in row:
            line.extend(bytes(p) * UPSCALE)
        out.extend([line] * UPSCALE)
    write_png(path, w * UPSCALE, h * UPSCALE, out)


# ---------------------------------------------------------------------------

def run_id_at(t):
    return time.strftime("%Y%m%d-%H%M%S", time.localtime(t))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--upto", type=int, default=len(EXPERIMENTS), help="how many experiments are logged in results.tsv")
    ap.add_argument("--no-pending", action="store_true", help="skip the unlogged (pending) run")
    args = ap.parse_args()
    upto = max(0, min(args.upto, len(EXPERIMENTS)))

    shutil.rmtree(MOCK, ignore_errors=True)
    os.makedirs(os.path.join(MOCK, "runs"))
    scenes = [scene(s) for s in range(16)]

    now = time.time()
    lt = time.localtime(now)
    start = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 11, 0, 5, 0, 0, -1))  # stable run ids: 11:00, then every 7 min
    kept_text = BASE_TRAIN
    lines = ["commit\tval_loss\tmemory_gb\tstatus\trun_id\tdescription"]

    def make_run(i, desc, status, val_loss, text, prev_text, idea, t, with_samples=True):
        rid = run_id_at(t)
        d = os.path.join(MOCK, "runs", rid)
        os.makedirs(d)
        commit = "%07x" % random.Random(i).getrandbits(28)
        rng = random.Random(i + 100)
        metrics = {
            "run_id": rid, "commit": commit, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t)),
            "val_loss": val_loss, "training_seconds": 300.0 + round(rng.uniform(0, 1.5), 1),
            "total_seconds": round(318 + rng.uniform(0, 12), 1),
            "peak_memory_gb": round(rng.uniform(7.5, 12.5) * (2 if "12 layers" in desc else 1), 1),
            "num_steps": int(4200 * (2 if "batch size 128" in desc else 1) * (0.55 if "12 layers" in desc else 1)
                             + rng.uniform(-80, 80)),
            "num_params_M": 22.71 if "12 layers" in desc else 11.42,
        }
        with open(os.path.join(d, "train.py"), "w") as f:
            f.write(text)
        if text != prev_text:
            with open(os.path.join(d, "change.diff"), "w") as f:
                f.write(unified(prev_text, text))
        if idea:
            with open(os.path.join(d, "idea.json"), "w") as f:
                json.dump(idea, f, indent=2)
        for junk in ("model.safetensors", "config.json"):
            open(os.path.join(d, junk), "w").close()
        if with_samples:
            badness = max(0.03, min(1.0, (val_loss - 0.345) / 0.09))
            imgs = [degrade(scenes[s], badness, s + 1000 * i) for s in range(16)]
            tile(imgs, COLS, os.path.join(d, "samples.png"))
            frames = [noisy_mix(imgs[0], 1 - k / (FRAMES - 1), 42 + i) for k in range(FRAMES)]
            tile(frames, FRAMES, os.path.join(d, "trajectory.png"))
            metrics.update(prompt="a photo of a horse", sample_seconds=round(rng.uniform(4, 9), 1), sample_over_limit=False)
        with open(os.path.join(d, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        return rid, commit, metrics

    for i, (desc, status, val_loss, edits, idea) in enumerate(EXPERIMENTS[:upto]):
        text = apply_edits(kept_text, edits)
        t = start + i * 7 * 60
        if status == "crash":   # crashed before save_run, so no run dir
            lines.append(f"{'%07x' % random.Random(i).getrandbits(28)}\t0.000000\t0.0\tcrash\t-\t{desc}")
            continue
        rid, commit, m = make_run(i, desc, status, val_loss, text, kept_text, idea, t)
        lines.append(f"{commit}\t{val_loss:.6f}\t{m['peak_memory_gb']:.1f}\t{status}\t{rid}\t{desc}")
        if status == "keep":
            kept_text = text

    with open(os.path.join(MOCK, "results.tsv"), "w") as f:
        f.write("\n".join(lines) + "\n")

    if not args.no_pending:
        desc, edits, idea = PENDING
        # Saved and sampled, but the agent has not logged it to results.tsv yet.
        make_run(upto, desc, "pending", 0.351877, apply_edits(kept_text, edits), kept_text, idea, now - 20)

    # run.log: the agent redirects train.py output here. A stale one shows the UI's "training" strip is off.
    with open(os.path.join(MOCK, "run.log"), "w") as f:
        f.write("Parameters: 11.42M\nTime budget: 300s\n"
                "\rstep 04180 (99.6%) | loss: 0.352114 | lrm: 0.02 | dt: 71ms | img/sec: 1,803 | epoch: 10 | remaining: 1s    \n"
                "---\nval_loss:         0.351877\n")

    print(f"Wrote {upto} logged experiments{'' if args.no_pending else ' + 1 pending run'} to {MOCK}")


if __name__ == "__main__":
    main()
