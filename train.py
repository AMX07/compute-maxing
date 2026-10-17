"""
Autoresearch image-model pretraining script. Single-device, single-file, MLX.
A text-conditioned diffusion transformer (DiT) trained with rectified flow on CIFAR-10.
Usage: uv run train.py
"""

import math
import time
from dataclasses import dataclass, asdict
from functools import partial

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from prepare import IMAGE_SIZE, TEXT_LEN, TIME_BUDGET, PAD, VOCAB_SIZE, tokenize, make_dataloader, evaluate_loss, save_run

# ---------------------------------------------------------------------------
# Diffusion Transformer
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    image_size: int = 32
    channels: int = 3
    patch_size: int = 4
    n_layer: int = 8
    n_head: int = 4
    n_embd: int = 256
    vocab_size: int = 258
    text_len: int = 48
    text_layers: int = 2


def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = mx.exp(-math.log(max_period) * mx.arange(half, dtype=mx.float32) / half)
    args = (1000 * t)[:, None] * freqs[None]
    return mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)


def sincos_pos_embed(grid, dim):
    """Fixed 2D sin-cos position embedding (as in DiT/MAE), shape (grid * grid, dim)."""
    def emb_1d(pos, d):
        omega = 1.0 / 10000 ** (np.arange(d // 2) / (d // 2))
        out = pos[:, None] * omega[None]
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)
    ys, xs = np.meshgrid(np.arange(grid), np.arange(grid), indexing="ij")
    emb = np.concatenate([emb_1d(ys.reshape(-1), dim // 2), emb_1d(xs.reshape(-1), dim // 2)], axis=1)
    return mx.array(emb.astype(np.float32))


def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]


class Attention(nn.Module):
    def __init__(self, dim, n_head):
        super().__init__()
        self.n_head = n_head
        self.head_dim = dim // n_head
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def __call__(self, x, mask=None):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.head_dim).transpose(2, 0, 3, 1, 4)
        y = mx.fast.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], scale=self.head_dim ** -0.5, mask=mask)
        return self.proj(y.transpose(0, 2, 1, 3).reshape(B, T, C))


class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, 4 * dim)
        self.proj = nn.Linear(4 * dim, dim)

    def __call__(self, x):
        return self.proj(nn.gelu_approx(self.fc(x)))


class TextEncoder(nn.Module):
    """Direct learned embedding of token IDs, mean-pooled to one vector."""
    def __init__(self, config):
        super().__init__()
        dim = config.n_embd
        self.wte = nn.Embedding(config.vocab_size, dim)

    def __call__(self, tokens):
        keep = tokens != PAD  # BOS is never PAD, so every prompt keeps at least one token
        x = self.wte(tokens)
        w = keep[..., None].astype(x.dtype)
        return (x * w).sum(axis=1) / w.sum(axis=1)


class Block(nn.Module):
    """DiT block with adaLN-Zero conditioning on (timestep + pooled text)."""
    def __init__(self, config):
        super().__init__()
        dim = config.n_embd
        self.norm1 = nn.LayerNorm(dim, eps=1e-6, affine=False)
        self.attn = Attention(dim, config.n_head)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6, affine=False)
        self.mlp = MLP(dim)
        self.ada = nn.Linear(dim, 6 * dim)

    def __call__(self, x, c):
        shift1, scale1, gate1, shift2, scale2, gate2 = mx.split(self.ada(nn.silu(c)), 6, axis=-1)
        x = x + gate1[:, None] * self.attn(modulate(self.norm1(x), shift1, scale1))
        x = x + gate2[:, None] * self.mlp(modulate(self.norm2(x), shift2, scale2))
        return x


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        p, dim = config.patch_size, config.n_embd
        patch_dim = p * p * config.channels
        self.patch_embed = nn.Linear(patch_dim, dim)
        self._pos_emb = sincos_pos_embed(config.image_size // p, dim)  # underscore: fixed, not a parameter
        self.t_embed = nn.Sequential(nn.Linear(256, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.text = TextEncoder(config)
        self.text_proj = nn.Linear(dim, dim)
        self.blocks = [Block(config) for _ in range(config.n_layer)]
        self.final_norm = nn.LayerNorm(dim, eps=1e-6, affine=False)
        self.final_ada = nn.Linear(dim, 2 * dim)
        self.final = nn.Linear(dim, patch_dim)
        # adaLN-Zero: every block starts as the identity and the output starts at zero
        for layer in [b.ada for b in self.blocks] + [self.final_ada, self.final]:
            layer.weight = mx.zeros_like(layer.weight)
            layer.bias = mx.zeros_like(layer.bias)

    def __call__(self, x, t, tokens):
        """x: (B, H, W, C) noisy images, t: (B,) in [0, 1], tokens: (B, TEXT_LEN). Returns velocity, same shape as x."""
        B, H, W, C = x.shape
        p = self.config.patch_size
        g = H // p
        x = x.reshape(B, g, p, g, p, C).transpose(0, 1, 3, 2, 4, 5).reshape(B, g * g, p * p * C)
        x = self.patch_embed(x) + self._pos_emb
        c = self.t_embed(timestep_embedding(t, 256)) + self.text_proj(self.text(tokens))
        for block in self.blocks:
            x = block(x, c)
        shift, scale = mx.split(self.final_ada(nn.silu(c)), 2, axis=-1)
        x = self.final(modulate(self.final_norm(x), shift, scale))
        return x.reshape(B, g, g, p, p, C).transpose(0, 1, 3, 2, 4, 5).reshape(B, H, W, C)


NULL_TOKENS = tokenize([""])  # the unconditional prompt, for classifier-free guidance


def loss_fn(model, x, tokens):
    B = x.shape[0]
    drop = mx.random.uniform(shape=(B,)) < CAPTION_DROPOUT
    tokens = mx.where(drop[:, None], NULL_TOKENS, tokens)
    t = mx.random.uniform(shape=(B,))
    noise = mx.random.normal(x.shape)
    tt = t[:, None, None, None]
    xt = (1 - tt) * x + tt * noise
    v = model(xt, t, tokens)
    return mx.mean((v - (noise - x)) ** 2)


def sample(model, tokens, noise):
    """Euler ODE sampler from t=1 (noise) to t=0 with classifier-free guidance. Returns every state; the last is the image."""
    B = noise.shape[0]
    tokens = mx.concatenate([tokens, mx.broadcast_to(NULL_TOKENS, tokens.shape)])
    x = noise
    xs = [x]
    for i in range(SAMPLE_STEPS):
        t, t_next = 1 - i / SAMPLE_STEPS, 1 - (i + 1) / SAMPLE_STEPS
        v_cond, v_uncond = mx.split(model(mx.concatenate([x, x]), mx.full((2 * B,), t), tokens), 2)
        v = v_uncond + CFG_SCALE * (v_cond - v_uncond)
        x = x + (t_next - t) * v
        mx.eval(x)
        xs.append(x)
    return xs

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture
DEPTH = 8               # number of DiT blocks
N_EMBD = 256            # model width
N_HEAD = 4              # attention heads
PATCH_SIZE = 4          # 32x32 image -> 8x8 = 64 tokens
TEXT_LAYERS = 2         # text encoder depth

# Optimization
BATCH_SIZE = 128        # images per optimizer step
LEARNING_RATE = 1e-3    # AdamW peak learning rate
WEIGHT_DECAY = 0.01     # AdamW weight decay
ADAM_BETAS = (0.9, 0.95) # Adam beta1, beta2
WARMUP_RATIO = 0.05     # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.3    # fraction of time budget for LR warmdown
FINAL_LR_FRAC = 0.0     # final LR as fraction of peak
CAPTION_DROPOUT = 0.1   # drop the prompt this often, to train the unconditional model for CFG

# Sampling (used by inference.py on the benchmark prompt)
SAMPLE_STEPS = 32       # Euler steps from noise to image
CFG_SCALE = 2.0         # classifier-free guidance scale

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC


def main():
    t_start = time.time()
    mx.random.seed(42)
    mx.reset_peak_memory()

    config = ModelConfig(
        image_size=IMAGE_SIZE, patch_size=PATCH_SIZE, n_layer=DEPTH, n_head=N_HEAD, n_embd=N_EMBD,
        vocab_size=VOCAB_SIZE, text_len=TEXT_LEN, text_layers=TEXT_LAYERS,
    )
    print(f"Model config: {asdict(config)}")
    model = Model(config)
    mx.eval(model.parameters())
    num_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"Parameters: {num_params / 1e6:.2f}M")
    print(f"Time budget: {TIME_BUDGET}s")

    optimizer = optim.AdamW(learning_rate=LEARNING_RATE, betas=list(ADAM_BETAS), weight_decay=WEIGHT_DECAY)
    optimizer.init(model.trainable_parameters())
    loss_and_grad = nn.value_and_grad(model, loss_fn)
    state = [model.state, optimizer.state, mx.random.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def train_step(x, tokens):
        loss, grads = loss_and_grad(model, x, tokens)
        optimizer.update(model, grads)
        return loss

    train_loader = make_dataloader(BATCH_SIZE, "train")
    x, tokens, epoch = next(train_loader)  # prefetch first batch

    step = 0
    total_training_time = 0.0
    smooth_loss = 0.0
    while True:
        t0 = time.time()
        progress = min(total_training_time / TIME_BUDGET, 1.0)
        lrm = get_lr_multiplier(progress)
        optimizer.learning_rate = LEARNING_RATE * lrm
        loss = train_step(x, tokens)
        mx.async_eval(loss, state)
        x, tokens, epoch = next(train_loader)  # load the next batch while the GPU works
        train_loss = loss.item()

        # Fast fail: abort if loss is exploding or NaN
        if math.isnan(train_loss) or train_loss > 100:
            print("\nFAIL")
            exit(1)

        dt = time.time() - t0
        if step > 10:
            total_training_time += dt

        # Logging
        ema_beta = 0.9
        smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * train_loss
        debiased = smooth_loss / (1 - ema_beta ** (step + 1))
        pct = 100 * progress
        remaining = max(0, TIME_BUDGET - total_training_time)
        print(f"\rstep {step:05d} ({pct:.1f}%) | loss: {debiased:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.0f}ms | img/sec: {BATCH_SIZE / dt:,.0f} | epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)

        step += 1
        # Time's up - but only stop after warmup steps so we don't count compilation
        if step > 10 and total_training_time >= TIME_BUDGET:
            break

    print()  # newline after \r training log

    # Final eval
    val_loss = evaluate_loss(model)
    peak_memory_gb = mx.get_peak_memory() / 1024 ** 3
    total_seconds = time.time() - t_start
    run_dir = save_run(model, asdict(config), {
        "val_loss": round(val_loss, 6), "training_seconds": round(total_training_time, 1),
        "total_seconds": round(total_seconds, 1), "peak_memory_gb": round(peak_memory_gb, 1),
        "num_steps": step, "num_params_M": round(num_params / 1e6, 2),
    })

    # Benchmark images for the fixed prompt, saved next to the checkpoint
    from inference import generate
    sample_seconds = generate(run_dir, model, sample)

    # Final summary
    print("---")
    print(f"val_loss:         {val_loss:.6f}")
    print(f"training_seconds: {total_training_time:.1f}")
    print(f"total_seconds:    {total_seconds:.1f}")
    print(f"peak_memory_gb:   {peak_memory_gb:.1f}")
    print(f"img_per_sec:      {step * BATCH_SIZE / total_training_time:,.0f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {num_params / 1e6:.2f}")
    print(f"depth:            {DEPTH}")
    print(f"sample_seconds:   {sample_seconds:.1f}")
    print(f"run_dir:          {run_dir}")


if __name__ == "__main__":
    main()
