"""Write architecture.json for karpathy/autoresearch's exp/H100/mar8 run, from its results.tsv and baseline train.py.

    git clone https://github.com/karpathy/autoresearch ~/Projects/autoresearch
    git -C ~/Projects/autoresearch checkout -b exp/H100/mar8 origin/exp/H100/mar8
    python3 webui/tools/karpathy_map.py ~/Projects/autoresearch

Components and baseline facts were read from train.py at the run's base commit (500114a). Each experiment's component
comes from keyword rules on its description, with the rows the rules get wrong pinned by hand. Labels are the
descriptions; "after", notes and costs for kept changes were checked against the commits and results.tsv.
"""
import csv
import io
import json
import re
import sys

proj = sys.argv[1]
rows = list(csv.reader(io.StringIO(open(f"{proj}/results.tsv").read()), delimiter="\t"))[1:]

RULES = [
    (r"random seed|gradient clipping", "seed"),
    (r"\bbatch\b", "batch"),
    (r"MLP hidden", "mlp"),
    (r"parallel attn|depth \d|aspect|dim 640", "shape"),
    (r"momentum warmup|muon|Muon|matrix LR|ns_steps|NorMuon", "muon"),
    (r"warmup|warmdown|FINAL_LR_FRAC", "schedule"),
    (r"window pattern|SSSL|SSSSL|SSSSSL|short window", "window"),
    (r"RoPE|rotary|multi-query|n_kv_head", "attn"),
    (r"softcap|weight tying", "head"),
    (r"init|c_proj", "init"),
    (r"WD|weight decay|cautious", "wd"),
    (r"LR|beta|scale reference", "adamw"),
]
# Checked by hand against the descriptions: rows the keyword rules get wrong.
PINNED = {33: "residual", 50: "residual", 76: "residual", 80: "residual", 10: "residual", 121: "residual",
          22: "ve", 32: "ve", 35: "ve", 82: "ve", 89: "ve", 123: "ve",
          111: "adamw", 124: "adamw", 117: "wd", 90: "wd", 91: "init", 92: "init"}

# The component's state after each kept change (cumulative), and notes checked against the code and results.
AFTER = {
    1: "262K tokens / step",
    2: "depth 9 · AR 57 → dim 640 (5 heads)",
    4: "no warmup · 70% warmdown · final 0",
    5: "SSSSL · short = ½ context (1024)",
    6: "SSSSL · short = ⅛ context (256)",
    7: "RoPE base 200K · QK-norm · FA3 · full MHA",
    8: "β (0.8, 0.95) · emb 0.8 · unemb 0.004 · scalars 0.5",
    10: "resid λ 1.0 · x0 λ 0.05",
    11: "no warmup · 70% warmdown · final 0.05",
    13: "β (0.8, 0.95) · emb 0.8 · unemb 0.006 · scalars 0.5",
    21: "LR 0.04 · momentum 0.85→0.95 over 200 steps · ns 5 · β2 0.95",
    37: "β (0.8, 0.95) · emb 0.8 · unemb 0.005 · scalars 0.5",
    40: "cautious 0.2 on matrices (→0) · lm_head 0.01",
    66: "uniform ±0.8·√3/√d · c_proj = 0",
    68: "uniform ±0.7·√3/√d · c_proj = 0",
    94: "uniform ±0.68·√3/√d · c_proj = 0",
    102: "cautious 0.2 on matrices (→0) · lm_head 0.01 · emb 0.001",
    104: "… · lm_head 0.01 · emb 0.001 · VE 0.001",
    105: "… · lm_head 0.01 · emb 0.001 · VE 0.002",
    106: "… · lm_head 0.01 · emb 0.001 · VE 0.003",
    111: "β (0.8, 0.95) · emb 0.9 · unemb 0.005 · scalars 0.5",
    118: "no warmup · 75% warmdown · final 0.05",
}
NOTES = {
    1: "Half the tokens per step, so about twice the optimizer steps in 5 minutes. Memory 44.0 → 43.9 GB.",
    2: "9 × 57 = 513 rounds up to 640 (HEAD_DIM 128), so the model got wider as well as deeper. The description's '~512' is off; memory rose 43.9 → 60.2 GB.",
    5: "With 9 layers, long-window layers drop from 3 to 2 (the last layer is always long).",
    6: "Short-window layers attend over 256 tokens instead of 1024.",
    15: "Crashed: 131K tokens is less than one micro-batch (128 × 2048 = 262K), so the batch-size assert failed.",
    88: "Broken run: val_bpb 3.22, far above the plot.",
}
COST = {
    1: "2× optimizer steps per 5 min",
    2: "≈1.8× matrix FLOPs per token (9·640² vs 8·512²)",
    5: "long-window layers 3 → 2 of 9",
    6: "short-layer attention O(T·w): w 1024 → 256",
    85: "KV heads 5 → 1",
    59: "attention and MLP read the same input",
}

changes = {}
for i, r in enumerate(rows):
    if i == 0:
        continue
    desc = r[4]
    comp = PINNED.get(i) or next((c for pat, c in RULES if re.search(pat, desc)), "other")
    label = re.sub(r"\s*\(.*?\)\s*$", "", desc).strip()
    label = label[0].upper() + label[1:]
    ch = {"component": comp, "label": label}
    if i in AFTER:
        ch["after"] = AFTER[i]
    if i in COST:
        ch["cost"] = COST[i]
    paren = re.search(r"\((.*?)\)\s*$", desc)
    note = NOTES.get(i) or (paren.group(1)[0].upper() + paren.group(1)[1:] + "." if paren else None)
    if note:
        ch["note"] = note
    changes[str(i)] = ch

arch = {
    "name": "karpathy/autoresearch · H100",
    "summary": "Karpathy's own autoresearch run (branch exp/H100/mar8): an agent edited train.py, a single-GPU nanochat GPT, "
               "for 125 experiments of 5 minutes each on one H100. Metric: val_bpb on a pinned validation shard, lower is "
               "better. Hypotheses below are synthesized from the experiment log.",
    "metric": {"name": "val_bpb", "goal": "min"},
    "groups": [
        {"id": "model", "name": "Model", "flow": True},
        {"id": "training", "name": "Training"},
    ],
    "components": [
        {"id": "embed", "group": "model", "name": "Token embedding",
         "baseline": {"label": "8192 vocab × 512 · bf16", "cost": "O(1) lookup", "detail":
                      "BPE vocabulary of 8192 (trained by prepare.py), embedding width = model dim. Init std 1.0."}},
        {"id": "ve", "group": "model", "name": "Value embeddings",
         "baseline": {"label": "alternate layers · gate 2σ(32 ch)", "cost": "O(1) lookup per VE layer", "detail":
                      "ResFormer-style value residual: a per-layer token embedding mixed into V with an input-dependent "
                      "gate per head, on every other layer (the last always included)."}},
        {"id": "attn", "group": "model", "name": "Attention",
         "baseline": {"label": "RoPE base 10K · QK-norm · FA3 · full MHA", "cost": "O(T·w·d) per layer", "detail":
                      "Causal self-attention with head dim 128, rotary embeddings (base 10,000), RMS-normed queries and "
                      "keys, FlashAttention-3, as many KV heads as query heads."}},
        {"id": "window", "group": "model", "name": "Window pattern",
         "baseline": {"label": "SSSL · short = ½ context (1024)", "cost": "short layers O(T·T/2)", "detail":
                      "Sliding-window attention per layer: S = half context (1024 tokens), L = full 2048. "
                      "The last layer is always L."}},
        {"id": "mlp", "group": "model", "name": "MLP",
         "baseline": {"label": "4× · ReLU²", "cost": "O(8·d²) per token", "detail":
                      "Two linear layers (d → 4d → d) with squared ReLU, no biases."}},
        {"id": "residual", "group": "model", "name": "Residual scalars",
         "baseline": {"label": "resid λ 1.0 · x0 λ 0.1", "cost": "2 scalars per layer", "detail":
                      "Learned per-layer scalars: one scales the residual stream, one mixes the initial embedding x0 "
                      "back in at every layer."}},
        {"id": "shape", "group": "model", "name": "Shape",
         "baseline": {"label": "depth 8 · AR 64 → dim 512 (4 heads)", "cost": "≈6·P FLOPs per token", "detail":
                      "model_dim = depth × ASPECT_RATIO rounded up to a multiple of HEAD_DIM (128). Pre-norm blocks "
                      "(RMSNorm), attention then MLP."}},
        {"id": "head", "group": "model", "name": "LM head",
         "baseline": {"label": "untied · logit softcap 15", "cost": "O(d·V) per token", "detail":
                      "Separate unembedding (init std 0.001); logits squashed with 15·tanh(x/15)."}},
        {"id": "init", "group": "model", "name": "Initialization",
         "baseline": {"label": "uniform ±√3/√d · c_proj = 0", "cost": "—", "detail":
                      "Attention and MLP input matrices uniform in ±√3·d^-½; output projections start at zero; "
                      "embedding std 1.0, lm_head std 0.001, value-embedding gates at zero (neutral)."}},
        {"id": "muon", "group": "training", "name": "Muon",
         "baseline": {"label": "LR 0.04 · momentum 0.85→0.95 over 300 steps · ns 5 · β2 0.95", "cost": "5 Newton–Schulz steps",
                      "detail": "Muon with Nesterov momentum and a NorMuon-style second moment for every transformer "
                                "matrix, grouped by shape."}},
        {"id": "adamw", "group": "training", "name": "AdamW groups",
         "baseline": {"label": "β (0.8, 0.95) · emb 0.6 · unemb 0.004 · scalars 0.5", "cost": "O(P) per step",
                      "detail": "AdamW for embeddings, value embeddings, lm_head and the residual scalars; LRs scaled "
                                "by (d/768)^-½."}},
        {"id": "wd", "group": "training", "name": "Weight decay",
         "baseline": {"label": "cautious 0.2 on matrices (→0) · none elsewhere", "cost": "—", "detail":
                      "Cautious weight decay (only where the update and the weight agree in sign) on Muon matrices, "
                      "decaying linearly to 0 over training. No decay on AdamW groups."}},
        {"id": "schedule", "group": "training", "name": "LR schedule",
         "baseline": {"label": "no warmup · 50% warmdown · final 0", "cost": "—", "detail":
                      "Time-based: flat LR, then a linear warmdown over the last half of the 5-minute budget to 0."}},
        {"id": "batch", "group": "training", "name": "Batch",
         "baseline": {"label": "524K tokens / step", "cost": "2 micro-batches of 128 × 2048", "detail":
                      "2^19 tokens per optimizer step, sequence length 2048, device batch 128."}},
        {"id": "seed", "group": "training", "name": "Seed & clipping",
         "baseline": {"label": "seed 42 · no grad clipping", "cost": "—", "detail": "Fixed seed; gradients are not clipped."}},
    ],
    "changes": changes,
    "hypotheses": [
        {"id": "H1", "status": "validated", "evidence": [1, 42, 47, 56, 79],
         "text": "In a fixed 5-minute budget, more optimizer steps beat more capacity per step."},
        {"id": "H2", "status": "validated", "evidence": [5, 6, 64, 99, 100],
         "text": "Cheaper local attention pays off: more, shorter sliding-window layers help, down to ⅛ context with 2 long layers."},
        {"id": "H3", "status": "validated", "evidence": [7, 38, 62, 122],
         "text": "A higher RoPE base helps at 2K context, with an optimum near 200K."},
        {"id": "H4", "status": "validated", "evidence": [66, 68, 94, 67, 95],
         "text": "Smaller init for the transformer matrices helps, best at about 0.68×."},
        {"id": "H5", "status": "validated", "evidence": [102, 104, 105, 106, 43, 107],
         "text": "A tiny weight decay on embeddings and value embeddings helps; 0.01 or more hurts."},
        {"id": "H6", "status": "validated", "evidence": [4, 118, 18, 119, 78],
         "text": "A longer linear warmdown helps, up to about 75% of the budget; cosine is worse."},
        {"id": "H7", "status": "validated", "evidence": [21, 12, 28, 30, 31, 96],
         "text": "Muon's defaults are near-optimal: only a shorter momentum warmup helped."},
        {"id": "H8", "status": "invalidated", "evidence": [3, 44],
         "text": "Some LR warmup helps."},
        {"id": "H9", "status": "invalidated", "evidence": [59, 85, 88, 35, 89],
         "text": "The architecture can be simplified for free: parallel blocks, multi-query attention, weight tying, fewer value embeddings."},
        {"id": "H10", "status": "invalidated", "evidence": [50, 33, 80],
         "text": "The x0 skip can be removed, and the residual scalars shrunk, without cost."},
    ],
}
json.dump(arch, open(f"{proj}/architecture.json", "w"), indent=2, ensure_ascii=False)
from collections import Counter
print(Counter(c["component"] for c in changes.values()))
print(sum(1 for r in rows[1:] if r[3] == "keep"), "kept;", len(rows) - 1, "experiments")
