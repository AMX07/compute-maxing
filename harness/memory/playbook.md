# Playbook

Lessons the loop has learned, maintained by the reflector.

- [L1] Baseline: val_loss 0.1838 in 2719 steps at 11.48M params (exp 0)
- [L2] Exp 1's cosine was inverted and exp 3's cosine was a no-op (missing math import); correct cosine decay is still untested (exp 1, 3, human review)
- [L3] Widening DiT from 256 to 384 hidden dim regresses val_loss ~1.5%; fewer steps (1525 vs 2719) outweigh capacity (exp 2, 5)
- [L4] Warmdown shape is not a lever (cosine vs linear within noise), but warmup removal hurts ~0.7%; keep warmup (exp 3, 6)
- [L5] EMA implementation crashed: lazy arrays without mx.eval per step caused memory growth, and model.update(ema_state) failed due to parameter name mismatch (exp 4)
- [L6] Removing text encoder for learned embedding improves val_loss ~0.5% via +12% steps; conditioning quality loss is minor on CIFAR-10 (exp 5)
- [L7] Removing warmup (constant LR) regresses val_loss ~0.7% vs baseline; warmup is protective in short runs (exp 6)
