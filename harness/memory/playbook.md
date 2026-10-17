# Playbook

Lessons the loop has learned, maintained by the reflector.

- [L1] Baseline: val_loss 0.1838 in 2719 steps at 11.48M params (exp 0)
- [L2] Exp 1's cosine was inverted and exp 3's cosine was a no-op (missing math import); correct cosine decay is still untested (exp 1, 3, human review)
- [L3] Widening DiT from 256 to 384 hidden dim regresses val_loss by ~1.5% in 2-min runs; fewer steps (1525 vs 2719) outweigh capacity (exp 2)
- [L4] A correct cosine warmdown over the last 30% matched the linear warmdown (+0.09%, within noise): warmdown shape is not a lever here (exp 3, human review)
- [L5] EMA implementation crashed: lazy arrays without mx.eval per step caused memory growth, and model.update(ema_state) failed due to parameter name mismatch (exp 4)
