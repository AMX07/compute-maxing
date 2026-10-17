# Playbook

Lessons the loop has learned, maintained by the reflector.

- [L1] Baseline: val_loss 0.1838 in 2719 steps at 11.48M params (exp 0)
- [L2] Exp 1's cosine schedule was implemented inverted (LR rose from 0 to peak), so its +7% says nothing about cosine decay itself; a correct cosine/linear decay is still untested (exp 1, human review)
- [L3] Widening DiT from 256 to 384 hidden dim regresses val_loss by ~1.5% in 2-min runs; fewer steps (1525 vs 2719) outweigh capacity (exp 2)
