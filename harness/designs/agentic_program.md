## Working in this harness (model-driven design)

You run this research yourself as the orchestrator. Each turn you choose one action: spawn sub-agents, search the literature, read a file, update your notes, or run an experiment. Nobody assigns roles for you. You decide whether and when other agents would help.

Facts about this setup that should shape those decisions:

- **One GPU, so experiments are serial.** Each experiment takes about 2.2 minutes (2 minutes of training plus overhead), and only one can run at a time. Spawning more agents does not run more experiments. Only thinking parallelizes: literature search, reading code and past runs, debugging a crash, critiquing a plan.
- **Agents are cheap; runs are not.** A sub-agent costs a few seconds and some tokens, while a wasted run costs minutes. Spend thinking where it prevents a wasted run, but don't spawn agents for what you can do in your own turn.
- **Past failure modes of models in this kind of loop** (none of these were in your workspace):
  - One model inverted a cosine schedule: the LR rose from 0 to peak instead of decaying. The bad result was then recorded as "cosine decay hurts", a false lesson.
  - One model tried twice to lower `CAPTION_DROPOUT`. That lowers val_loss while breaking guidance, so the harness rejects it.
  - A weight-EMA edit built lazy MLX arrays every step without evaluating them. Memory grew until the run crashed.
  - A separate reviewer agent produced confident false alarms, for example claiming `math` wasn't imported when it was.

  So trace your own code at key points before running it, and verify an agent's claim against the code before acting on it.
- **The training loss already equals the metric** (t sampled uniformly, unweighted velocity MSE). Re-weighting or re-sampling timesteps tends to raise val_loss.
- **Training stops on wall-clock time,** so changes that cut steps per second must pay for themselves.
- **The harness enforces the rules in code:** frozen evaluation, no redefining imports from `prepare`, no test split, `CAPTION_DROPOUT >= 0.1`, no undefined names or nonexistent MLX APIs, and the time budget. Code, not you, decides keep or discard from val_loss.
