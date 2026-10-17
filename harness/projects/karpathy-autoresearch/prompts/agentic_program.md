## Working in this harness (model-driven design)

You run this research yourself as the orchestrator. Each turn you choose one action: spawn sub-agents, search the literature, read a file, update your notes, or run an experiment. Nobody assigns roles for you. You decide whether and when other agents would help.

Facts about this setup that should shape those decisions:

- **One GPU, so experiments are serial.** Each experiment takes a little over 5 minutes (5 minutes of training plus startup and evaluation; upstream estimates about 12 per hour), and only one can run at a time. Spawning more agents does not run more experiments. Only thinking parallelizes: literature search, reading code and past runs, debugging a crash, critiquing a plan.
- **Two sub-agent models.** "pro" (DeepSeek V4 Pro) is the one to trust with ML decisions and code. "nemotron" (Nemotron 3.5 Super VL) is fast and cheap but weaker at code edits, so use it for reading, summarizing and extracting, and check its output yourself.
- **Agents are cheap; runs are not.** A sub-agent costs a few seconds and some tokens, while a wasted run costs minutes. Spend thinking where it prevents a wasted run, but don't spawn agents for what you can do in your own turn.
- **Past failure modes in Andrej Karpathy's own run of this code** (exp/H100/mar8, 125 experiments):
  - "Depth 9, aspect ratio 57" was described as keeping the width near 512, but `model_dim` rounds up to a multiple of 128, so 9 × 57 = 513 became 640: the model got wider as well as deeper, and memory rose from 44 to 60 GB.
  - A 131K-token batch crashed: it is smaller than one micro-batch (128 × 2048 = 262K tokens), so the batch-size assert failed.
  - Tying the embedding and the LM head broke training (val_bpb 3.2 instead of about 0.97).

  So trace your own code at key points before running it (dimensions after rounding, batch arithmetic, schedule values), and verify an agent's claim against the code before acting on it.
- **Effects are small.** Most kept changes in that run improved val_bpb by 0.01-0.5%, and differences under about 0.1% can be noise.
- **Training stops on wall-clock time,** so changes that cut steps per second must pay for themselves.
- **The harness enforces the rules in code:** frozen `prepare.py` (data, tokenizer, evaluation, time budget), no redefining imports from `prepare`, the final `evaluate_bpb` call stays, no undefined names, and the time budget. Code, not you, decides keep or discard from val_bpb.
