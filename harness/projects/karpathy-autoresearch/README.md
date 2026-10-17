# karpathy/autoresearch

The harness profile and prompts for running this repo's agent loop on a clone of [karpathy/autoresearch](https://github.com/karpathy/autoresearch): a single-GPU nanochat GPT, 5-minute runs, metric `val_bpb`. Upstream needs one NVIDIA GPU (CUDA, FlashAttention-3), so on a Mac the training runs on a remote CUDA box over ssh.

Set up a clone (a separate directory, on a branch of your own, never pushed upstream):

```bash
git clone https://github.com/karpathy/autoresearch ~/Projects/autoresearch-harness
```

```bash
git -C ~/Projects/autoresearch-harness checkout -b harness/run1
```

```bash
cp -R harness/projects/karpathy-autoresearch/project.json harness/projects/karpathy-autoresearch/prompts ~/Projects/autoresearch-harness/
```

Then set `remote.host` in its `project.json` to an ssh host with an NVIDIA GPU, where `remote.dir` is a clone of the same repo that has already run `uv sync` and `uv run prepare.py`. The harness copies `train.py` there before each run and streams the output back into `run.log`. With `"host": null` it trains locally (`uv run train.py`), which needs CUDA.

Run the loop:

```bash
python3 harness/run.py --design fixed --workspace ~/Projects/autoresearch-harness --no-researcher
```

The first run adopts the folder: it adds `workspace.json`, `memory/` and `traces/` (run output is kept out of git via `.git/info/exclude`) and never overwrites the project's files. Commits go to the clone's checked-out branch; `"push": null` means nothing is ever pushed. Test without a GPU by adding `HARNESS_FAKE_TRAIN=1`.

- `project.json`: the metric and how to read it from the summary `train.py` prints, memory in MB (`peak_vram_mb`), the frozen `prepare.py`, the required `evaluate_bpb(model` call, the train command, timeouts, and the remote host.
- `prompts/`: nanochat versions of every role's prompt, with the same reply formats as `harness/prompts/`. `agentic_program.md` is appended to `program.md` when the folder is adopted by the model-driven design.
- Karpathy's own 125-experiment run is on the upstream branch `exp/H100/mar8`; `webui/tools/karpathy_map.py` draws its architecture map for the web UI.
