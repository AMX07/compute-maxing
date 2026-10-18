# Import modded-nanogpt speedrun history

The adapter imports the main track's published record table from [KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) into Turing. It reads a pinned Git revision, does not execute training code or call an LLM, and writes a separate generated project.

```sh
git clone https://github.com/KellerJordan/modded-nanogpt ../modded-nanogpt
uv run --no-project python webui/tools/modded_nanogpt.py ../modded-nanogpt --register
uv run --no-project python webui/server.py
```

Skip cloning if the repository already exists. The default output is `../modded-nanogpt/.turing/`; `--output /path/to/history` selects another location. `--register` adds that output to Turing's recent projects. Re-running the importer refreshes its generated files; it refuses to overwrite a non-import project. Generated files should not be edited by hand. To update from upstream, update the clone explicitly, then rerun the importer.

## What the view means

- The metric is **training seconds**, lower is better, at a published FineWeb validation-loss target of **≤3.28 on 8 H100s**. The chart uses a logarithmic scale so recent gains remain visible next to the 45-minute baseline.
- Each row is a published leaderboard milestone, not a local experiment or an independently reproduced result. There are no fabricated failures, predictions, measured losses, or validation claims. Dates, contributors, source links, the upstream commit, and a README hash are retained.
- Repeated record numbers are re-timings, not new architecture changes. They reset the best-time comparison; the chart breaks the best-time line at these boundaries. Comparisons across different timing rules or software versions are not controlled experiments.
- Categories come from keyword matches on descriptions. Bundled changes appear under **Combined changes**. **Latest** means the most recent published change in each category, not a reconstructed architecture or proof of causality.
- Where the exact linked log embeds a syntactically valid Python source block, the importer saves it as `runs/<id>/train.py`. Diffs compare adjacent available snapshots within a timing series. Missing or multi-file source stays unavailable rather than being replaced with the current trainer. Links remain available for logs that are directories, unavailable locally, or lack an extractable source block.
- A sparse clone can be used; import works offline with the table and whatever linked logs have been checked out. More locally available logs can yield more source snapshots on a subsequent import.

The output contains `results.tsv`, `architecture.json`, `import.json`, run metadata, extracted source, available diffs, and the upstream license. Upstream trainer files remain unchanged.

## Training support

The imported project displays published history. Its Agents view also offers a standalone training launcher for a prepared local or SSH machine; see [Start training](../README.md#start-training). The current upstream trainer uses `train_gpt.py` plus `track_1_short/`, eight NVIDIA H100 GPUs, and CUDA-specific kernels. Turing's existing single-file `train.py` editing loop is not an adapter for that trainer.

A live research adapter would need multi-file edits, frozen data/evaluation rules, remote distributed execution, and a time objective gated on reaching the loss threshold. A faster run that misses the threshold must not be accepted as an improvement. No GPU jobs or dependency installs are triggered by this import.

## Checks

```sh
uv run --no-project python -m unittest discover -s tests -v
```
