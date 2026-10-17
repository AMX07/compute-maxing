You fix a crashed ML experiment. The experiment edited `train.py` (a nanochat-style GPT in PyTorch with FlashAttention-3 and Muon, pretrained for 5 minutes on one NVIDIA H100) and the run crashed. Fix the bug so the experiment's idea runs as intended. Do not abandon or change the idea, and do not touch anything else.

The same hard rules apply: only `train.py` changes, nothing imported from `prepare` is redefined, the final `evaluate_bpb(model, ...)` call stays, `TOTAL_BATCH_SIZE` stays a multiple of `DEVICE_BATCH_SIZE × MAX_SEQ_LEN`, and the time accounting stays honest.

If the idea itself can't work (for example, it runs out of memory on an 80 GB H100 at any reasonable size, or FlashAttention-3 doesn't support what it needs), reply with `{"give_up": true, "reason": "..."}` and no edit blocks.

Otherwise reply with:

```json
{"give_up": false, "fix": "one sentence: what was wrong"}
```

followed by edit blocks against the CURRENT train.py:

<<<<<<< SEARCH
exact lines from train.py
=======
replacement lines
>>>>>>> REPLACE
