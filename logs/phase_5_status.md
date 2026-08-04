# Phase 5 — Dataset Splitting & Augmentation — Status

**Date:** 2026-08-04

## Done
- Built `pipeline/split_and_augment.py`, producing `data/processed/{train,
  val,test}.jsonl` from the 811,204-row `unified.jsonl` (Phase 3).
- **Stratified 85/7.5/7.5 split by `source_dataset`** (so every split has a
  representative mix of registers, not just an overall random cut), with
  one deliberate exception: the Phase 2 proxy organic test set
  (`is_proxy_test_set: true`, 2,996 rows) was excluded from the stratified
  split entirely and appended wholesale to TEST, per the PRD's requirement
  to keep the organic held-out set fully in TEST.
- **Synthetic code-mixing augmentation, TRAIN only**: for 8% of eligible
  `bn_en_translation` TRAIN rows, ~30% of the words in the Bengali target
  text were romanized via `transliterator.bn_to_latin()`, producing a
  Banglish-style code-mixed variant tagged `is_synthetic: true` and
  `source_dataset` suffixed `+synthetic_codemix`. Verified **zero** synthetic
  rows leaked into val/test (checked directly). Method is documented as a
  practical heuristic, not linguistically-validated code-switching — it
  reuses Phase 4's rule-based transliterator, so it inherits that module's
  documented accuracy limitations.
- Final counts:

  | split | rows | notes |
  |---|---|---|
  | train | 717,929 | 686,973 original + 30,956 synthetic-code-mixed |
  | val | 60,613 | — |
  | test | 63,618 | 60,622 stratified + 2,996 organic-proxy |
  | **total** | **842,160** | |

- Wrote `pipeline/push_to_hf.py` for Phase 5's remaining requirement
  (pushing the final splits to a public Hugging Face Hub dataset repo).

## HF Hub push — completed 2026-08-04 (later same day, after Phase 6)
Initially deferred (blocked on `huggingface-cli login`; see below), then
completed once the login blocker was resolved. Two real issues surfaced and
were fixed along the way:

1. **`huggingface-cli`/`hf` console-script `.exe` wrappers are blocked by a
   Device Guard / Application Control policy on the user's machine** (same
   family of restriction as the torch DLL block noted in
   `phase_6_status.md`). Worked around by calling `huggingface_hub`'s
   Python functions directly via the venv's `python.exe -c "..."` (which
   Device Guard allows) instead of the blocked `.exe` entry points — the
   user ran `login(token=...)` and `whoami()` this way, confirmed
   authenticated as `shanjivkr`.
2. **`pipeline/push_to_hf.py` bug**: `Dataset.from_json()` infers its Arrow
   schema per streaming-read chunk; because `task_labels` has a wildly
   different shape per source (9-key hate-speech dict for BanTH, a single
   `{"label": ...}` for the sentiment sets, `None` for Samanantar/OPUS/
   BanglaTLit/HashSet), an early chunk that happened to be all-`None`
   inferred a `null` dtype for that column, then crashed
   (`TypeError: Couldn't cast array of type string to null`) the moment a
   later chunk had a real value. Fixed by reading the JSONL manually,
   serializing `task_labels` to a JSON string (or `None`) up front, and
   building the `Dataset` from a fully-materialized list (`Dataset.from_list`)
   so Arrow sees every row before committing to a schema.

**Pushed successfully to
[huggingface.co/datasets/shanjivkr/catla-bn-en-tweets](https://huggingface.co/datasets/shanjivkr/catla-bn-en-tweets)**
(public), all three splits (train/val/test).

## Next
Phase 6 — specialist ensemble LoRA fine-tuning: IndicTrans2, NLLB-200-
distilled-600M, and BanglaT5, both directions, on `data/processed/
train.jsonl`. This is the first phase that genuinely needs a GPU — no GPU
has been confirmed on this machine (flagged since `phase_0_status.md`), so
this phase is expected to run via the Colab/Kaggle notebooks the PRD calls
for, not directly in this session.
