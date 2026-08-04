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

## Not completed: HF Hub push
**Not run.** Pushing requires an authenticated Hugging Face account
(`huggingface-cli login`) — free, but needs the user to complete an
interactive browser-based login themselves; this session has no credential
input path for it (per this project's security rules, API tokens are never
entered on the user's behalf, even if supplied). Once the user runs
`huggingface-cli login` in their own terminal, `python pipeline/push_to_hf.py
--repo-id <username>/catla-bn-en-tweets` will complete Phase 5's last step
using their now-authenticated CLI session.

## Next
Phase 6 — specialist ensemble LoRA fine-tuning: IndicTrans2, NLLB-200-
distilled-600M, and BanglaT5, both directions, on `data/processed/
train.jsonl`. This is the first phase that genuinely needs a GPU — no GPU
has been confirmed on this machine (flagged since `phase_0_status.md`), so
this phase is expected to run via the Colab/Kaggle notebooks the PRD calls
for, not directly in this session.
