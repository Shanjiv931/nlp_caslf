# Phase 8 — Evaluation (BLEU/chrF++/TER/COMET/PFS) — Status

**Date:** 2026-08-16

## Context
Followed the PRD's own phase ordering (Evaluation before Reassembly
integration/Phase 9) rather than my own initial instinct to build Phase 9
first — on reflection the PRD's ordering is sound: BLEU/chrF++/TER/COMET
are computed on the core translation text itself (most of the held-out
test set is formal/informal-register sentence pairs, not hashtag-dense
tweets specifically), so evaluating the core `quality_pipeline.py` doesn't
actually need Phase 9's reassembly wired in first. Phase 9 matters more
for the eventual demo's user-facing output quality than for measuring
translation accuracy itself.

## Built

### `eval/platform_fidelity.py`
The PRD's own custom Platform-Fidelity Score, computed from 3
sub-dimensions that are genuinely just string comparisons: emoji
preservation (set-based), verbatim mention/URL preservation (exact-match —
these must never be translated), hashtag *count* preservation (not
translation *correctness*, disclosed as a real limitation — no automatic
ground truth exists for hashtag-translation naturalness). Each example's
score only averages over sub-dimensions actually present in that example
(an example with no emoji doesn't get penalized or artificially boosted on
that dimension). Explicitly does NOT score the PRD's fourth stated
dimension (slang naturalness) — no honest automatic way to measure that
without a reference or human judgment, and a fake proxy score would
misrepresent what's being measured. **This module runs for real on this
machine** (pure text logic, no models) — verified with 11 passing unit
tests covering full/partial/zero preservation, count-capping, and
dataset-level aggregation.

### `eval/evaluate.py`
Three-way comparison: `pretrained_baseline` (base model, no LoRA — isolates
how much fine-tuning itself contributed) vs. `single_model_baseline` (one
fine-tuned engine, beam search only) vs. `full_pipeline`
(`quality_pipeline.produce_translation()`, the complete Phase 7 chain).
Computes BLEU/chrF++/TER (`sacrebleu`) and reference-based COMET
(`Unbabel/wmt22-comet-da` — distinct from `qe_rerank.py`'s reference-free
CometKiwi used at real inference time) plus Platform-Fidelity Score for
each mode, and explicitly prints the `full_pipeline - single_model_baseline`
delta per metric — proving that delta is the entire point, per the PRD's
own framing (a pipeline that doesn't measurably beat a single fine-tuned
model isn't worth its added complexity).

**Genuinely verified, not just structurally**: `compute_metrics()` actually
runs — `sacrebleu` is pure Python, no torch needed. Installed it locally
and tested for real: perfect match scores ~100 BLEU/0 TER (within float
tolerance — the test initially asserted exact equality and caught a real
`100.00000000000004` floating-point artifact, fixed with `pytest.approx`),
completely wrong text scores low, and partial matches land between the
two. `load_test_pairs()` was also validated against the *actual*
`data/processed/test.jsonl` from Phase 5 (still present locally): row
counts, bn2en vs en2bn correctly pulling different-language source text
(checked via Bengali-Unicode-block ratio), and `sanitize_text()`
correctly stripping literal `<unk>` from real rows. 7/7 tests passing.

**Not verified**: `compute_comet()` and all three `translate_*` functions
need real models — correct-by-construction against the real
transformers/peft/comet APIs, consistent with every other model-calling
function built in this project on this machine, but unrun here.

### `eval/RESULTS.md`
Methodology document, not fabricated numbers — states plainly that actual
BLEU/chrF++/TER/COMET/PFS results require running `evaluate.py` on
Kaggle/Colab, with the exact commands to do so, and reserves the sections
(delta table, 15-20 worked examples per the PRD's FR10) for real numbers
once they exist.

## Test count
**89/89 passing project-wide** (71 from Phases 4/6/7 + 18 new: 11 for
`platform_fidelity.py`, 7 for `evaluate.py`).

## Next
Two independent paths, neither blocking the other: Phase 9 (wire
`quality_pipeline.py` into a full `translate_tweet()` callable via
`catla.py`, using Phase 4's tokenizer/slang_normalizer/transliterator/
reassembler) or actually running `eval/evaluate.py` for real on Kaggle to
get the first genuine accuracy numbers for this project. Both still
require the user's Kaggle/Colab session, same as Phases 6-7.
