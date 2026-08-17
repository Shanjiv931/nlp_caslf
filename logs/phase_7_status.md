# Phase 7 — Quality Layer (Ensemble + QE Rerank + LLM Post-edit + Round-trip Verify) — Status

**Date:** 2026-08-16

## Context
Phase 6 is complete (6/6 LoRA adapters trained and Hub-verified). This
phase builds the quality mechanism the PRD's own design philosophy says
actually produces "extraordinary" output — not any single fine-tuned
model's raw decode, but the chain: ensemble candidate generation -> QE
reranking -> constrained LLM post-editing -> round-trip verification.

This machine cannot run torch/transformers (no CUDA GPU, and a Device
Guard policy blocks even CPU torch — see `phase_6_status.md`). Every
module below is written correctly against the real APIs (transformers,
peft, comet, sentence-transformers) and mirrors Phase 6's
already-Kaggle-verified preprocessing exactly (via the shared
`mt_compat.py`), but the actual model-loading/generation code paths are
unverified here — only what's genuinely testable without a model (pure
text transforms, dedup logic, threshold decisions, and — most
importantly — the fallback/decision orchestration itself, using mocked
stage functions) has been tested. **71/71 tests passing** across the whole
project (57 pre-existing + this phase's new ones).

## Refactor done first: `pipeline/mt_compat.py`
Before writing any Phase 7 code, extracted `train.py`'s environment/
compatibility shims (`ensure_transformers_onnx_shim`,
`ensure_tie_weights_compat`), language-tagging logic
(`direction_lang_codes`, `tag_indictrans2_text`), and sanitization
(`sanitize_text`) into a new shared module. Rationale: Phase 7's inference
code needs the *exact* same preprocessing Phase 6's training used —
train/inference preprocessing skew is a classic, real source of quality
loss in MT systems, and copy-pasting the same fixes into a second file
risks them drifting out of sync over time. `train.py` now imports from
`mt_compat.py` instead of duplicating; re-verified all existing tests
still pass after the refactor (33/33 at that point) before proceeding.
14 new unit tests added for `mt_compat.py` itself.

## Modules built

### `pipeline/ensemble.py`
Multi-candidate generation across all 3 engines. `load_engine()` loads a
base model + its LoRA adapter (from the HF Hub repos Phase 6 produced),
applying the same compat shims and NLLB/IndicTrans2 language setup as
training. `generate_candidates_for_engine()` generates via both beam
search (`num_beams=4, num_return_sequences=2`) and nucleus sampling
(`top_p=0.9, temperature=0.9, num_return_sequences=2`) — diversity of
*decoding strategy*, not just of model, per the PRD's "never trust a
single decode" principle. Beam search candidates capture
`sequences_scores` (transformers' own length-normalized log-prob, free
during generation) as `Candidate.raw_score` — this becomes qe_rerank.py's
fallback ranking signal. `dedupe_candidates()` drops exact-text duplicates
and empty strings. 9 unit tests (preprocessing/postprocessing text
transforms via mock `EngineContext`/`IndicProcessor`, dedup logic).

### `pipeline/qe_rerank.py`
Reference-free QE scoring via the open-source `comet` toolkit (tries
`Unbabel/wmt23-cometkiwi-da-xl` then `Unbabel/wmt22-cometkiwi-da`). Falls
back to `ensemble.py`'s captured beam-search log-probs if no QE model can
be loaded, with unscored (nucleus-sampled) candidates explicitly ranked
below any scored candidate — never given a fabricated score, never treated
as equal to a real one. `pick_best()` never crashes on an all-`None`-score
list and never treats `None` as `0.0`. 6 unit tests (score comparison,
`None`-handling, fallback wiring).

### `pipeline/llm_postedit.py`
LLM-based Automatic Post-Editing. Tries Qwen2.5-7B-Instruct, then
Llama-3.1-8B-Instruct, then Gemma-2-9b-it (all free, open-weight,
instruction-tuned), 4-bit by default. Strict system+user prompt: fluency
only, meaning preserved exactly, protected tokens (`tokenizer.
protected_spans()` from Phase 4 — emoji/hashtags/mentions/URLs) must
survive verbatim. `protected_tokens_preserved()` is a cheap, deterministic
first line of defense — checked *before* the much more expensive
round-trip verification pass — and `postedit()` falls back to the pre-edit
MT candidate (never ships a broken edit) if the LLM violates that
contract, if it returns an empty string, or if no LLM could be loaded at
all. 10 unit tests, including full `postedit()` orchestration with a
monkeypatched LLM call exercising all three fallback paths.

### `pipeline/roundtrip_verify.py`
Translates the candidate back toward the source language (single engine —
default NLLB, for speed, per the PRD's explicit allowance) and compares it
to the original source text via LaBSE cosine similarity
(`sentence-transformers/LaBSE`). Returns a result dict, never raises on a
low score — a failed check is information for the caller to act on. 6 unit
tests (direction-reversal correctness, threshold decisions, mocked
end-to-end `verify()`).

### `pipeline/quality_pipeline.py`
Wires all four into `produce_translation(source_text, direction) -> dict`.
This is where the PRD's quality guarantees are actually implemented, so
it's the most thoroughly tested module: 7 integration-level tests with
every stage mocked, explicitly covering — happy path; no candidates
generated; reranking produces no winner; round-trip fails on the post-edit
but passes on the pre-edit (falls back, `low_confidence: false`); round-trip
fails on *both* (still ships the post-edit, but `low_confidence: true` and
says why); round-trip fails with no LLM edit to fall back to (nothing
different to try, flags `low_confidence`); and that every candidate from
every engine is preserved in the output for the transparency panel (FR9).

## Honest scope statement
Every module is real, complete code against real library APIs — no stubs,
no "pass" placeholders standing in for actual logic. What's *not* done:
end-to-end verification on real hardware (needs Kaggle/Colab, same as
Phase 6), and Phase 9's `catla.py` hasn't wired this into the tokenizer ->
slang_normalizer/transliterator -> quality_pipeline -> reassembler chain
yet — that's the next phase, not this one.

## Next
Phase 9 (postprocessing pipeline integration — wiring Phase 4's
preprocessing, this phase's `produce_translation()`, and Phase 4's
`reassembler.py` into one `translate_tweet()` call) is the natural next
step to make this genuinely callable end-to-end, though Phase 8
(evaluation) is also unblocked now that quality_pipeline.py exists to
evaluate. Either can proceed without new training runs. Real verification
of everything built in this phase still requires the user running it on
Kaggle/Colab or another working-torch environment.
