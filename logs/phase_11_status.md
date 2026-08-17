# Phase 11 — Gradio Demo & HF Spaces Publishing — Status

**Date:** 2026-08-17

## What was built

### `fast_mode` plumbed end-to-end (prerequisite for the demo toggle)
The PRD requires the demo to offer a "fast mode" (single model, skip
QE/LLM-postedit/round-trip-verify) vs. "quality mode" (full pipeline)
toggle. Rather than reimplementing a second translation path inside the
demo, this was pushed down into the shared pipeline so every caller gets
the exact same definition of "fast":

- `ensemble.translate_single()` (added, shared by two callers) — single
  engine, beam-search-only, no ensemble diversity/QE/LLM/round-trip.
- `eval/evaluate.py`'s `translate_single_model_baseline()` now calls it
  instead of duplicating the same logic — Phase 8's "single model" baseline
  and Phase 11's "fast mode" are now provably the same thing, not two
  subtly different reimplementations.
- `catla.translate_tweet(..., fast_mode=False)` — new parameter. When
  `True`, skips `quality_pipeline.produce_translation()` entirely and
  builds an equivalent-shaped result dict from `translate_single()`
  directly (`qe_score`/`roundtrip_similarity`=None, `winning_method`=
  `"beam_fast_mode"`, `low_confidence`=False, an explicit
  `verification_note` disclosing what was skipped) — so every caller
  (demo, CLI, `image_to_translation.py`) gets the same dict shape
  regardless of mode and doesn't need mode-specific branching downstream.
- `catla.py`'s CLI gained `--fast_mode` and `--roundtrip_engine` flags.
- `image_to_translation.handle_uploaded_image()` gained `fast_mode` and
  `roundtrip_engine` passthrough parameters.

### `demo/app.py` — the Gradio demo
Two tabs sharing a mode toggle (`Quality mode` default / `Fast mode`) and
an engine dropdown (`nllb` default / `indictrans2` / `banglat5` — the
fast-mode engine, or the round-trip-verification engine in quality mode):

- **📷 Translate Image (primary)** — `gr.Image(type="filepath")` upload,
  wired straight to `image_to_translation.handle_uploaded_image()`.
  Displays OCR text + OCR confidence, detected language + confidence, and
  on success the translation plus the full quality-layer detail panel
  (winning engine, QE score, round-trip similarity, low-confidence badge,
  verification notes). On an unsupported language, shows the PRD's
  required alert message and nothing else — the pipeline is never invoked
  (inherited from Phase 10's `handle_uploaded_image`, which already
  guarantees this).
- **⌨️ Translate Text** — direct text input for testing/when OCR isn't
  needed. Runs the same `language_gate.detect_and_route()` gate before
  calling `catla.translate_tweet()`, so both tabs make identical
  accept/reject decisions on the same input.

### `demo/README.md` + `pipeline/push_to_hf_space.py` — HF Spaces publishing
`demo/README.md` carries the Space config front matter (`sdk: gradio`,
`sdk_version: 6.24.0`, `app_file: demo/app.py`). `push_to_hf_space.py`
follows the exact same pattern as Phase 5's `push_to_hf.py` (NOT run
automatically — needs `huggingface-cli login` first, which isn't set up in
this environment): uploads `demo/README.md`, `demo/app.py`,
`requirements.txt`, and the whole `pipeline/` folder (app.py imports from
it via a relative `sys.path` insert, so the Space's layout mirrors this
repo's). LoRA adapters and the dataset are deliberately *not* re-uploaded
into the Space — they're pulled from the HF Hub at request time, already
public since Phases 5/6.

## Real verification this time — a genuine highlight
Unlike almost everything model-calling in this project, **Gradio itself
has no torch dependency**, so this phase got real, not just
correct-by-construction, verification:

1. `import app; app.demo` — the `gr.Blocks` app genuinely builds with no
   errors, both callbacks correctly wired to their declared inputs/outputs
   (confirmed via `Blocks`' own introspection output).
2. `translate_text_tab()`'s unsupported-language path was run for real
   against real Spanish input, using the real downloaded `lid.176.ftz`
   model (same one Phase 10 verified) — correctly returned the alert
   message and an empty translation, without ever calling
   `catla.translate_tweet` (never mocked out — the real gate genuinely
   ran and rejected it).
3. Empty-input handling verified for real.
4. The supported-language formatting path (`_format_meta`, language line,
   translation display) was verified with `catla.translate_tweet`
   monkeypatched at the module level (the model call itself still needs
   torch) — confirmed correct field mapping for both quality mode (QE
   score, round-trip similarity, LLM-post-edit note all shown) and fast
   mode (both correctly show "skipped (fast mode)").

What's still unverified on this machine: the OCR-driven image tab
end-to-end (needs PaddleOCR/Tesseract, same gap as Phase 10) and any path
that actually calls a fine-tuned model (needs torch — Kaggle/Colab/the HF
Space itself once deployed).

## Test count
**148/148 passing project-wide** (143 from Phases 4-10 + 5 new: 4 for
`catla.py`'s `fast_mode` branching, 1 for `image_to_translation.py`'s
`fast_mode`/`roundtrip_engine` passthrough).

## Next
Phase 12 — documentation: full `README.md` polish and `FINAL_REPORT.md`
(dataset composition, per-engine scores once Phase 8 is actually run on
Kaggle, quality-layer delta, language-gate accuracy, example translations
including a rejected unsupported-language case). Also still pending from
earlier phases, not blocking Phase 11/12: running `eval/evaluate.py` for
real on Kaggle for actual BLEU/chrF/TER/COMET/PFS numbers, and the
deferred IndicTrans2-with-IndicTransToolkit retrain.
