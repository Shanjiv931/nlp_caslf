# Phase 9 — Postprocessing Pipeline Integration — Status

**Date:** 2026-08-16

## Real bug found and fixed while building this: hashtags were never actually being translated
`reassembler.py`'s original logic treated a hashtag "already present
verbatim in the MT output" as success — nothing left to do. That was
correct reasoning *before* Phase 7 existed, but `llm_postedit.py`
deliberately puts hashtags in its protected-token list (so the LLM doesn't
mangle them mid-edit), which means the source-language hashtag now
*always* ends up sitting verbatim in the translated sentence by the time
it reaches `reassemble()`. The old logic would therefore never actually
segment/translate/rebuild a hashtag in the success path — directly
contradicting the PRD's FR8 ("hashtags are segmented, translated/
transliterated, and rebuilt as valid target-language hashtags"), silently
leaving a source-language hashtag stuck inside an otherwise fully-
translated sentence.

Fixed: hashtags are now *always* segmented+translated+rebuilt; if the old
verbatim hashtag is present in the MT output it's removed first (not left
duplicated alongside the new one). Mentions/URLs/emoji are unaffected —
those genuinely should stay verbatim (a username or link isn't language-
dependent), so their "preserve as-is" logic is correct and untouched.
2 new regression tests added (`test_reassemble_actually_translates_
hashtag_when_verbatim_in_mt_output`, `test_reassemble_no_translate_fn_
still_rebuilds_not_just_skips`); all 21 Phase 4 tests still pass.

## Built: `pipeline/catla.py`
`translate_tweet(text, direction) -> dict`, the single end-to-end entry
point the Gradio demo (Phase 11) and a CLI both call. Wires:
`slang_normalizer.normalize_slang()` (always, both directions — the
combined Banglish+English dictionary is safe to apply regardless of
direction) -> conditional `transliterator.latin_to_bn()` -> `quality_
pipeline.produce_translation()` -> `reassembler.reassemble()` (using a new
lightweight word-level translator for hashtag rebuilding, not the full
quality pipeline per word — far too expensive).

**Transliteration trigger, deliberately conservative**: only fires for
`bn2en` direction, and only when `looks_fully_romanized_bengali()` finds
Latin script with *zero* Bengali Unicode presence at all. Genuinely
code-mixed input (Bengali script with English words/loanwords mixed in —
common, and what the fine-tuned engines actually trained on via BanTH/
BnSentMix) is deliberately left alone rather than blindly transliterated,
which would corrupt real English content sitting inside otherwise-Bengali
text. Documented explicitly in the module docstring as a real, disclosed
limitation, not silently assumed correct.

**Word-level hashtag translator** (`make_word_translator`): single engine
(default NLLB), beam-search-only, cached per (engine, direction, word) —
applies the same romanization-detection to individual hashtag words before
translating them, for consistency with the main sentence body. Fails safe
(keeps the original word) on any error rather than crashing hashtag
rebuilding — verified with an explicit test that forces `ensemble.
load_engine` to raise and confirms the result still comes back usable.

## Testing
14 new tests for `catla.py` (pure `looks_fully_romanized_bengali` logic —
fully romanized / proper Bengali script / code-mixed / pure English /
edge cases — plus full `translate_tweet()` orchestration with mocked
`quality_pipeline`/`ensemble` calls, including the real fallback path).
2 new regression tests for the reassembler fix. **105/105 tests passing
project-wide.**

One test initially made a real (slow, ~5s) network call to Hugging Face
inside `ensemble.load_engine` before its own try/except caught the
failure — not wrong, but wasteful for a unit test. Fixed by mocking
`make_word_translator` directly for the "happy path" hashtag test, and
added a *separate*, explicitly-named test
(`test_translate_tweet_word_translator_falls_back_on_failure`) that
deliberately exercises the real fallback path via a forced exception —
faster (0.12s for the whole 14-test file, down from ~5s) and clearer about
which test is checking which thing.

## Honest scope statement
`translate_tweet()`'s actual model-calling behavior is unverified on this
machine (same as every other model-calling function built in this
project) — correct-by-construction against the already-built and tested
Phase 4/7 modules it wires together, not run end-to-end. Real verification
needs Kaggle/Colab or another working-torch environment.

## Next
Phase 10 (image input pipeline: OCR + cleanup + language-ID gate) is the
next PRD phase and doesn't depend on any new training — buildable now,
same pattern as everything since Phase 6. Running any of Phases 6-9 for
real still requires the user's Kaggle/Colab session.
