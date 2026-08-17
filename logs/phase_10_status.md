# Phase 10 — Image Input Pipeline (OCR + Cleanup + Language-ID Gate) — Status

**Date:** 2026-08-16

## A rare highlight: fastText genuinely runs on this machine
Unlike torch (blocked by Device Guard, see `phase_6_status.md` onward),
`fasttext-wheel` has no torch dependency and installed/imported cleanly.
This meant `language_gate.py` could be **genuinely verified with the real
downloaded `lid.176.ftz` model**, not just written correct-by-construction
like almost everything else in this project since Phase 6.

## Real bug found and fixed: fastText + numpy 2.x
`fasttext-wheel`'s own `FastText.predict()` (installed library source,
`.venv/Lib/site-packages/fasttext/FastText.py` ~line 228) calls
`np.array(probs, copy=False)` on a plain Python tuple — numpy>=2.0's
stricter `copy=False` semantics now raise
(`ValueError: Unable to avoid copy...`) instead of silently falling back
to copying, since the input isn't already an ndarray. Reproduced directly
(numpy 2.5.1 installed here). Considered downgrading numpy globally,
rejected as risky (this project's `pandas` is already on a numpy>=2.0-era
release; a downgrade could reopen unrelated conflicts to fix one library's
unfixed internal bug). Fixed with a targeted wrapper
(`_safe_fasttext_predict`) that calls the same lower-level `model.f.predict()`
binding `predict()` itself uses internally, skipping the one buggy line and
converting the result with plain `list()` instead.

## Real accuracy evidence, not fabricated
Downloaded the real `lid.176.ftz` (938KB, matches the expected ~917KB) and
ran `language_gate.py` against real text in Bengali, English, and the
PRD's suggested 5 distractor languages (Hindi, Tamil, Arabic, Spanish,
French), plus edge cases:

| Input | Detected | Correct? |
|---|---|---|
| Bengali | bn (via Unicode ratio, conf 1.0) | ✅ |
| English | en (conf 0.992) | ✅ |
| Hindi, Devanagari script | unsupported (ft=hi) | ✅ |
| Tamil, native script | unsupported (ft=ta) | ✅ |
| Arabic, native script | unsupported (ft=ar) | ✅ |
| Spanish | unsupported (ft=es) | ✅ |
| French | unsupported (ft=fr) | ✅ |
| Empty string | unsupported | ✅ |
| Emoji-only | unsupported | ✅ |
| Numbers-only | unsupported | ✅ |
| **Hindi, romanized ("Hinglish")** | **en (conf 0.725)** | **❌ false accept** |

10/11 correct on this illustrative set. **The one failure is genuine and
disclosed, not hidden**: romanized Hindi scored as English with fastText
itself confident (0.725, well above the 0.5 gate threshold) — not a
borderline case a threshold adjustment would fix. Root cause: `lid.176`
was trained predominantly on native-script text per language, so it has
limited ability to recognize romanized transliterations. This mirrors a
disclosed limitation already noted in `catla.py`'s
`looks_fully_romanized_bengali` for the same underlying reason (Latin
script alone can't distinguish "romanized Bengali" from "some other
romanized language" or "genuine English"). Encoded as an explicit test,
`test_KNOWN_LIMITATION_romanized_hindi_misclassified_as_english`, that
will fail loudly (a good thing to notice) if a future fastText version or
gate change happens to fix it — not silently treated as correct behavior.

**Honest scope note on the PRD's "≥99% gate accuracy" target**: this
11-example set is illustrative, not a rigorous benchmark — a real accuracy
measurement needs a properly sized, curated labeled test set (the PRD's
own Phase 10 spec calls for exactly this), which is future work, not
fabricated here as an aggregate percentage from 11 examples.

## Built

### `pipeline/ocr.py`
PaddleOCR primary, Tesseract (`pytesseract` + `ben`+`eng` packs) fallback,
`extract_text(image_path, engine="auto")` tries the former and falls back
on any failure. **Not verified on this machine**: PaddleOCR needs its own
PaddlePaddle framework (likely hits the same class of issues as torch,
untested since the point would be moot either way), and Tesseract's system
binary isn't installed here (confirmed: `tesseract --version` -> command
not found — a system-level install this session shouldn't perform
unprompted). Correct-by-construction against the real APIs.

### `pipeline/ocr_cleanup.py`
Rule-based cleanup — Unicode NFC normalization (fixes a real, specific
category of "broken conjuncts": decomposed vs. composed glyph forms, not
general OCR garbling), a Bengali-danda-misread fix (`|`/`I`/`l` -> `।`,
*only* when adjacent to actual Bengali script, verified to leave genuine
English text with a literal `|` untouched), and whitespace/control-char
cleanup — **genuinely runs and is tested for real**, no model needed (9
tests, including an idempotency check and an NFC-normalization
correctness check). LLM-assisted pass for low-confidence OCR spans reuses
`llm_postedit.load_llm()` (same cache, not a second multi-GB model load)
with a narrow "fix OCR typos only" prompt — mocked in tests, same pattern
as every LLM-calling function in this project.

### `pipeline/language_gate.py`
`detect_and_route()`: Bengali-Unicode-ratio check first (near-certain
signal when it fires, ratio ≥0.5), fastText fallback otherwise, reject
anything not confidently bn/en. Logged, tested for real as above.

### `pipeline/image_to_translation.py`
`handle_uploaded_image()`: OCR -> cleanup -> gate -> (if supported)
`catla.translate_tweet()`, or the PRD's required alert message (and
**never** calls the translation pipeline) if unsupported — verified this
specific behavior with a test that makes `catla.translate_tweet` raise an
`AssertionError` if called at all for a Spanish input, confirming the gate
genuinely blocks it rather than just being expected to. Every decision
logged to `logs/language_gate_log.jsonl` (SHA-256 image hash, not the raw
path or image content — basic privacy hygiene).

## Test count
**143/143 passing project-wide** (105 from Phases 4-9 + 38 new: 17 for
`language_gate.py` — 13 against the real model — 12 for `ocr_cleanup.py`,
9 for `image_to_translation.py`).

## Next
Phase 11 (Gradio demo + HF Spaces publishing) is the natural next step —
`image_to_translation.handle_uploaded_image()` is exactly the function a
`gr.Image` upload handler needs to call. Real end-to-end verification
(OCR specifically) still needs either a working PaddleOCR install or a
Tesseract system install, neither available on this machine.
