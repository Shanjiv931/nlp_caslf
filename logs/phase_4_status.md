# Phase 4 — Tweet-Aware Preprocessing Pipeline — Status

**Date:** 2026-08-04

## Done
Built five modules under `pipeline/`, each with unit tests on hand-written
example tweets in `pipeline/tests/test_phase4_preprocessing.py` (**19/19
passing**, run via `pytest pipeline/tests/ -v`):

- **`tokenizer.py`** — splits tweets into `(token, kind)` pairs, keeping
  hashtags, mentions, URLs, and emoji as atomic tokens (never split
  mid-token). `protected_spans()` returns exactly the set of substrings
  Phase 7's LLM post-editor must be told never to alter.
- **`hashtag_segmenter.py`** — segments a hashtag into constituent words.
  Tries CamelCase splitting first (exact whenever the author used it), then
  falls back to Viterbi word segmentation (the classic `wordninja`-style
  algorithm, self-implemented) over a Bengali+English word-frequency
  dictionary built from our own corpus (`data/raw/samanantar_bn`,
  `opus_opensubtitles`), cached to `data/processed/word_freq.json` (not
  committed — regenerates automatically on first use, ~4.8MB, derived data).
  Honest limitation: the Viterbi fallback is a frequency heuristic, not a
  trained segmenter — it can misparse novel Banglish blends absent from the
  dictionary (e.g. "duitattoo" -> "du it at too" instead of "dui tattoo" in
  manual testing); CamelCase-authored hashtags segment exactly.
- **`slang_normalizer.py`** — hand-curated Banglish + English internet
  slang/abbreviation dictionary (~45 entries), case-preserving, whole-word
  matching only. Explicitly scoped as a starting set, not claimed exhaustive.
- **`transliterator.py`** — rule-based (not trained) bidirectional Bengali
  script <-> Banglish phonetic transliterator, in the spirit of open phonetic
  input schemes like Avro. Documented limitation: Bengali orthography
  doesn't map 1:1 onto informal Latin spelling, which varies per typist, so
  this won't exactly reproduce anyone's actual spelling — it's a
  deterministic baseline, not evaluated for exact-match accuracy. A trained
  seq2seq transliterator using the 46,864 aligned BanglaTLit pairs
  (`data/raw/banglatlit/`, see `phase_3_status.md`) remains a legitimate
  upgrade path better suited to Phase 6's fine-tuning infrastructure.
- **`reassembler.py`** — reinserts protected tokens (emoji/mentions/URLs)
  that a translation step dropped, and rebuilds hashtags in the target
  language from segmented+translated words (CamelCase for English, joined
  for Bengali, matching real Bengali-Twitter hashtag convention).

## Dependencies added
`pytest` added to `requirements.txt` (already had `regex`/`emoji`, installed
during this phase).

## Next
Phase 5 — dataset splitting & augmentation: 85/7.5/7.5 train/val/test split
of `data/processed/unified.jsonl`, with the Phase 2 proxy organic test set
(`is_proxy_test_set: true` rows) kept entirely in TEST, synthetic
code-mixing augmentation on TRAIN only, and pushing the final splits to a
free public Hugging Face Hub dataset repo.
