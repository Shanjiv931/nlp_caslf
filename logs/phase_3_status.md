# Phase 3 — Cleaning, Normalization & Schema Unification — Status

**Date:** 2026-08-04

## Done
- Built `pipeline/clean.py`, merging every Phase 1 raw dataset (plus the
  Phase 2 proxy test set) into one unified JSONL schema at
  `data/processed/unified.jsonl` — see the schema docstring at the top of
  that file for the full field list (`text`/`parallel_text`/`romanized_text`,
  `pair_type`, `register`, `hashtags`/`emojis`/`mentions`, `is_code_mixed`,
  `is_romanized_bn`, `task_labels`, `license`, `is_proxy_test_set`).
- **811,204 unified rows, 512MB.** Breakdown by source:

  | source | rows | pair_type |
  |---|---|---|
  | samanantar_bn | 300,000 (sampled from 8.6M) | bn_en_translation |
  | opus_opensubtitles | 300,000 (sampled from 8.9M) | bn_en_translation |
  | banth | 37,350 | bn_en_translation |
  | banglatlit | 46,864 (see note below) | bn_transliteration |
  | en_bn_code_mixed_sentiment | 100,000 | monolingual_social |
  | bnsentmix | 20,015 | monolingual_social |
  | emomix_3l | 1,071 | monolingual_social |
  | sentmix_3l | 1,007 | monolingual_social |
  | hashset | 1,901 | hashtag_segmentation |
  | collected_tweets_proxy | 2,996 | monolingual_social |

- **Samanantar and OPUS-OpenSubtitles were deliberately sampled** (300k of
  8.6M / 300k of 8.9M respectively), not carried through in full. Rationale:
  (1) IndicTrans2 — the planned primary Phase 6 engine — was itself
  pretrained on Samanantar, so re-fine-tuning on the full raw set is largely
  redundant; (2) this project targets free-tier Colab/Kaggle GPU fine-tuning,
  where several epochs over 17M+ pairs is impractical; (3) the highest-value
  addition for this project's actual gap (informal/social-media register) is
  the smaller, genuinely noisy datasets, which are kept in full. The
  complete raw data remains untouched in `data/raw/` if the user wants to
  scale up later.
- Hashtag/mention/emoji extraction and Bengali+Latin code-mix detection were
  applied to every social/code-mixed source (skipped for Samanantar/OPUS,
  where hashtags/emojis essentially don't occur, to keep processing fast).
- Only HashSet's 1.9k **manually-annotated** segmentations were used, not
  the 3.3M loosely-supervised rows, to keep its contribution proportionate.

## Data-quality finding (BanglaTLit)
`BanglaTLit_train.csv` has 245,727 rows, but **202,863 of them have a null
`text_bengali` column** — only 42,864 train rows actually carry a gold
Bengali-script target (val/test are fully populated: 1,500 / 2,500). This
isn't a bug in the cleaning script; it reflects the dataset's actual shape,
confirmed by direct inspection (`df['text_bengali'].isna().sum()`). Rows
without a gold Bengali-script target were correctly excluded from the
`bn_transliteration` pairs, giving 46,864 (= 42,864 train + 1,500 val + 2,500
test) usable pairs, plus the untouched `BanglaTLit-PT.txt` (245,726 lines) in
`data/raw/` for potential future pretraining use if a use for the
unlabeled-target rows is worked out.

## Not included in the unified corpus
- **`be_cm_ud_bn_en`**: as already flagged in `data/raw/SOURCES.md`, this
  dataset only ships tweet IDs and dependency-parse annotations, not the
  raw tweet text (Twitter ToS-compliant redistribution). With no text, it
  cannot contribute rows to a text-based corpus. Left out rather than
  fabricating placeholder text.

## Corrections carried over from Phase 1
`data/raw/SOURCES.md` was corrected during this phase: BanTH's
`train.csv`/`val.csv`/`test.csv` splits (distinct from `full_with_stats.csv`,
which really does lack bn/en columns) turned out to genuinely contain
`bangla`/`english` parallel columns — 37,350 informal-register bn-en pairs,
now included above. This confirms a detail from an earlier, otherwise-
unverifiable user message that had been provisionally flagged as a possible
mix-up; see `phase_1_status.md` for the original discrepancy note.

## Next
Phase 4 — tweet-aware preprocessing pipeline: `pipeline/tokenizer.py`,
`hashtag_segmenter.py` (can now be trained/tested against the 1,901 HashSet
manual pairs above), `slang_normalizer.py`, `transliterator.py` (trainable
against the 46,864 BanglaTLit pairs), `reassembler.py`, each with unit tests.
