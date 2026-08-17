# CATLA — Final Report

**Context-Aware Translation & Localization Algorithm**
Bidirectional Bengali↔English tweet translation, built entirely on free and
open-source models, datasets, and compute.

**Report date:** 2026-08-17 · **Status:** Phases 0–11 complete, Phase 12 (this
document) in progress · **Repo:** [github.com/Shanjiv931/nlp_caslf](https://github.com/Shanjiv931/nlp_caslf)

---

## 1. Executive Summary

CATLA translates Bengali↔English social-media text (tweets, submitted as
either raw text or a screenshot) while preserving the things generic MT
systems routinely destroy: emoji, hashtags, mentions, URLs, and slang
register. It does this with a 4-stage architecture — OCR/language-gate →
a 3-engine LoRA-fine-tuned MT ensemble → a quality layer (QE reranking,
LLM post-editing, round-trip verification) → protected-token reassembly —
under a hard constraint: every model, dataset, and compute resource used is
free or open-source. No paid APIs, no paid compute, no proprietary data.

All 6 planned LoRA adapters (3 engines × 2 directions) are trained and
verified on the HF Hub. The full quality pipeline, evaluation harness, image
pipeline, and a Gradio demo are built and unit-tested (**153/153 tests
passing**). **Quantitative MT-quality numbers (BLEU/chrF++/TER/COMET/PFS)
are pending** a Kaggle run of `eval/evaluate.py` — see §7 for why, and
§7.1 for how this section gets updated once they land.

## 2. Problem Statement & Scope

General-purpose MT (Google Translate, generic NLLB) is trained on formal,
clean text and predictably fails on tweet-shaped input: it mistranslates or
drops emoji and hashtags, "translates" mentions and URLs it shouldn't touch
at all, and misses code-mixed Banglish and internet slang entirely — none of
it is in its training distribution. CATLA is a specialist system built
specifically for this register, using the same class of models but
fine-tuned on informal, code-mixed, social-media-style Bengali↔English text,
wrapped in a pipeline that treats emoji/hashtags/mentions/slang as
first-class concerns rather than noise to be discarded.

## 3. Architecture

```
Image input ──► OCR (PaddleOCR/Tesseract) ──► OCR cleanup (NFC + danda fix)
                                                       │
Text input  ───────────────────────────────────────┐  ▼
                                                     ▼  Language gate
                                          slang_normalizer  (fastText lid.176 +
                                                     │        Bengali-Unicode ratio)
                                          transliterator          │
                                          (romanized-only)   reject if neither
                                                     │        bn nor en, with a
                                                     ▼        clear message —
                                    ┌──── MT ENSEMBLE ────┐   never mistranslate
                                    │ IndicTrans2 (1B)     │
                                    │ NLLB-200-distilled   │
                                    │ BanglaT5             │
                                    │ (beam + nucleus each)│
                                    └──────────┬───────────┘
                                               ▼
                                     QE reranking (CometKiwi,
                                     falls back to beam log-probs)
                                               ▼
                                     LLM post-edit (Qwen2.5-7B /
                                     Llama-3.1-8B / Gemma-2-9B,
                                     fluency-only, protected tokens)
                                               ▼
                                     Round-trip verify (back-translate
                                     + LaBSE cosine similarity)
                                               ▼
                                     Reassembly (re-insert protected
                                     emoji/mention/URL spans, rebuild
                                     hashtags in target language)
                                               ▼
                                         Final translation
```

Every stage's design rationale, failure modes, and fallback behavior is
documented in that module's own docstring (`pipeline/*.py`) and in the
per-phase build logs (`logs/phase_N_status.md`) — this report summarizes,
it doesn't replace them.

## 4. Dataset

### 4.1 Sources and composition

811,204 rows unified from 10 verified free/open sources into one schema
(`data/processed/unified.jsonl`, 512MB) — full source table with exact
license per dataset in [`data/raw/SOURCES.md`](data/raw/SOURCES.md):

| Source | Rows used | Type | License |
|---|---|---|---|
| Samanantar (bn) | 300,000 (sampled from 8.6M) | bn↔en translation | CC-BY-NC-4.0 |
| OPUS OpenSubtitles (bn-en) | 300,000 (sampled from 8.9M) | bn↔en translation | Per-file, OPUS redistribution terms |
| BanTH | 37,350 | bn↔en translation (informal, YouTube) | MIT |
| BanglaTLit | 46,864 | Bengali transliteration | MIT |
| En-Bn code-mixed sentiment | 100,000 | monolingual social | MIT |
| BnSentMix | 20,015 | monolingual social | MIT |
| EmoMix-3L | 1,071 | monolingual social (bn-en-hi code-mixed) | GPL-3.0 |
| SentMix-3L | 1,007 | monolingual social (bn-en-hi code-mixed) | AGPL-3.0 |
| HashSet (manual subset only) | 1,901 | hashtag segmentation | Research use |
| Self-collected proxy test set | 2,996 | monolingual social | — (resampled from above) |

Samanantar and OPUS-OpenSubtitles were deliberately capped at 300k of
8.6M/8.9M respectively — IndicTrans2 was already pretrained on Samanantar,
free-tier GPU budgets make multi-epoch runs over 17M+ pairs impractical, and
the actual gap this project targets (informal register) is better served by
keeping the smaller, genuinely noisy datasets in full. Full raw data remains
untouched in `data/raw/` for future scale-up.

**Not included, disclosed honestly rather than fabricated:** FLORES-200
(gated HF repo, needs manual access request), WAT Indic Multilingual Corpus
(mixed/unclear licensing across sub-corpora), IndicCorp v2/Naamapadam
(monolingual only, no bn-en pairs). "BE-CM" from the original PRD reading
list has no exact published match; `UD_bn-en` was used as the closest
verified Bengali-English code-mixed Twitter resource, but it ships only
tweet IDs and dependency-parse annotations (Twitter ToS-compliant, no raw
text), so it could not contribute rows to the text corpus.

### 4.2 Splits

Stratified 85/7.5/7.5 split by source dataset (`pipeline/split_and_augment.py`),
with the self-collected proxy organic test set kept **entirely in TEST**
per the PRD's requirement to hold out organic data:

| Split | Rows | Composition |
|---|---|---|
| train | 717,929 | 686,973 original + 30,956 synthetic code-mixed |
| val | 60,613 | stratified sample |
| test | 63,618 | 60,622 stratified + 2,996 organic-proxy |
| **total** | **842,160** | |

**Synthetic code-mixing augmentation** (train only): 8% of eligible
`bn_en_translation` rows had ~30% of their Bengali target words romanized
via the rule-based transliterator, producing Banglish-style variants. Zero
synthetic rows leak into val/test (verified directly). Documented as a
practical heuristic, not linguistically-validated code-switching.

Published publicly: [huggingface.co/datasets/shanjivkr/catla-bn-en-tweets](https://huggingface.co/datasets/shanjivkr/catla-bn-en-tweets).

### 4.3 Local-only supplementary data (not part of the free claim)

`data/collected_tweets/live_eval_LOCAL_ONLY.jsonl` — 2,640 real, live tweets
(1,348 Bengali, 1,292 English) collected via `twitterapi.io`'s paid API, at
the user's own means and explicit choice. **Excluded from git, from the
public HF dataset, from training, and from any "100% free" claim** — local
evaluation use only, per an explicit decision made when this data was
introduced. See `data/raw/twitterapi_io_live_tweets/README_NOT_FREE.md`.

## 5. Model Training (Phase 6)

Three specialist engines, both directions, LoRA fine-tuned on
`train.jsonl`'s `bn_en_translation` rows via `pipeline/train.py`, driven from
free Kaggle/Colab GPU sessions (this project's own dev machine has no CUDA
GPU and a Device Guard policy that blocks even CPU torch — see
[`logs/phase_6_status.md`](logs/phase_6_status.md)).

| Engine | Base model | bn2en | en2bn |
|---|---|---|---|
| IndicTrans2 | `ai4bharat/indictrans2-{indic-en,en-indic}-1B` (direction-specific checkpoints) | ✅ `shanjivkr/catla-indictrans2-bn2en` | ✅ `shanjivkr/catla-indictrans2-en2bn` |
| NLLB-200-distilled-600M | `facebook/nllb-200-distilled-600M` | ✅ `shanjivkr/catla-nllb-bn2en` | ✅ `shanjivkr/catla-nllb-en2bn` |
| BanglaT5 | `csebuetnlp/banglat5` | ✅ `shanjivkr/catla-banglat5-bn2en` | ✅ `shanjivkr/catla-banglat5-en2bn` |

All 6 adapters verified directly against the HF Hub (real
`adapter_config.json`/`adapter_model.safetensors`, sane sizes, fresh
timestamps) — never taken on a training log's claim alone, since `Trainer`'s
`push_to_hub=True` creates the Hub repo empty at construction time, a real
bug class hit and fixed during this phase.

### 5.1 Real engineering obstacles resolved (abbreviated — full detail in the phase log)

Getting IndicTrans2 training to run at all took **7 debugging rounds**, each
a genuine root-cause fix, not a workaround: a `torchao`/`peft` version-gate
crash, a `Trainer` kwarg rename (`tokenizer=` → `processing_class=`), a
`DataParallel` OOM on T4×2 (fixed by pinning to a single GPU), a resume-logic
flaw where `push_to_hub=True` creates an empty Hub repo the trainer then
mistook for "already trained", missing `transformers.onnx`/`.utils`
submodules AI4Bharat's remote code still imports, a `tie_weights()` signature
mismatch between newer `transformers` and IndicTrans2's custom model class,
swapped source/target language codes plus NLLB never having
`tokenizer.src_lang`/`tgt_lang` set at all (both fixed; NLLB was retrained to
apply the fix), and a literal `<unk>` string appearing in real BanTH data
that corrupted IndicTrans2's custom tokenizer's language-tag parsing. See
[`logs/phase_6_status.md`](logs/phase_6_status.md) for the full
investigation of each, including the ones read directly from AI4Bharat's
source rather than guessed from stack traces alone.

### 5.2 Disclosed compromises (not hidden, flagged explicitly)

- **1 epoch, 150k-row cap per direction** (not the full ~570k+ available) —
  a deliberate feasibility tradeoff against free-tier GPU-hour budgets
  (unconstrained defaults were estimated at months of GPU time across all 6
  combinations).
- **IndicTrans2 trained without `IndicTransToolkit`** (AI4Bharat's
  recommended preprocessing — script normalization, sentence splitting,
  NER/number masking), using this project's own minimal tag-prefix fallback
  instead. Root cause investigated and fixed at the code level (a namespace
  collision between this project's own now-removed `indic-nlp-library`
  dependency and `IndicTransToolkit`'s `indic-nlp-library-itt`), but **not
  yet applied** — deferred by explicit user choice pending Phase 8
  evaluation results, to decide whether the quality gap justifies a retrain.
- **BanglaT5 has no prior translation pretraining** (unlike NLLB/IndicTrans2,
  which are pretrained MT models), so 1 epoch on top of a non-MT base risks
  undertraining it specifically relative to the other two — flagged as a
  likely weak ensemble member, not yet confirmed either way by real
  evaluation numbers.
- No validation-based checkpoint selection (last-checkpoint used).

## 6. Quality Layer (Phase 7)

`pipeline/quality_pipeline.py`'s `produce_translation()` — where the actual
accuracy claim lives, not in any single fine-tuned model's raw decode:

1. **Ensemble generation** (`ensemble.py`) — up to ~10 diverse candidates per
   request: beam search + nucleus sampling, across all 3 engines.
2. **QE reranking** (`qe_rerank.py`) — reference-free CometKiwi picks the
   best candidate; falls back to the beam search's own log-probs if no QE
   model loads, with unscored (sampled) candidates always ranked below any
   scored one.
3. **LLM post-editing** (`llm_postedit.py`) — Qwen2.5-7B-Instruct (then
   Llama-3.1-8B, then Gemma-2-9B as fallbacks), constrained to fluency-only
   edits with emoji/hashtags/mentions/URLs (from Phase 4's
   `tokenizer.protected_spans()`) explicitly protected. Falls back to the
   pre-edit candidate if the protected-token contract is violated or the
   edit is empty.
4. **Round-trip verification** (`roundtrip_verify.py`) — back-translates the
   result and compares to the original via LaBSE cosine similarity. On
   failure: falls back to the pre-edit candidate if *it* round-trips
   successfully (isolating whether the LLM edit was the problem); otherwise
   ships the result flagged `low_confidence: true` rather than hiding a
   failed check.
5. **Reassembly** (`reassembler.py`) — protected spans re-inserted;
   hashtags segmented, translated, and rebuilt in the target language (a
   real bug — hashtags silently never being translated once step 3's
   protected-token contract existed — was found and fixed here, see
   [`logs/phase_9_status.md`](logs/phase_9_status.md)).

Every fallback/decision branch above is unit-tested with mocked model
outputs (52 tests across `ensemble.py`/`qe_rerank.py`/`llm_postedit.py`/
`roundtrip_verify.py`/`quality_pipeline.py`) — the actual model-calling code
paths are correct-by-construction against the real APIs but unexercised on
this machine (no working torch environment locally).

## 7. Evaluation (Phase 8)

`eval/evaluate.py` compares three modes over the same held-out test pairs:
`pretrained_baseline` (base model, no LoRA — isolates fine-tuning's
contribution), `single_model_baseline` (one fine-tuned engine, beam search
only), and `full_pipeline` (the complete Phase 7 chain) — computing
BLEU/chrF++/TER (`sacrebleu`), reference-based COMET
(`Unbabel/wmt22-comet-da`), and the Platform-Fidelity Score
(`eval/platform_fidelity.py` — emoji/mention/URL preservation + hashtag-count
preservation). The `full_pipeline − single_model_baseline` delta is the
central claim this architecture needs to prove.

### 7.1 Results — PENDING real Kaggle run

**No fabricated or estimated numbers appear in this report.** As of this
writing, `eval/evaluate.py` is mid-run on Kaggle (`--direction bn2en
--max_examples 200`); this section will be updated with the real
BLEU/chrF++/TER/COMET/PFS table and the full_pipeline-vs-baseline delta as
soon as the run completes and results are shared back. The evaluation
infrastructure itself is genuinely verified locally where it can be
(`sacrebleu`, `platform_fidelity.py`, and `load_test_pairs()` against the
real `test.jsonl` all run for real, no torch needed — 89/89 tests passing
at the time this infra was built) — only the actual model-calling numbers
are pending.

### 7.2 Platform-Fidelity Score — honest scope note

PFS measures three genuinely-automatic sub-dimensions: emoji preservation
(set-based), verbatim mention/URL preservation (exact-match), and hashtag
*count* preservation (not translation *correctness* — no automatic ground
truth exists for hashtag-translation naturalness). It deliberately does
**not** score the PRD's fourth stated dimension (slang naturalness) — there
is no honest automatic way to measure that without a reference or human
judgment, and a fabricated proxy score would misrepresent what's being
measured. Slang quality belongs in worked-example qualitative review, not
the numeric PFS.

## 8. Language Detection Gate (Phase 10)

`pipeline/language_gate.py`: Bengali-Unicode-ratio check first (near-certain
signal when it fires), fastText `lid.176` fallback otherwise, rejects
anything not confidently Bengali or English. **Genuinely verified against
the real downloaded `lid.176.ftz` model** (unlike almost everything else in
this project, since fastText has no torch dependency):

| Input | Detected | Correct? |
|---|---|---|
| Bengali | bn (Unicode ratio, conf 1.0) | ✅ |
| English | en (conf 0.992) | ✅ |
| Hindi (Devanagari) | unsupported | ✅ |
| Tamil (native script) | unsupported | ✅ |
| Arabic (native script) | unsupported | ✅ |
| Spanish | unsupported | ✅ |
| French | unsupported | ✅ |
| Empty string | unsupported | ✅ |
| Emoji-only | unsupported | ✅ |
| Numbers-only | unsupported | ✅ |
| Romanized Hindi ("Hinglish") | en (conf 0.725) | ❌ known limitation |

**10/11 correct** on this illustrative set (not a rigorous benchmark — a
real accuracy measurement needs a properly sized, curated labeled test set,
which is future work). The one failure is genuine and disclosed, not
hidden: `lid.176` was trained predominantly on native-script text per
language, so romanized transliterations (Bengali or otherwise) are its
documented weak point — encoded as an explicit test that will fail loudly
if a future fix changes this behavior, rather than silently accepted as
correct.

A real fastText+numpy 2.x compatibility bug (`predict()` calling a numpy API
broken by numpy 2.x's stricter `copy=False` semantics) was found and fixed
with a targeted wrapper rather than a global numpy downgrade.

## 9. Demo & Deployment (Phase 11)

`demo/app.py` — a Gradio app with two tabs sharing a mode/engine control:

- **📷 Translate Image** (primary, per the PRD) — `gr.Image` upload wired to
  `image_to_translation.handle_uploaded_image()`: OCR → cleanup → language
  gate → (if supported) full translation, or the required rejection alert
  otherwise, with OCR text, detected language + confidence, winning engine,
  QE score, round-trip similarity, and a low-confidence badge all surfaced.
- **⌨️ Translate Text** — direct text input, running the identical language
  gate before translating.
- **Quality mode** (default) — the full Phase 7 pipeline. **Fast mode** —
  a single beam-search decode from one engine only, for usable latency on
  free CPU-only HF Spaces hardware.

Genuinely verified where possible (Gradio has no torch dependency): the
`gr.Blocks` app builds cleanly, the unsupported-language rejection path
runs for real against the real fastText model, and result formatting is
verified against mocked pipeline output for both modes.

`pipeline/push_to_hf_space.py` (not run automatically — needs
`huggingface-cli login`, not set up in this environment) publishes the demo
to a free HF Space; `demo/README.md` carries the Space's config front matter.

## 10. Test Coverage

**153/153 tests passing project-wide**, grown incrementally per phase:

| Phase | New tests | Running total |
|---|---|---|
| 4 — preprocessing | 19 | 19 |
| 6/7 — mt_compat refactor + quality layer | 52 | 71 |
| 8 — evaluation infra | 18 | 89 |
| 9 — end-to-end pipeline + reassembler fix | 16 | 105 |
| 10 — image input pipeline | 38 | 143 |
| 11 — demo fast-mode plumbing | 5 | 148 |
| 11 (bugfix) — `transformers.onnx` shim dunder-safety | 5 | 153 |

Everything that can run without a working torch install genuinely does —
`sacrebleu`, `platform_fidelity.py`, `language_gate.py` against the real
fastText model, `ocr_cleanup.py`'s rule-based logic, `mt_compat.py`'s
`ensure_transformers_onnx_shim()` (transformers imports without torch, so
this got real verification too), and Gradio's `Blocks` construction. Every
model-calling function (LoRA loading, generation, QE, LLM post-edit,
round-trip verification, OCR) is correct-by-construction against the real
library APIs, with its orchestration logic thoroughly unit-tested against
mocked model outputs, but not exercised end-to-end on this machine — that
requires Kaggle/Colab/HF Spaces, consistently disclosed throughout every
phase log rather than glossed over.

## 11. Free/Open-Source Compliance

Every tool, dataset, model, and compute resource used to build and run
CATLA is free or open-source:

- **Models**: IndicTrans2, NLLB-200, BanglaT5, CometKiwi, Qwen2.5/
  Llama-3.1/Gemma-2 (LLM post-editor fallback chain), LaBSE, fastText
  `lid.176` — all open-weight, freely downloadable from the HF Hub.
- **Datasets**: all 10 sources in §4.1 are free (one, FLORES-200, is gated
  but still free — not pursued, not paid).
- **Compute**: free-tier Kaggle/Colab GPU sessions for training; this
  project's own dev machine (no GPU) for everything else.
- **The one exception, explicitly excluded from every claim above**: 2,640
  live tweets collected via a paid third-party API (`twitterapi.io`), kept
  local-only, gitignored, never used for training, never published — see
  §4.3.

## 12. Known Limitations (consolidated)

- Quantitative MT-quality numbers not yet available (§7.1).
- IndicTrans2 trained without its author-recommended preprocessing toolkit
  (§5.2) — fix ready, retrain deferred pending eval results.
- BanglaT5 has no prior MT pretraining and may be undertrained at 1 epoch
  (§5.2) — not yet confirmed by real numbers.
- Romanized (non-Bengali) language input can be misclassified as English by
  the language gate (§8) — a fastText training-data limitation, not a
  threshold problem.
- Hashtag *translation naturalness* (as opposed to count preservation) has
  no automatic metric (§7.2) — requires human/qualitative review.
- OCR (`pipeline/ocr.py`) is correct-by-construction but unverified on this
  machine — needs a working PaddleOCR or system Tesseract install.
- The self-collected "organic" test set (Phase 2) is a proxy — resampled
  noisy/social-register rows from Phase 1's own datasets, not genuinely
  live-scraped tweets (`snscrape` is unmaintained and non-functional
  post-2023; free-tier X API access has no search capability). The
  separate, real live-tweet set (§4.3) exists but is local-eval-only by
  policy, not part of the published pipeline.
- No validation-based checkpoint selection during LoRA training.

## 13. Reproducibility

```bash
git clone https://github.com/Shanjiv931/nlp_caslf.git
cd nlp_caslf
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
huggingface-cli login   # free account
pytest pipeline/tests eval/tests -q   # 153/153, no GPU needed

# Training (needs a GPU — Kaggle/Colab notebooks provided):
#   notebooks/train_kaggle.ipynb or train_colab.ipynb

# Evaluation (needs a GPU):
python eval/evaluate.py --direction bn2en --max_examples 200
python eval/evaluate.py --direction en2bn --max_examples 200

# Demo (needs a GPU for real translations; UI itself runs anywhere):
python demo/app.py
```

## 14. Future Work

1. Run `eval/evaluate.py` for real, fill in §7.1 with actual numbers.
2. Decide on IndicTrans2 retrain-with-`IndicTransToolkit` based on those
   numbers.
3. Build a proper, curated language-gate accuracy benchmark (§8's 11-example
   set is illustrative, not rigorous).
4. Human/qualitative review of hashtag and slang translation naturalness.
5. Verify OCR (`pipeline/ocr.py`) and the full image pipeline end-to-end
   once a working PaddleOCR/Tesseract environment is available.
6. If evaluation shows a clear winning single engine, consider whether the
   3-engine ensemble's added latency is worth its quality delta for the
   demo's fast-mode default choice.
