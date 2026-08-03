# CATLA — Context-Aware Translation & Localization Algorithm

Bidirectional Bengali <-> English translation of Twitter/X posts submitted as
images, with emoji/hashtag/slang preservation, built entirely from free and
open-source tools, datasets, models, and compute.

## Architecture

1. **Image input** -> OCR (PaddleOCR/Tesseract) -> OCR cleanup -> language
   identification gate (fastText `lid.176` + Unicode-range heuristic). Inputs
   that are not confidently Bengali or English are rejected with a clear
   message rather than mistranslated.
2. **Translation ensemble**: IndicTrans2 (primary, AI4Bharat's specialist
   English<->Indic model), NLLB-200-distilled-600M (generalist backup),
   BanglaT5 (Bengali-native third vote) — each LoRA-fine-tuned on the same
   Twitter-domain data.
3. **Quality layer**: multi-candidate generation -> Quality Estimation
   reranking (COMET-QE / CometKiwi) -> LLM post-editing (Qwen2.5-7B-Instruct /
   Llama-3.1-8B-Instruct / Gemma-2-9B-it, fluency-only edits, protected
   emoji/hashtag/slang tokens) -> round-trip back-translation + LaBSE
   semantic-similarity self-check.
4. **Reassembly**: re-insert protected emoji/hashtag/slang spans into the
   verified output.

See `/logs/phase_0_status.md` onward for the build log of each phase.

## Project layout

```
/data/raw/              raw downloaded datasets
/data/processed/        cleaned, unified-schema data
/data/collected_tweets/ self-collected organic test set (tweet IDs + text only)
/pipeline/               all processing/inference modules
/models/                 LoRA adapters per ensemble engine
/eval/                   evaluation scripts and RESULTS.md
/demo/                   Gradio image-upload demo (app.py)
/logs/                   phase status logs + language_gate_log.jsonl
/notebooks/              Colab/Kaggle training notebooks
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
huggingface-cli login   # free account, needed to push/pull models & datasets
```

If using Tesseract instead of PaddleOCR, install Tesseract itself and its
`ben` (Bengali) + `eng` (English) language data files separately — the
`pytesseract` pip package is only a wrapper around the system binary.

## Constraint

Every tool, dataset, model, and compute resource used in this project is
free / open-source. No paid APIs, no paid compute, no proprietary datasets.
