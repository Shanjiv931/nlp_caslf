# CATLA — Context-Aware Translation & Localization Algorithm

Bidirectional Bengali↔English translation of Twitter/X posts — submitted as
raw text or as a screenshot — with emoji/hashtag/mention/slang preservation,
built entirely from free and open-source tools, datasets, models, and
compute. No paid APIs, no paid compute, no proprietary datasets.

**Status: Phases 0–11 complete** (all 6 LoRA adapters trained and
Hub-verified, full quality pipeline, evaluation harness, image pipeline,
and Gradio demo built, 153/153 unit tests passing). See
[`FINAL_REPORT.md`](FINAL_REPORT.md) for the full writeup — dataset
composition, training details, architecture rationale, disclosed
limitations, and evaluation results (pending a real Kaggle run — see that
document's §7.1 for status). Every phase's detailed build log lives in
[`logs/phase_0_status.md`](logs/phase_0_status.md) through
[`logs/phase_11_status.md`](logs/phase_11_status.md).

## Architecture

1. **Input**: an image (OCR'd via PaddleOCR/Tesseract, then cleaned up) or
   raw text, either way passed through a **language identification gate**
   (fastText `lid.176` + a Bengali-Unicode-ratio heuristic). Anything not
   confidently Bengali or English is rejected with a clear message rather
   than mistranslated or guessed.
2. **Translation ensemble**: IndicTrans2 (AI4Bharat's specialist
   English↔Indic model), NLLB-200-distilled-600M (generalist backup),
   BanglaT5 (Bengali-native third vote) — each LoRA-fine-tuned on the same
   Twitter-domain data, both directions.
3. **Quality layer**: multi-candidate generation (beam + nucleus sampling,
   all 3 engines) → Quality Estimation reranking (CometKiwi) → LLM
   post-editing (Qwen2.5-7B-Instruct / Llama-3.1-8B-Instruct /
   Gemma-2-9B-it, fluency-only edits, protected emoji/hashtag/mention/URL
   tokens) → round-trip back-translation + LaBSE semantic-similarity
   self-check.
4. **Reassembly**: re-insert protected spans, rebuild hashtags in the
   target language.

## Project layout

```
/data/raw/              raw downloaded datasets (SOURCES.md has licenses)
/data/processed/        cleaned, unified-schema data + train/val/test splits
/data/collected_tweets/ self-collected proxy test set + local-only live-eval set
/pipeline/               all processing/inference modules + tests
/models/                 local artifacts (LoRA adapters live on the HF Hub)
/eval/                   evaluation scripts, methodology, and results
/demo/                   Gradio app (app.py) + HF Spaces config (README.md)
/logs/                   phase status logs + language_gate_log.jsonl
/notebooks/              Colab/Kaggle training notebooks
FINAL_REPORT.md          full project writeup
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

## Running the tests

```bash
pytest pipeline/tests eval/tests -q
```

153/153 pass without a GPU — everything that can be verified without torch
genuinely is (pure text logic, mocked-model orchestration, and a few
modules — `sacrebleu`, `platform_fidelity.py`, the real fastText language
gate — that run for real). Model-calling code (LoRA loading, generation,
QE, LLM post-edit, OCR) needs a working torch environment (Kaggle/Colab/HF
Spaces) to exercise end-to-end.

## Running the demo

```bash
python demo/app.py
```

Two tabs (image upload, primary; direct text input) sharing a **quality
mode** (default — full pipeline) / **fast mode** (single model, low-latency)
toggle. To publish to a free HF Space:

```bash
python pipeline/push_to_hf_space.py --repo-id <your-username>/catla-demo
```

## Running evaluation

```bash
python eval/evaluate.py --direction bn2en --max_examples 200
python eval/evaluate.py --direction en2bn --max_examples 200
```

Needs a GPU (Kaggle/Colab) for the model-calling modes. See
[`eval/RESULTS.md`](eval/RESULTS.md) for methodology and
[`FINAL_REPORT.md`](FINAL_REPORT.md) §7 for results once a run completes.

## Constraint

Every tool, dataset, model, and compute resource used in this project is
free / open-source. No paid APIs, no paid compute, no proprietary datasets.
The one disclosed exception (a small local-only evaluation set collected via
a paid third-party API) is excluded from git, from the public dataset, from
training, and from this claim — see `FINAL_REPORT.md` §4.3.
