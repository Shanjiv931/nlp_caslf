# Phase 0 — Project Setup — Status

**Date:** 2026-08-03

## Done
- Created folder structure: `data/{raw,processed,collected_tweets}`,
  `pipeline/`, `models/`, `eval/`, `demo/`, `logs/`, `notebooks/`.
- Created `requirements.txt` with all planned free/open-source dependencies
  (transformers, datasets, peft, bitsandbytes, accelerate, sentencepiece,
  sacrebleu, evaluate, unbabel-comet, sentence-transformers, snscrape,
  emoji, indic-nlp-library, regex, tqdm, huggingface_hub, gradio, pandas,
  scikit-learn, torch, paddleocr, pytesseract, fasttext-wheel).
- Created `.gitignore` (excludes venv, raw/processed data payloads, model
  weights, and the language-gate log — keeps the repo itself light).
- Initialized local git repo, added remote `origin` ->
  https://github.com/Shanjiv931/nlp_caslf.git (confirmed empty/reachable via
  `git ls-remote`, not yet pushed — pushing needs explicit go-ahead).
- Created `.venv` Python virtual environment (Python 3.12.10 confirmed on
  this machine via the `python` launcher — note: `python3` is NOT on PATH
  here, only `python`; all setup docs/scripts for this machine should use
  `python`/`.venv\Scripts\python.exe`).
- Wrote project `README.md` describing the architecture and layout.

## Environment notes / risks flagged
- **No GPU confirmed on this machine.** Phase 6 (LoRA fine-tuning of three
  translation engines) and any local LLM post-editor inference are far more
  practical on free Colab/Kaggle T4 GPUs than on this Windows machine's CPU.
  The plan already accounts for this via `/notebooks/train_colab.ipynb` and
  `/notebooks/train_kaggle.ipynb` (Phase 6) driving free-GPU sessions.
- **bitsandbytes** has historically had weak/partial Windows support (it's
  primarily targeted at Linux CUDA). If 4-bit/8-bit loading fails locally on
  Windows, fall back to fp16/fp32 CPU loading for lightweight stages (OCR,
  language-ID, LaBSE embedding) and reserve quantized/LoRA training for the
  Colab/Kaggle notebooks.
- **paddleocr** has a large first-run model download and native deps; if it
  fails to install cleanly on Windows, fall back to `pytesseract` + a manual
  Tesseract-OCR Windows installer with the `ben`+`eng` language data files
  (documented in README).
- Dependencies have **not yet been installed** — `requirements.txt` lists
  several GB of ML packages (torch, transformers, paddleocr, etc.) plus
  large first-run model downloads. Installation was deferred pending
  confirmation of scope/pace for the next phases, since this is a
  significant, non-trivial download.
- Nothing has been pushed to GitHub yet — local commits only, by design
  (pushing to a shared remote requires explicit confirmation).

## Next
Phase 1 — dataset discovery & download (Samanantar, FLORES-200 bn/en,
BnSentMix, SentMix-3L, EmoMix-3L, BanglaTLit, BanTH, BE-CM, HashSet, etc.)
into `/data/raw/`, logged to `/data/raw/SOURCES.md`.
