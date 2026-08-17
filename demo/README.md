---
title: CATLA Bengali-English Tweet Translator
emoji: 🐦
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.24.0
app_file: demo/app.py
pinned: false
license: mit
---

# CATLA — Context-Aware Translation & Localization Algorithm

Bidirectional Bengali↔English tweet translation, built entirely on free and
open-source models and free-tier compute (see the project's `README.md` for
the full architecture: OCR → language gate → 3-engine LoRA-fine-tuned
ensemble → QE reranking → LLM post-edit → round-trip verification).

**Primary flow**: upload a tweet screenshot. It's OCR'd, language-detected,
and (if Bengali or English) translated. Anything else is explicitly
rejected with a clear message rather than mistranslated. A "Translate Text"
tab is also available for pasting text directly.

**Quality mode** (default) runs the full pipeline above — slower, highest
quality. **Fast mode** runs a single fine-tuned model, beam search only —
usable on this Space's free CPU-basic hardware without a multi-minute wait
per request.

Source: [github.com/Shanjiv931/nlp_caslf](https://github.com/Shanjiv931/nlp_caslf).
Model adapters and dataset: [huggingface.co/shanjivkr](https://huggingface.co/shanjivkr).
