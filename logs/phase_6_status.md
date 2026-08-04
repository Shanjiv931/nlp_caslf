# Phase 6 — Specialist Ensemble LoRA Fine-Tuning — Status

**Date:** 2026-08-04

## Environment finding: this machine cannot run any local ML training
Two separate blockers, discovered in this phase:
1. **No NVIDIA/CUDA GPU** — confirmed via `Get-CimInstance Win32_VideoController`:
   only an integrated AMD Radeon 890M. Already anticipated since
   `phase_0_status.md`.
2. **A system Application Control policy actively blocks PyTorch's native
   DLL from loading at all**, even on CPU: installing `torch` (CPU build)
   and running `import torch` failed with `OSError: [WinError 4551] An
   Application Control policy has blocked this file... torch_python.dll`.
   This is a security control on this specific Windows machine, not
   something this session attempted to work around (matches this project's
   own rule against modifying system/security settings) — it means **no**
   local ML execution is possible here at all, not just training. This will
   also affect Phase 7 (sentence-transformers/COMET) and any other
   torch-dependent module later, on this specific machine.
   `torch` was uninstalled again after confirming this, to avoid leaving a
   non-functional multi-GB install around; `transformers`/`peft`/
   `accelerate`/`sentencepiece` remain installed (pure-Python-importable,
   harmless, and will be needed again once genuinely running on Colab/
   Kaggle/another machine).

Given both blockers, Phase 6's actual training was never going to run in
this session — this matches the PRD's own plan (`/notebooks/train_colab.ipynb`
and `/notebooks/train_kaggle.ipynb`, driven by the user across multiple
free-GPU sessions with an active browser tab). What follows is what *was*
achievable here: the training infrastructure itself.

## Done
- **`pipeline/train.py`** — a generic LoRA fine-tuning script shared by all
  three ensemble engines (IndicTrans2, NLLB-200-distilled-600M, BanglaT5),
  both directions. Supports checkpoint-based resume (auto-detects the
  latest `checkpoint-N` in `--output_dir`), optional 4-bit loading, and
  pushing to the HF Hub. Uses `IndicTransToolkit` for AI4Bharat's
  recommended IndicTrans2 preprocessing when installed, with a documented
  fallback to plain tokenization otherwise.
- **Bug caught and fixed before any testing**: the first draft hardcoded
  LoRA `target_modules` using M2M100/NLLB-style attention naming
  (`q_proj`/`v_proj`/...). BanglaT5 is a T5 model, and HF's `T5Attention`
  uses `q`/`k`/`v`/`o` (no `_proj` suffix) — the original code would have
  silently attached LoRA to zero modules for that engine (or errored,
  depending on peft version), while appearing to work fine for the other
  two. Fixed with `default_target_modules()`, auto-selected from
  `--model_name`, overridable via `--lora_target_modules`.
- Verified what could be verified without a working torch install:
  `python -m py_compile pipeline/train.py` (syntax-clean), and the
  torch-independent half of the data pipeline (`load_direction_dataset`'s
  logic, tested standalone against the real `data/processed/train.jsonl` —
  correctly filters to `bn_en_translation` rows and builds a HF `Dataset`).
  The model-loading/LoRA-wrapping/training-loop half could **not** be
  functionally smoke-tested on this machine — that remains genuinely
  unverified until run for real on Colab/Kaggle.
- **`notebooks/train_colab.ipynb`** and **`notebooks/train_kaggle.ipynb`** —
  both install requirements, authenticate to HF Hub (interactively —
  `notebook_login()` on Colab, Kaggle Secrets on Kaggle; no token is ever
  hard-coded), pull `train.jsonl`/`val.jsonl`/`test.jsonl` from the HF Hub
  dataset repo Phase 5 was meant to push, then call `pipeline/train.py`
  once per engine+direction with `--resume --push_to_hub` so a session
  disconnect never loses progress.

## Blocked / needs the user
- **Phase 5's HF Hub push was never completed** (needs `huggingface-cli
  login`, see `phase_5_status.md`) — both notebooks depend on that dataset
  repo existing before they can pull training data. This needs to happen
  first.
- **Actual training has not been run anywhere.** No LoRA adapters exist yet
  in `models/`. The user needs to open one of the two notebooks in Colab or
  Kaggle, fill in `HF_DATASET_REPO`/`HF_MODEL_REPO_PREFIX`, and run all six
  engine+direction combinations (3 engines x 2 directions) themselves —
  this genuinely cannot be done unattended from this session.

## Correction — 2026-08-05, prompted by the user asking "does resume actually work"
Good catch: it didn't, for the case that matters most. `find_latest_checkpoint()`
only ever checked the **local** `--output_dir` for a `checkpoint-N` folder.
That's fine on Colab if Drive is mounted (persists across disconnects), but
on **Kaggle, `/kaggle/working/` is wiped between sessions** — so after the
weekly GPU-quota reset, a fresh Kaggle session would find no local
checkpoint and silently start LoRA from scratch, discarding everything
already trained, even though `--push_to_hub` had been pushing checkpoints
to the Hub the whole time. The Kaggle notebook's own markdown cell already
*claimed* resume would fall back to the Hub — that claim was aspirational,
not something the code actually did.

Fixed in `pipeline/train.py`: `--resume` now tries the local checkpoint
first (exact resume — model + optimizer + LR-schedule state, via Trainer's
`resume_from_checkpoint`), and if none exists locally but `--push_to_hub`
points at an existing Hub repo, loads the LoRA adapter weights from there
instead (`PeftModel.from_pretrained(model, hub_repo, ...)`). Documented
honestly, in both the script's docstring and the Kaggle notebook's markdown
cell: the Hub fallback is an **approximate** resume — Trainer's automatic
Hub push only uploads model/tokenizer files, not optimizer/scheduler state,
so the LR schedule restarts even though the learned weights carry over.
Still far better than losing the trained weights outright, which is what
would have silently happened before this fix.

## Next
Given training itself is blocked pending the user's action, Phase 7 (the
quality layer: ensemble/QE-rerank/LLM-postedit/round-trip-verify) can still
have its *code* written and unit-tested against mocked model outputs in this
session, the same way Phase 6's `train.py` was — the actual end-to-end
pipeline run (which needs real LoRA-fine-tuned models) will be blocked on
the same two issues (no working local torch, no trained models yet) until
the user completes the Colab/Kaggle training and the HF Hub push.
