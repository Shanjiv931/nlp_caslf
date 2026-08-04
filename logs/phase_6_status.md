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

## Reduced scope — 2026-08-05
User asked for a realistic time estimate. Math with the original defaults
(`--epochs 3`, full ~572,700 `bn_en_translation` rows in `train.jsonl`,
`--batch_size 8 --grad_accum 4`): ~215,000 forward/backward passes per
engine+direction run, roughly 15-40+ hours each depending on model size —
i.e. potentially **months** across all 6 combinations against a ~30
GPU-hr/week free quota. Reduced both notebooks' training cell to
`--epochs 1 --max_train_rows 150000`, cutting this to a rough 2-6 hours per
combination / ~15-25 hours total — a deliberate quality-for-feasibility
tradeoff (LoRA's limited capacity means the marginal value of the extra
data/epochs was shrinking anyway, per the earlier discussion when the user
asked whether the 300k Samanantar/OPUS caps should be raised).

## Real training run, first attempt — 2026-08-05
User started the first real training run on Kaggle (`nllb`/`bn2en`,
T4 x2). Confirmed working: `--resume` correctly detected no local/Hub
checkpoint yet and started fresh (exactly the fix from above, validated for
real); model downloaded fine. Then hit an environment issue blocking
**all** six combinations (happens inside the generic `get_peft_model()`
call, before any model-specific code runs): Kaggle's base image ships
`torchao 0.10.0`, but the installed `peft` version's LoRA module dispatcher
has a version gate that raises `ImportError` for any torchao below 0.16.0,
instead of gracefully skipping that optional integration — even though
plain LoRA fine-tuning never uses torchao at all here.

Fixed by adding `!pip uninstall -y torchao -q` to both notebooks'
install cell, right after `IndicTransToolkit`. Uninstalling entirely
(rather than trying to pin an upgrade) routes `peft`'s check into its
normal "not installed" path, which returns `False` cleanly, rather than
the "installed but too old" path, which raises — the safer fix since an
upgrade-version target couldn't be verified against Kaggle's actual
environment without testing there directly.

## Real training run, second attempt — 2026-08-05
With the torchao fix in place, got much further: LoRA attached correctly to
NLLB (`trainable params: 8,650,752 || all params: 1,410,789,376 ||
trainable%: 0.6132` — confirms `default_target_modules()`'s M2M100-style
`q_proj`/`k_proj`/`v_proj`/`out_proj`/`fc1`/`fc2` list actually matches
NLLB's real module names, not just in theory), both train (150,000 rows)
and val (2,000 rows) datasets tokenized successfully. Then hit:
`TypeError: Seq2SeqTrainer.__init__() got an unexpected keyword argument
'tokenizer'` — Kaggle's pre-installed `transformers` has fully removed the
old `tokenizer=` argument to `Trainer`/`Seq2SeqTrainer` (renamed to
`processing_class=` a few releases back, then the old name was dropped
entirely, not just deprecated-with-warning).

Fixed by inspecting `Seq2SeqTrainer.__init__`'s actual signature at runtime
(`inspect.signature(...).parameters`) rather than hardcoding either kwarg
name, so `train.py` works whichever `transformers` version an environment
happens to have — Kaggle's current one, an older pinned one elsewhere, or
whatever Colab ships.

## Real training run, third attempt — 2026-08-05
Got further still: LoRA setup and dataset tokenization (150,000 train /
2,000 val rows) both succeeded again, training actually **started**
(`0%| | 0/2344 [00:00<?, ?it/s]`) and ran a few seconds into the first
backward pass before failing:
`torch.OutOfMemoryError: CUDA out of memory... GPU 0 has a total capacity
of 14.56 GiB of which 6.81 MiB is free`. The stack trace pinpoints the
cause precisely: it's failing inside `torch/nn/parallel/_functions.py`'s
`reduce_add_coalesced`/`_flatten_dense_tensors` — `DataParallel`'s
gradient-gathering mechanism. On a T4 x2 session, `transformers`' `Trainer`
auto-wraps the model in naive `DataParallel` across both GPUs, but
`DataParallel` funnels loss computation and gradient-gathering through GPU
0 specifically, so GPU 0 carries a disproportionate memory load and can OOM
even though the *combined* memory across both T4s (~29GB) would be plenty.

This corrects earlier advice given to the user (when asked "T4x2 or P100")
that the second GPU would give a "free" (if imperfect) speedup via
automatic `DataParallel` — that's true in principle but doesn't hold up
under this specific OOM failure mode in practice.

Fixed by restricting both notebooks' training cell to a single GPU
(`CUDA_VISIBLE_DEVICES=0` prefix on the `!python pipeline/train.py` call),
sidestepping the imbalance entirely rather than trying to tune batch size
around it. Single T4 still has Tensor Cores (unlike P100), so there was no
reason to recommend switching accelerators — just to stop using the second
GPU via the broken automatic path.

## Real training run, fourth attempt — root cause found and fixed — 2026-08-05
Single-GPU fix worked (OOM gone). New failure on this attempt, in the
resume-fallback code from the "correction" section above:
`ValueError: Can't find 'adapter_config.json' at 'shanjivkr/catla-nllb-bn2en'`.
Root cause: `Seq2SeqTrainingArguments(push_to_hub=True, ...)` makes
`Trainer` **create the Hub repo the moment `Trainer.__init__` runs** — not
when the first checkpoint is actually pushed. The third attempt's OOM
happened during the very first backward pass, before `--save_steps 500` was
ever reached, so `Trainer` had already created `shanjivkr/catla-nllb-bn2en`
but never pushed any adapter files into it. The resume code's check
(`repo_exists(args.push_to_hub, repo_type="model")`) is exactly the wrong
signal here — the repo genuinely "exists," it's just empty.

Real fix (this was a design flaw, not a version/environment quirk like the
earlier three — worth getting right rather than patching around): removed
the `repo_exists()` pre-check entirely. `pipeline/train.py` now just
*attempts* `PeftModel.from_pretrained(model, args.push_to_hub, ...)`
directly and catches any failure, falling back to a fresh LoRA init
(factored into a shared `build_lora_config()` helper to avoid duplicating
that logic across the try/except branches). This is robust to the actual
failure mode (empty-but-existing repo) and to anything else that could go
wrong with the Hub load, without needing to enumerate specific error cases
in advance.

## Next
Given training itself is blocked pending the user's action, Phase 7 (the
quality layer: ensemble/QE-rerank/LLM-postedit/round-trip-verify) can still
have its *code* written and unit-tested against mocked model outputs in this
session, the same way Phase 6's `train.py` was — the actual end-to-end
pipeline run (which needs real LoRA-fine-tuned models) will be blocked on
the same two issues (no working local torch, no trained models yet) until
the user completes the Colab/Kaggle training and the HF Hub push.
