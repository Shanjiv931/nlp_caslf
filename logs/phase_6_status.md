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

## First engine complete — 2026-08-05
**NLLB-200-distilled-600M, both directions, confirmed done.** Verified
directly against the Hub (`HfApi.model_info(..., files_metadata=True)`),
not just taking the training run's own claim at face value:

| Repo | adapter_model.safetensors | Last modified |
|---|---|---|
| `shanjivkr/catla-nllb-bn2en` | 34.6 MB | 2026-08-04 20:14 |
| `shanjivkr/catla-nllb-en2bn` | 34.6 MB | 2026-08-05 06:50 |

Both have real `adapter_config.json` + `adapter_model.safetensors` (not the
empty-repo case fixed earlier), sizes consistent with the reported 8.65M
trainable LoRA params. **2 of 6 engine+direction combinations done.**
Remaining: `indictrans2` (bn2en, en2bn), `banglat5` (bn2en, en2bn),
`nllb`'s two are done.

## IndicTrans2 attempt — two real bugs caught before wasting GPU time — 2026-08-05
1. **Gated repo**: `ai4bharat/indictrans2-en-indic-1B` requires accepting
   access terms while logged in (same class of gate as FLORES-200 in Phase
   1, but this one is the "share contact info" self-serve kind — checked
   the model page directly rather than assuming, confirmed no manual-review
   language, should be instant). Both IndicTrans2 repos need this:
   https://huggingface.co/ai4bharat/indictrans2-indic-en-1B and
   https://huggingface.co/ai4bharat/indictrans2-en-indic-1B.
2. **Real bug in both notebooks, caught proactively**: `ENGINES["indictrans2"]`
   only listed `en-indic-1B` (English->Indic), used for *both* directions.
   Unlike NLLB/BanglaT5 (single bidirectional checkpoints), IndicTrans2
   ships as two direction-specific models — `indictrans2-indic-en-1B` for
   bn2en, `indictrans2-en-indic-1B` for en2bn. Using the wrong one as the
   base for `bn2en` fine-tuning wouldn't have crashed; it would have
   silently fine-tuned from a base model pretrained for the opposite
   direction the entire run, likely producing a materially worse result
   without any error to signal it. This was caught by inspecting the
   notebooks before the user hit it in practice (the gating error surfaced
   it), not by running into a failure from it directly.

Fixed: `ENGINES["indictrans2"]` is now `{"bn2en": "...indic-en-1B",
"en2bn": "...en-indic-1B"}` (a dict, unlike the plain string for the other
two engines), and the training cell resolves `model_name` via
`_engine_entry[direction] if isinstance(_engine_entry, dict) else
_engine_entry` so both dict-per-direction and plain-string engines work
through the same cell unchanged.

## IndicTrans2 attempt, second try — gating resolved, new import error — 2026-08-05
User requested access to both gated repos; download proceeded fully this
time (tokenizer + custom `tokenization_indictrans.py`/`configuration_indictrans.py`
all fetched). New failure loading the model config via `trust_remote_code`:
`ModuleNotFoundError: No module named 'transformers.onnx'`. Root cause:
AI4Bharat's custom `configuration_indictrans.py` does `from transformers.onnx
import OnnxConfig, OnnxSeq2SeqConfigWithPast` at import time, but newer
`transformers` releases removed that submodule entirely (ONNX export moved
to the separate `optimum` package). This import is almost certainly only
needed for an optional ONNX-export code path this project never exercises
during ordinary `from_pretrained()` + LoRA fine-tuning.

Considered downgrading `transformers` to a version old enough to still have
`transformers.onnx`, rejected: that risks reopening the torchao/peft
version-gate issue and the `Seq2SeqTrainer` `tokenizer`/`processing_class`
rename, both already fixed against the *current* transformers version, for
the sake of one unused import in someone else's remote code.

Fixed instead with `ensure_transformers_onnx_shim()` in `pipeline/train.py`:
injects a minimal placeholder `transformers.onnx` module (via
`sys.modules`) with no-op `OnnxConfig`/`OnnxSeq2SeqConfigWithPast` classes,
satisfying just the import — called unconditionally at the top of `main()`,
harmless no-op for the other two engines (which never hit this import path
in the first place). Not yet re-verified end-to-end on Kaggle.

## IndicTrans2 attempt, third try — shim was incomplete, generalized — 2026-08-05
The first shim fixed the `OnnxConfig`/`OnnxSeq2SeqConfigWithPast` import,
but `configuration_indictrans.py` needed a *second*, different thing one
line further down: `from transformers.onnx.utils import
compute_effective_axis_dimension`, failing with `ModuleNotFoundError: No
module named 'transformers.onnx.utils'; 'transformers.onnx' is not a
package` — the original shim was a bare `types.ModuleType` without
`__path__` set, so Python wouldn't treat it as a package capable of
resolving submodules at all, regardless of what's inside it.

Rather than keep discovering and patching one missing symbol at a time as
more of that file's imports surface (each round costs the user a real
Kaggle run), rewrote `ensure_transformers_onnx_shim()` to be general:
shims `transformers.onnx` as a proper package (`__path__ = []`) *and*
`transformers.onnx.utils`, with a `_AutoAttrModule` that synthesizes *any*
requested attribute via `__getattr__` — a permissive placeholder class
usable both as a subclassable base (`OnnxConfig`/`OnnxSeq2SeqConfigWithPast`
are subclassed) and as a plain callable
(`compute_effective_axis_dimension` is called with numeric args). This
only needs to survive module-level class definitions/imports, since the
actual ONNX-export methods are never invoked during ordinary
`from_pretrained()` + LoRA forward/backward.

Verified the import mechanics standalone before shipping (this machine
can't run real `transformers`, so tested the exact same shim logic against
a fake package name instead): `from X.onnx import A, B`,
`from X.onnx.utils import C`, subclassing `B`, and calling `C(...)` all
resolved correctly through Python's real import system. Not yet verified
against the real IndicTrans2 remote code on Kaggle, but this removes the
need for further single-symbol patching regardless of what else that file
imports from `transformers.onnx*`.

## IndicTrans2 attempt, fourth try — a third distinct incompatibility, decided to stop chasing them individually — 2026-08-05
Shim fix worked (config + tokenizer + `modeling_indictrans.py` all
downloaded, model weights loaded — `model.safetensors: 100%|...| 4.09G`,
`Loading weights: 100%|...| 763/763`). New, different failure:
`TypeError: IndicTransForConditionalGeneration.tie_weights() got an
unexpected keyword argument 'missing_keys'`. Root cause: newer
`transformers` refactored how `PreTrainedModel._finalize_load_state_dict`
calls `tie_weights()` (now passes `missing_keys=`/`recompute_mapping=`
kwargs), but AI4Bharat's custom `IndicTransForConditionalGeneration`
overrides `tie_weights()` with an older, no-argument signature — Python's
MRO resolves straight to that override, so there's no way to fix this by
patching the base `PreTrainedModel` class; the override fully shadows it.

This is the **third** distinct IndicTrans2-remote-code compatibility break
in a row (ONNX imports x2, now this), each only surfacing after fixing the
previous one and spending another real Kaggle run to discover it. Checked
AI4Bharat's own `install.sh`
(https://github.com/AI4Bharat/IndicTrans2/blob/main/huggingface_interface/install.sh):
pins `transformers>=4.33.2` — an old, open-ended floor with no upper
bound. Both breaking changes (ONNX submodule removal, `tie_weights()`
signature refactor) happened in `transformers` releases well after that,
which their custom code was never updated to handle. Given the pattern,
concluded that patching individual internal API mismatches one at a time
isn't the fix — pinning to the version they actually tested against is.

Fix: both notebooks' training cell now does `!pip install -q
"transformers==4.33.2"`, conditional on `engine_key == "indictrans2"` only
— NLLB (already trained, both directions) and BanglaT5 (not yet run) keep
using whatever transformers Kaggle/Colab ships by default, since neither
hit any of these three issues. `train.py`'s existing shims/fixes
(`ensure_transformers_onnx_shim()`, the `inspect.signature`-based
Trainer-kwarg detection) are left in place, not removed — both already
no-op cleanly when the underlying compatibility issue isn't present (e.g.
`ensure_transformers_onnx_shim()`'s `try: import transformers.onnx.utils;
return` short-circuits immediately if the real submodule already exists at
4.33.2), so they cost nothing to keep as a safety net.

## IndicTrans2 attempt, fifth try — the pin was a dead end, fixed at the code level instead — 2026-08-05
`!pip install -q "transformers==4.33.2"` failed outright:
`error: subprocess-exited-with-error` / `Building wheel for tokenizers
(pyproject.toml) did not run successfully`. Root cause: transformers
4.33.2 transitively requires an old `tokenizers` release that has no
prebuilt wheel for Python 3.12 (Kaggle's Python version), so pip falls
back to building it from source — which needs a Rust toolchain, not
available in Kaggle's default image. Pip's failed build left the
originally-installed (newer) transformers untouched, so the script ran
anyway and hit the *exact same* `tie_weights()` error as before — the pin
never actually took effect at any point.

Abandoned the version-pin approach entirely (not viable in this
environment) and fixed it directly at the code level instead:
`ensure_tie_weights_compat()` in `pipeline/train.py`, called alongside
`ensure_transformers_onnx_shim()` at the top of `main()`. Monkey-patches
`transformers.modeling_utils.PreTrainedModel._finalize_load_state_dict`
(detecting whether the original is a `classmethod`/`staticmethod`/plain
method and preserving that exact type, since guessing wrong would break
every model, not just IndicTrans2) to check the *specific model instance's*
`tie_weights` signature right before calling the real, unmodified original
implementation — if it doesn't accept `missing_keys`/`**kwargs` (the
old-style override AI4Bharat's custom class has), the instance's bound
`tie_weights` is transparently replaced with a wrapper that calls the real
no-arg version instead. Deliberately does NOT catch-and-skip the exception
at the call site (an earlier design considered and rejected): that would
have skipped whatever `_finalize_load_state_dict` does *after* the
`tie_weights()` call, which could leave the model in a subtly
under-initialized state without any error to signal it. This approach lets
the original method run to completion unmodified, just with one internal
call made compatible.

Verified end-to-end with a standalone mock (real `transformers` doesn't
run on this machine): reproduced the exact `TypeError` against a fake
`PreTrainedModel`/subclass pair shaped like the real
`IndicTransForConditionalGeneration` bug, confirmed the patch fixes it,
confirmed logic *after* the `tie_weights()` call in the original method
still executes and its return value is preserved, and confirmed a model
whose `tie_weights()` already accepts the new kwargs (representative of
NLLB/BanglaT5) passes its real argument values through completely
unaffected by the patch. Reverted the dead-end pip-pin notebook cells back
to the plain training command. Not yet verified against the real
IndicTrans2 remote code on Kaggle — this fix lives in `pipeline/train.py`,
so a plain re-run (fresh `git clone`) picks it up automatically, no
notebook edit needed this time.

## IndicTrans2 attempt, sixth try — two language-tagging bugs found, one affects already-completed NLLB training — 2026-08-05
The `tie_weights` patch worked (model loaded, LoRA applied: `trainable
params: 17,694,720 || all params: 1,040,701,440`, resume-fallback correctly
handled an again-empty Hub repo). New failure, this time in tokenization:
`AssertionError: Invalid source language tag: <bengali word>` inside
AI4Bharat's custom `tokenization_indictrans.py`. Root cause, two distinct
bugs found together:

1. **`src_lang_code`/`tgt_lang_code` were swapped relative to `direction`**
   in `build_preprocess_fn` (`src_lang_code = "ben_Beng" if direction ==
   "en2bn" else "eng_Latn"` — backwards for both branches). Would have
   silently fed IndicProcessor the wrong language pair had
   IndicTransToolkit actually been installed; instead surfaced as a hard
   crash because of bug 2.
2. **No fallback tagging when IndicTransToolkit isn't installed.**
   `IndicTransTokenizer._src_tokenize()` requires every input string to
   already start with `"{src_lang} {tgt_lang} "` — normally IndicProcessor's
   job. Without it (confirmed not installed in this Kaggle session — the
   `WARNING: IndicTransToolkit not installed` line printed right before the
   crash), the tokenizer tries to parse the sentence's own first word as
   the language tag and fails.

Fixed both: added `direction_lang_codes()` as the single source of truth
mapping `direction` -> `(src_lang_code, tgt_lang_code)` (replacing the
inline swapped logic), and `build_preprocess_fn` now does the minimum
required manual tag-prefixing (`f"{src} {tgt} {text}"`) when
`indic_processor` is unavailable, rather than crashing — still recommends
installing IndicTransToolkit for AI4Bharat's full preprocessing (script
normalization, sentence splitting, NER/number masking), but no longer hard
depends on it. Verified the corrected logic standalone (can't run real
`transformers`/`torch` here): both directions map to the correct language
pair, and the manual tag-prefix format matches what the tokenizer expects.

**More significant finding, while fixing this: NLLB's tokenizer never had
`src_lang`/`tgt_lang` set anywhere in the script, for either already-
completed direction.** NLLB/M2M100-family tokenizers need this set
explicitly — it affects not just inference-time generation but which
language token gets embedded in both the *input* and the *label* text
during training itself. Without it, the tokenizer silently uses whatever
it defaults to (commonly `eng_Latn`) regardless of the actual example's
language. Honest assessment of the risk, not overstated in either
direction: `en2bn`'s *source* text is genuinely English, so that side may
have coincidentally matched the tokenizer's default — but `tgt_lang` was
*also* never set, so the *label*-side tagging is uncertain for **both**
already-completed NLLB directions, not just `bn2en`. Fixed going forward:
`main()` now sets `tokenizer.src_lang`/`tokenizer.tgt_lang` explicitly from
`direction_lang_codes()` whenever `is_nllb` is detected, guarded by
`hasattr()` so it's a no-op for tokenizers without these attributes
(IndicTrans2's custom tokenizer, BanglaT5's plain T5 tokenizer).

**This is a judgment call for the user, not something to decide
unilaterally**: whether the two already-completed and Hub-pushed NLLB
adapters (`shanjivkr/catla-nllb-bn2en`, `shanjivkr/catla-nllb-en2bn`) are
worth retraining now that this is fixed, given the ~4 hours of GPU time
that would cost, versus accepting the quality risk and moving on. Flagged
to the user directly rather than assumed.

## NLLB retrained with the language-tag fix — 2026-08-15
User opted to retrain both directions rather than accept the risk. Both
confirmed complete via direct Hub verification (not taken on faith):

| Repo | adapter_model.safetensors | Last modified |
|---|---|---|
| `shanjivkr/catla-nllb-bn2en` | 34.6 MB | 2026-08-15 12:40:08 |
| `shanjivkr/catla-nllb-en2bn` | 34.6 MB | 2026-08-15 14:44:01 |

Both timestamps fresh (well after the original 2026-08-04 runs), sizes
match the expected adapter size. **NLLB, both directions, done with
correct `tokenizer.src_lang`/`tokenizer.tgt_lang` handling in place.**

## IndicTrans2 attempt, seventh try — root cause finally read from source, not guessed — 2026-08-15
Furthest yet: model loaded, LoRA applied, `.map()` tokenization ran 38%
through (57,000/150,000 examples) before: `ValueError: not enough values
to unpack (expected 3, got 1)` inside `_src_tokenize`
(`src_lang, tgt_lang, text = text.split(" ", 2)`).

Six previous rounds on this engine were fixed by reasoning from stack
traces and error messages alone. This one, downloaded and read the actual
`tokenization_indictrans.py` source directly (via
`hf_hub_download(repo_id='ai4bharat/indictrans2-indic-en-1B',
filename='tokenization_indictrans.py')`, since WebFetch can't reach a
gated repo) rather than guessing further. Root cause, precisely: AI4Bharat's
`IndicTransTokenizer._switch_to_input_mode()` assigns `self._tokenize =
self._src_tokenize` directly (line 146) — so whatever `PreTrainedTokenizer
.tokenize()`'s base implementation passes to `self._tokenize()` per
fragment, after its own special-token-boundary pre-splitting, `_src_tokenize`
receives verbatim. Any sentence containing a literal occurrence of one of
the tokenizer's own special-token strings (`<unk>`, `<pad>`, `<s>`,
`</s>`) gets split at that boundary before `_src_tokenize` ever sees it —
and the fragment *after* the boundary no longer has the
`"{src_lang} {tgt_lang} "` prefix, since that was only ever prepended once
to the start of the original string.

This ties directly to something already observed firsthand in this
project: Phase 3's row-inspection of BanTH data showed literal `<unk>` text
(`"Notok kom koro priyo <unk>👻🥣🤣"`) — a data-collection artifact in
BanTH marking a redacted/unknown word, not meaningful content. That's
exactly what's triggering this.

Fixed generally, not just for IndicTrans2: added `sanitize_text()` in
`pipeline/train.py`, applied to every source/target string in
`load_direction_dataset()` regardless of engine — strips literal
`<unk>`/`<pad>`/`<s>`/`</s>` occurrences (collapsing the whitespace left
behind), and skips any row that sanitizes down to an empty string. Applied
unconditionally rather than only for IndicTrans2, since training any model
on a literal "<unk>" as if it were meaningful vocabulary is bad signal
regardless of tokenizer, not just a crash risk for this one.

**Smaller, separate honesty note**: the already-completed NLLB training
(both directions, including the just-finished retrain) ran *without* this
sanitization — it wouldn't have crashed NLLB's tokenizer the way it did
IndicTrans2's (NLLB doesn't have this custom `_tokenize` reassignment), but
a handful of examples containing literal `<unk>` as raw text is still a
minor, diffuse data-quality issue, not a systematic one like the earlier
language-tag bug. Not recommending another retrain for this alone — noting
it for completeness, the same way the language-tag issue was surfaced
rather than silently accepted or silently ignored.

Verified `sanitize_text()` standalone (stripped `<unk>` correctly, and the
resulting tag-prefixed string now round-trips through `.split(" ", 2)`
into exactly 3 parts). Not yet verified against the real IndicTrans2 remote
code on Kaggle — lives in `pipeline/train.py`, picked up by a plain re-run.

## Next
4 combinations remain (`indictrans2` x2 directions, `banglat5` x2
directions) — user-driven. In parallel, Phase 7 (the quality layer:
ensemble/QE-rerank/LLM-postedit/round-trip-verify) can have its *code*
written and unit-tested against mocked model outputs in this session, the
same way Phase 6's `train.py` was — a full end-to-end pipeline run needs
all 6 adapters plus a working local torch, neither of which exist yet, but
per-module code + tests don't need to wait for that.
