"""Phase 6 — LoRA fine-tuning for one ensemble engine, one direction.

Generic seq2seq LoRA fine-tuning script shared by all three ensemble
engines (IndicTrans2, NLLB-200-distilled-600M, BanglaT5). Designed to run
on a free Colab/Kaggle T4 GPU (see notebooks/train_colab.ipynb and
notebooks/train_kaggle.ipynb, which call this script) — the machine this was
authored on has no NVIDIA GPU (confirmed via `Get-CimInstance
Win32_VideoController`: only an integrated AMD iGPU), so it was only
smoke-tested for import/argument correctness there. It has since been run
for real on Kaggle (T4 x2) and verified working through LoRA setup, dataset
tokenization, and the start of training.

IndicTrans2 caveat: AI4Bharat's own best-practice preprocessing for
IndicTrans2 uses their `IndicTransToolkit` (script normalization,
sentence-level cleanup) ahead of tokenization, not just a plain
AutoTokenizer call. This script uses IndicTransToolkit automatically if
it's installed (`pip install IndicTransToolkit`) and falls back to plain
AutoTokenizer otherwise — the fallback will run and produce a working
model, but likely at somewhat lower quality than AI4Bharat's own recipe.

Checkpointing / resume: saves every `--save_steps` steps to
`--output_dir/checkpoint-N`. `--resume` tries two things in order:
1. A local `checkpoint-N` folder in `--output_dir` (exact resume — restores
   model, optimizer, and LR-scheduler state, so training continues as if
   nothing happened). This is what Colab gets if you mount Google Drive as
   `--output_dir` (persists across session disconnects).
2. If no local checkpoint exists AND `--push_to_hub` is set, the LoRA
   adapter weights are pulled from that Hub repo instead. This is the path
   Kaggle actually needs: `/kaggle/working/` is wiped between sessions, so
   after a weekly-quota reset there is no local checkpoint to find, only
   whatever was last pushed to the Hub. IMPORTANT CAVEAT: this is an
   approximate resume, not exact — Trainer's automatic hub push only
   uploads model/tokenizer files, not optimizer/scheduler state, so the
   learning-rate schedule and optimizer momentum restart from scratch even
   though the learned weights carry over. Still far better than discarding
   the weights and starting over, but don't expect bit-identical behavior
   to an uninterrupted run.

   Note this step *attempts* the Hub load and falls back to a fresh LoRA
   init on any failure, rather than pre-checking whether the repo "exists":
   Trainer's push_to_hub=True creates the Hub repo the moment Trainer is
   constructed, well before any real checkpoint is pushed — so a repo can
   exist yet still have no adapter_config.json in it (e.g. a previous run
   that crashed before its first --save_steps interval, as happened for
   real during this project's own Kaggle run on 2026-08-05).
"""
import argparse
import glob
import json
import os
import sys
import types

from datasets import Dataset
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

PROCESSED = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

try:
    from IndicTransToolkit.processor import IndicProcessor
    _HAS_INDIC_TOOLKIT = True
except ImportError:
    _HAS_INDIC_TOOLKIT = False


_SPECIAL_TOKEN_LITERALS = ["<unk>", "<pad>", "<s>", "</s>"]


def sanitize_text(text):
    """Strip literal occurrences of common tokenizer special-token strings
    from raw text. Some source datasets embed these as data-collection
    artifacts marking a redacted/unknown word, not meaningful content —
    confirmed directly during Phase 3 unification, a BanTH row's text
    contained a literal "<unk>" (e.g. "Notok kom koro priyo <unk>...").
    Left in place, these coincide with tokenizers' own special-token
    strings: AI4Bharat's custom IndicTransTokenizer assigns `_src_tokenize`
    directly as `self._tokenize` (see tokenization_indictrans.py), which
    the base `PreTrainedTokenizer.tokenize()` calls per-fragment after
    pre-splitting the input around any literal special-token substring —
    so a sentence containing "<unk>" gets split at that boundary, and the
    fragment after it no longer has the "{src_lang} {tgt_lang} " prefix
    tag_prefixing added to the *start* of the original string. Hit for
    real on 2026-08-15: `ValueError: not enough values to unpack (expected
    3, got 1)` inside `_src_tokenize`, 38% through a 150,000-row
    IndicTrans2 training batch. Applied unconditionally (not just for
    IndicTrans2) since training any model on literal "<unk>" as if it were
    meaningful vocabulary is bad signal regardless of tokenizer.
    """
    for tok in _SPECIAL_TOKEN_LITERALS:
        text = text.replace(tok, "")
    return " ".join(text.split())  # collapse whitespace left behind


def load_direction_dataset(path, direction, max_rows=None):
    """direction: 'bn2en' or 'en2bn'. Only bn_en_translation pair_type rows
    are used for MT fine-tuning (transliteration/hashtag/monolingual rows
    are excluded — they're for other pipeline components, not this task)."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("pair_type") != "bn_en_translation":
                continue
            if not row.get("text") or not row.get("parallel_text"):
                continue
            if direction == "bn2en":
                src, tgt = row["text"], row["parallel_text"]  # text is bn, parallel_text is en
            else:
                src, tgt = row["parallel_text"], row["text"]
            src, tgt = sanitize_text(src), sanitize_text(tgt)
            if not src or not tgt:
                continue  # sanitizing could leave an empty string, e.g. a row that was only "<unk>"
            rows.append({"source": src, "target": tgt})
            if max_rows and len(rows) >= max_rows:
                break
    return Dataset.from_list(rows)


def default_target_modules(model_name):
    name = model_name.lower()
    if "t5" in name or "banglat5" in name:
        return ["q", "k", "v", "o"]  # HF T5Attention naming (no "_proj" suffix)
    # M2M100/NLLB/IndicTrans2-style architectures
    return ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]


def ensure_transformers_onnx_shim():
    """Newer `transformers` releases removed the `transformers.onnx`
    submodule (ONNX export moved to the separate `optimum` package), but
    AI4Bharat's IndicTrans2 custom modeling code — loaded dynamically via
    `trust_remote_code=True` — still imports from it at module-load time,
    for an optional ONNX-export code path this project never exercises.
    Hit for real on Kaggle's pre-installed transformers, 2026-08-05 — first
    `ModuleNotFoundError: No module named 'transformers.onnx'` (a plain
    `from transformers.onnx import OnnxConfig, OnnxSeq2SeqConfigWithPast`),
    then, after shimming just that, a second, different missing piece one
    line further down in the same file: `ModuleNotFoundError: No module
    named 'transformers.onnx.utils'; 'transformers.onnx' is not a package`
    (`from transformers.onnx.utils import compute_effective_axis_dimension`
    — the first shim was a bare ModuleType without `__path__`, so Python
    wouldn't treat it as a package capable of having submodules at all).

    Downgrading transformers to a version old enough to still have this
    submodule would risk reopening the torchao/peft version-gate and
    Seq2SeqTrainer tokenizer/processing_class issues already fixed for the
    current version, for the sake of an import path we don't use. Instead
    of patching one missing symbol at a time as more surface (as just
    happened once already), this shims BOTH `transformers.onnx` (as a real
    package, via `__path__`) and `transformers.onnx.utils`, and — since we
    can't be sure this is the last symbol that file needs from either —
    auto-synthesizes *any* attribute requested from them via `__getattr__`,
    as a permissive placeholder usable both as a subclassable base class
    (`OnnxConfig`, `OnnxSeq2SeqConfigWithPast` are subclassed) and as a
    plain callable (`compute_effective_axis_dimension` is called with
    numeric args). This only needs to survive *module-level* class
    definitions and imports — the actual ONNX-export methods are never
    invoked during ordinary `from_pretrained()` + LoRA forward/backward,
    so the placeholders never need to behave correctly, just exist.
    """
    try:
        import transformers.onnx.utils  # noqa: F401
        return  # already importable (older transformers, or already shimmed)
    except ImportError:
        pass

    class _Permissive:
        """Accepts any constructor args and does nothing — safe as a base
        class for subclassing, and safe as the "return value" when called
        like a function, since nothing in the code paths we actually run
        inspects what these placeholders produce."""

        def __init__(self, *args, **kwargs):
            pass

    class _AutoAttrModule(types.ModuleType):
        def __getattr__(self, name):
            value = type(name, (_Permissive,), {})
            setattr(self, name, value)
            return value

    onnx_shim = _AutoAttrModule("transformers.onnx")
    onnx_shim.__path__ = []  # mark as a package so `transformers.onnx.utils` resolves
    utils_shim = _AutoAttrModule("transformers.onnx.utils")
    onnx_shim.utils = utils_shim

    sys.modules["transformers.onnx"] = onnx_shim
    sys.modules["transformers.onnx.utils"] = utils_shim


def ensure_tie_weights_compat():
    """Newer `transformers` refactored how `PreTrainedModel.
    _finalize_load_state_dict` calls `tie_weights()` — it now passes
    `missing_keys=`/`recompute_mapping=` kwargs. AI4Bharat's custom
    `IndicTransForConditionalGeneration` (loaded via `trust_remote_code`)
    overrides `tie_weights()` with an older, argument-less signature, so
    this crashes: `TypeError: IndicTransForConditionalGeneration.
    tie_weights() got an unexpected keyword argument 'missing_keys'`. Since
    it's a full override, not an inherited method, patching the *base*
    PreTrainedModel.tie_weights wouldn't even be consulted — Python's MRO
    resolves straight to the subclass's version.

    A `transformers==4.33.2` pin (the version AI4Bharat's own install.sh
    requires) was tried first and abandoned: it transitively needs an old
    `tokenizers` release with no prebuilt wheel for Python 3.12, and
    building it from source fails with no Rust toolchain available on
    Kaggle. So: patch `_finalize_load_state_dict` itself to make whatever
    model it's finalizing tolerant of an old-style `tie_weights()`, right
    before calling the real (unmodified) original implementation — this
    doesn't skip anything the original method does after that call, it
    just makes the one problematic call succeed. Safe no-op for models
    whose `tie_weights()` already accepts the new kwargs (NLLB, BanglaT5,
    or any other model that doesn't override it), since the signature
    check below only rewraps when the older-style signature is detected.
    """
    import inspect as _inspect

    import transformers.modeling_utils as _mu

    if getattr(_mu.PreTrainedModel, "_catla_tie_weights_patched", False):
        return  # already patched (e.g. a previous call in the same process)

    try:
        raw = _mu.PreTrainedModel.__dict__["_finalize_load_state_dict"]
    except KeyError:
        return  # this transformers version doesn't have this method at all — nothing to patch

    def _make_tie_weights_compatible(model):
        try:
            sig = _inspect.signature(model.tie_weights)
        except (TypeError, ValueError):
            return
        has_var_kwargs = any(p.kind == _inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        if has_var_kwargs or "missing_keys" in sig.parameters:
            return  # already compatible with the new calling convention
        original_tie_weights = model.tie_weights

        def _compat_tie_weights(*args, **kwargs):
            return original_tie_weights()

        model.tie_weights = _compat_tie_weights

    if isinstance(raw, classmethod):
        original_func = raw.__func__

        def wrapper(cls, model, load_config, load_info):
            _make_tie_weights_compatible(model)
            return original_func(cls, model, load_config, load_info)

        _mu.PreTrainedModel._finalize_load_state_dict = classmethod(wrapper)
    elif isinstance(raw, staticmethod):
        original_func = raw.__func__

        def wrapper(model, load_config, load_info):
            _make_tie_weights_compatible(model)
            return original_func(model, load_config, load_info)

        _mu.PreTrainedModel._finalize_load_state_dict = staticmethod(wrapper)
    else:
        def wrapper(self, model, load_config, load_info):
            _make_tie_weights_compatible(model)
            return raw(self, model, load_config, load_info)

        _mu.PreTrainedModel._finalize_load_state_dict = wrapper

    _mu.PreTrainedModel._catla_tie_weights_patched = True


def find_latest_checkpoint(output_dir):
    ckpts = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not ckpts:
        return None
    return max(ckpts, key=lambda p: int(p.rsplit("-", 1)[-1]))


def build_lora_config(args):
    target_modules = (
        args.lora_target_modules.split(",") if args.lora_target_modules
        else default_target_modules(args.model_name)
    )
    return LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        target_modules=target_modules,
        task_type="SEQ_2_SEQ_LM",
    )


def direction_lang_codes(direction):
    """Single source of truth for which language is source vs. target.
    `load_direction_dataset` already puts the correct text in `source`/
    `target` per direction; this must agree with it exactly, or every
    language-tagging step downstream (IndicTrans2's required tag prefix,
    NLLB's tokenizer.src_lang/tgt_lang) silently uses the wrong language --
    which happened for real: the first version of this function had
    src_lang_code/tgt_lang_code swapped relative to `direction` (returned
    "eng_Latn" as the *source* for direction == "bn2en"), caught only when
    it made IndicTrans2's tokenizer crash outright on 2026-08-05. NLLB has
    no such crash to catch a similar mistake by -- see the caller in
    main() for why that direction's already-completed training is at risk.
    """
    if direction == "bn2en":
        return "ben_Beng", "eng_Latn"
    return "eng_Latn", "ben_Beng"


def build_preprocess_fn(tokenizer, direction, max_len, indic_processor, is_indictrans2):
    src_lang_code, tgt_lang_code = direction_lang_codes(direction)

    def preprocess(batch):
        sources = batch["source"]
        targets = batch["target"]
        if indic_processor is not None:
            sources = indic_processor.preprocess_batch(sources, src_lang=src_lang_code, tgt_lang=tgt_lang_code)
            targets = indic_processor.preprocess_batch(targets, src_lang=tgt_lang_code, tgt_lang=src_lang_code)
        elif is_indictrans2:
            # IndicTransTokenizer._src_tokenize() requires every input to
            # already start with "{src_lang} {tgt_lang} " -- normally
            # IndicProcessor's job, but without it installed the tokenizer
            # misreads the sentence's own first word as the language tag
            # and raises (hit for real on 2026-08-05:
            # `AssertionError: Invalid source language tag: <bengali word>`).
            # This is the minimum required tagging, not the full
            # IndicProcessor pipeline (script normalization, sentence
            # splitting, NER/number masking) -- install IndicTransToolkit
            # for AI4Bharat's actual recommended preprocessing.
            sources = [f"{src_lang_code} {tgt_lang_code} {s}" for s in sources]
            targets = [f"{tgt_lang_code} {src_lang_code} {t}" for t in targets]
        model_inputs = tokenizer(sources, max_length=max_len, truncation=True)
        labels = tokenizer(text_target=targets, max_length=max_len, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return preprocess


def main():
    ensure_transformers_onnx_shim()
    ensure_tie_weights_compat()

    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True, help="e.g. ai4bharat/indictrans2-en-indic-1B, facebook/nllb-200-distilled-600M, csebuetnlp/banglat5")
    p.add_argument("--direction", required=True, choices=["bn2en", "en2bn"])
    p.add_argument("--output_dir", required=True)
    p.add_argument("--train_file", default=os.path.join(PROCESSED, "train.jsonl"))
    p.add_argument("--val_file", default=os.path.join(PROCESSED, "val.jsonl"))
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--max_train_rows", type=int, default=None)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_target_modules", default=None,
                    help="Comma-separated module names for LoRA to wrap. If omitted, "
                         "auto-picked from --model_name: T5-family models (e.g. BanglaT5) "
                         "use HF's 'q,k,v,o' attention naming (NOT 'q_proj'/'v_proj', which "
                         "only exist on M2M100/NLLB/IndicTrans2-style architectures) — mixing "
                         "these up silently attaches LoRA to zero modules.")
    p.add_argument("--use_4bit", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--push_to_hub", default=None, help="HF Hub repo id to push checkpoints to")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    quant_kwargs = {}
    if args.use_4bit:
        from transformers import BitsAndBytesConfig
        quant_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype="bfloat16", bnb_4bit_quant_type="nf4",
        )

    resume_path = None  # local checkpoint dir, for exact Trainer-level resume
    if args.resume:
        resume_path = find_latest_checkpoint(args.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, trust_remote_code=True, **quant_kwargs)

    if resume_path:
        print(f"resuming LoRA weights + optimizer/scheduler state from local checkpoint {resume_path}")
        model = PeftModel.from_pretrained(model, resume_path, is_trainable=True)
    else:
        loaded_from_hub = False
        if args.resume and args.push_to_hub:
            # Don't just check the Hub repo *exists* — Trainer's push_to_hub=True
            # creates the repo the moment Trainer is constructed, well before any
            # real checkpoint is pushed (e.g. a previous run that crashed before
            # its first --save_steps interval leaves behind an empty repo that
            # "exists" but has no adapter_config.json). Actually attempt the load
            # and fall back on failure, instead of guessing in advance.
            try:
                model = PeftModel.from_pretrained(model, args.push_to_hub, is_trainable=True)
                print(f"no local checkpoint in {args.output_dir} — resumed LoRA weights from Hub "
                      f"repo {args.push_to_hub} instead (approximate resume: optimizer/LR-schedule "
                      f"state restarts, learned weights carry over)")
                loaded_from_hub = True
            except Exception as e:
                print(f"--resume given, but Hub repo {args.push_to_hub} has no usable adapter "
                      f"checkpoint yet (commonly because Trainer's push_to_hub=True creates the "
                      f"repo immediately at startup, before any checkpoint is pushed — e.g. a "
                      f"previous run that crashed before reaching --save_steps) — starting fresh. "
                      f"({type(e).__name__}: {e})")
        if not loaded_from_hub:
            model = get_peft_model(model, build_lora_config(args))
    model.print_trainable_parameters()

    is_indictrans2 = "indictrans2" in args.model_name.lower()
    is_nllb = "nllb" in args.model_name.lower()
    src_lang_code, tgt_lang_code = direction_lang_codes(args.direction)

    indic_processor = IndicProcessor(inference=False) if (_HAS_INDIC_TOOLKIT and is_indictrans2) else None
    if is_indictrans2 and indic_processor is None:
        print("WARNING: IndicTransToolkit not installed — falling back to plain tokenization for IndicTrans2. "
              "Install with `pip install IndicTransToolkit` for AI4Bharat's recommended preprocessing.")

    if is_nllb:
        # NLLB/M2M100-family tokenizers need to be told which language
        # each example is actually in -- without this they silently use
        # whatever src_lang the tokenizer defaults to (eng_Latn) for every
        # example regardless of direction, which affects both the
        # language-token embedded in the input AND in the labels (so it's
        # not just an inference-time nicety, it changes what the model is
        # trained to produce). Previously not set at all anywhere in this
        # script.
        if hasattr(tokenizer, "src_lang"):
            tokenizer.src_lang = src_lang_code
        if hasattr(tokenizer, "tgt_lang"):
            tokenizer.tgt_lang = tgt_lang_code

    train_ds = load_direction_dataset(args.train_file, args.direction, args.max_train_rows)
    val_ds = load_direction_dataset(args.val_file, args.direction, max_rows=2000)

    preprocess_fn = build_preprocess_fn(tokenizer, args.direction, args.max_len, indic_processor, is_indictrans2)
    train_ds = train_ds.map(preprocess_fn, batched=True, remove_columns=["source", "target"])
    val_ds = val_ds.map(preprocess_fn, batched=True, remove_columns=["source", "target"])

    collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=True,
        gradient_checkpointing=True,
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        logging_steps=50,
        predict_with_generate=True,
        push_to_hub=bool(args.push_to_hub),
        hub_model_id=args.push_to_hub,
        report_to=[],
    )

    # transformers renamed Trainer's `tokenizer=` kwarg to `processing_class=`
    # a few releases back and later removed `tokenizer=` outright (hit for
    # real on Kaggle's pre-installed transformers, 2026-08-05). Inspect the
    # actual signature rather than hardcoding one name, so this works
    # whichever version the current environment happens to have.
    import inspect
    tok_kwarg = "processing_class" if "processing_class" in inspect.signature(Seq2SeqTrainer.__init__).parameters else "tokenizer"

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        **{tok_kwarg: tokenizer},
    )

    trainer.train(resume_from_checkpoint=resume_path if resume_path else None)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    if args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
