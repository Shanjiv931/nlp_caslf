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

Environment-compatibility shims, language-tagging, and sanitization logic
were factored out to mt_compat.py on 2026-08-16 so Phase 7's inference code
(ensemble.py etc.) can reuse the identical preprocessing this script uses
for training — see mt_compat.py's module docstring.
"""
import argparse
import glob
import json
import os

from datasets import Dataset
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from mt_compat import (
    HAS_INDIC_TOOLKIT as _HAS_INDIC_TOOLKIT,
    INDIC_TOOLKIT_IMPORT_ERROR as _INDIC_TOOLKIT_IMPORT_ERROR,
    default_target_modules,
    direction_lang_codes,
    ensure_tie_weights_compat,
    ensure_transformers_onnx_shim,
    get_indic_processor,
    sanitize_text,
    tag_indictrans2_text,
)

PROCESSED = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


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


def find_latest_checkpoint(output_dir):
    ckpts = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not ckpts:
        return None
    return max(ckpts, key=lambda p: int(p.rsplit("-", 1)[-1]))


def find_nan_lora_params(peft_model):
    """Return the names of every LoRA parameter tensor containing NaN/Inf.

    A checkpoint can "exist" at the right file size/shape and still be
    worthless: a real, confirmed failure mode (2026-08-22) is an unstable
    training run silently writing an adapter where every single value in
    every tensor is NaN (576/576 tensors, 17,694,720/17,694,720 params, for
    catla-indictrans2-bn2en specifically) -- resuming from that guarantees
    every subsequent forward pass is NaN too, regardless of learning rate or
    warmup, since NaN parameters never recover once loaded. File-existence/
    size checks (used everywhere else in this project to verify a training
    run) never catch this; only actually inspecting the values does. Only
    the LoRA delta params are checked (not the frozen base model) -- cheap,
    and the only params a broken training run could actually have corrupted.

    Used for BOTH resume paths (local checkpoint and Hub fallback, see
    module docstring) -- a corrupted checkpoint is corrupted regardless of
    which of the two places it was loaded from, and only checking one path
    (as an earlier version of this script did) leaves the other free to
    silently burn an entire GPU-hour quota re-training on top of NaN
    weights, producing `loss: 0, grad_norm: nan` at every single step with
    no early warning -- exactly what happened resuming from a local
    checkpoint-3000 on 2026-08-23, which this check would have caught in
    seconds instead.
    """
    import torch
    return [name for name, p in peft_model.named_parameters()
            if "lora_" in name and (torch.isnan(p).any() or torch.isinf(p).any())]


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


def build_preprocess_fn(tokenizer, direction, max_len, indic_processor, is_indictrans2):
    src_lang_code, tgt_lang_code = direction_lang_codes(direction)

    def preprocess(batch):
        sources = batch["source"]
        targets = batch["target"]
        if indic_processor is not None:
            sources = indic_processor.preprocess_batch(sources, src_lang=src_lang_code, tgt_lang=tgt_lang_code)
            # is_target=True here is not optional decoration -- IndicProcessor's own
            # _preprocess() (confirmed by reading the real source, processor.pyx)
            # prepends "{src_lang} {tgt_lang} " to its output UNLESS is_target=True.
            # Without it (the bug this replaces), the label text handed to the
            # decoder would itself start with two language-tag tokens, i.e. the
            # model would be trained to predict a tag prefix as part of its own
            # output -- never how tag_indictrans2_text's fallback works, and not a
            # real seq2seq label contract either. Found while investigating a
            # loss=0/grad_norm=nan run that persisted even with a fresh LoRA init
            # and a confirmed-active warmup (2026-08-22) -- both of which ruled out
            # every other suspected cause, leaving this as the one code path that's
            # never actually run for real in this project before now.
            targets = indic_processor.preprocess_batch(targets, src_lang=tgt_lang_code, tgt_lang=src_lang_code, is_target=True)
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
            sources = tag_indictrans2_text(sources, src_lang_code, tgt_lang_code)
            targets = tag_indictrans2_text(targets, tgt_lang_code, src_lang_code)
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
    p.add_argument("--warmup_ratio", type=float, default=0.03,
                    help="Fraction of total steps spent ramping the learning rate up from 0 to "
                         "--lr, rather than applying full strength from step 1. Matters most when "
                         "--resume picks up an already fine-tuned adapter from the Hub (not a "
                         "fresh LoRA init) -- slamming a full, un-ramped LR onto weights that are "
                         "already near a good optimum is a real, hit-for-real way to blow up fp16 "
                         "gradients to inf/nan on the very first step. Applies even on a fresh "
                         "init, where it's just standard practice, not just the resume case.")
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
    p.add_argument("--disable_indic_toolkit", action="store_true",
                    help="Force the tag_indictrans2_text fallback even if IndicTransToolkit "
                         "imports successfully. Diagnostic escape hatch: IndicProcessor's real "
                         "preprocessing path has never run in a successful training in this "
                         "project before 2026-08-22, and that's also the first time loss=0/"
                         "grad_norm=nan showed up on a fresh LoRA init with warmup confirmed "
                         "active -- both other suspected causes (resume corruption, LR shock) "
                         "were ruled out with direct evidence. This flag isolates the toolkit as "
                         "the variable: if a run with this flag set trains with a sane, non-zero "
                         "loss, that confirms the toolkit integration (not something else "
                         "entirely) is the real cause, cheaply, without waiting on a full run.")
    p.add_argument("--no_gradient_checkpointing", action="store_true",
                    help="Disable gradient checkpointing. Diagnostic: every single IndicTrans2 "
                         "bn2en attempt so far (7 in a row, 2026-08-22, every combination of "
                         "resume/fresh-init, toolkit on/off, is_target fixed, fp16/fp32, full/"
                         "150k-row data) has logged grad_norm: nan on every logged step without "
                         "exception, and every one of those logs has also carried this exact "
                         "transformers warning, unchanged: 'You are using an old version of the "
                         "checkpointing format that is deprecated ... _set_gradient_checkpointing'. "
                         "That's AI4Bharat's custom modeling_indictrans.py using an old-style "
                         "checkpointing hook current transformers/accelerate explicitly flags as "
                         "deprecated -- the one variable that's been constant across every attempt "
                         "and never actually been turned off. Costs more GPU memory (no more "
                         "activation recomputation), which is a real OOM risk on a T4 for a 1B "
                         "model -- if it OOMs, that's still useful information, fast.")
    p.add_argument("--no_fp16", action="store_true",
                    help="Train in full fp32 instead of fp16 mixed precision. Diagnostic/fallback: "
                         "IndicTrans2 bn2en (the ai4bharat/indictrans2-indic-en-1B checkpoint "
                         "specifically) has produced loss=0/grad_norm=nan on every attempt so far "
                         "(2026-08-22) -- resumed, fresh-init, with IndicTransToolkit's real "
                         "preprocessing, and with it disabled entirely -- while IndicTrans2 en2bn "
                         "(the separate indictrans2-en-indic-1B checkpoint, same architecture, "
                         "different pretrained weights) has never failed under the exact same fp16 "
                         "setup. That pattern points at the indic-en-1B checkpoint's own weights "
                         "being numerically fragile under fp16's limited dynamic range, independent "
                         "of data or preprocessing. fp32 has far more headroom before overflow, at "
                         "the cost of slower training and more memory.")
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

    loaded_from_hub = False
    local_resume_ok = False
    if resume_path:
        # Check the local checkpoint for NaN/Inf BEFORE trusting it, exactly like
        # the Hub-fallback path below already does -- this local path used to load
        # unconditionally with zero corruption checking, which is a real gap: a
        # corrupted local checkpoint-N (e.g. one written by an earlier unstable
        # segment of the SAME run, before this script's warmup/is_target fixes
        # existed, or simply resumed on top of an already-NaN adapter) produces
        # `loss: 0, grad_norm: nan` at literally every logged step with no early
        # warning, silently burning the whole GPU-hour quota re-training on top of
        # dead weights. See find_nan_lora_params()'s docstring for the exact
        # 2026-08-23 run this was caught from.
        _resumed_model = PeftModel.from_pretrained(model, resume_path, is_trainable=True)
        _bad_params = find_nan_lora_params(_resumed_model)
        if _bad_params:
            print(f"local checkpoint {resume_path} contains NaN/Inf in {len(_bad_params)} of its "
                  f"parameter tensors (e.g. {_bad_params[0]}) — a previous training run silently "
                  f"corrupted this checkpoint. Refusing to resume from it; falling back to the Hub "
                  f"checkpoint (if any) or a fresh LoRA init instead.")
            resume_path = None  # so trainer.train() below doesn't also try to load it
            # Reload a clean base model: the rejected PeftModel.from_pretrained() call
            # above may have already attached LoRA modules onto this same underlying
            # object before the NaN check ran, and both the Hub-fallback path and
            # get_peft_model() below expect an unwrapped base model, not one PEFT has
            # already touched.
            model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, trust_remote_code=True, **quant_kwargs)
        else:
            model = _resumed_model
            print(f"resuming LoRA weights + optimizer/scheduler state from local checkpoint {resume_path}")
            local_resume_ok = True

    if not local_resume_ok:
        if args.resume and args.push_to_hub:
            # Don't just check the Hub repo *exists* — Trainer's push_to_hub=True
            # creates the repo the moment Trainer is constructed, well before any
            # real checkpoint is pushed (e.g. a previous run that crashed before
            # its first --save_steps interval leaves behind an empty repo that
            # "exists" but has no adapter_config.json). Actually attempt the load
            # and fall back on failure, instead of guessing in advance.
            try:
                _resumed_model = PeftModel.from_pretrained(model, args.push_to_hub, is_trainable=True)
                _bad_params = find_nan_lora_params(_resumed_model)
                if _bad_params:
                    raise ValueError(
                        f"resumed adapter from {args.push_to_hub} contains NaN/Inf in "
                        f"{len(_bad_params)} of its parameter tensors (e.g. {_bad_params[0]}) — "
                        f"a previous training run silently corrupted this checkpoint. Refusing "
                        f"to resume from it."
                    )
                model = _resumed_model
                print(f"no usable local checkpoint in {args.output_dir} — resumed LoRA weights from "
                      f"Hub repo {args.push_to_hub} instead (approximate resume: optimizer/LR-schedule "
                      f"state restarts, learned weights carry over)")
                loaded_from_hub = True
            except Exception as e:
                print(f"--resume given, but Hub repo {args.push_to_hub} has no usable adapter "
                      f"checkpoint yet (commonly because Trainer's push_to_hub=True creates the "
                      f"repo immediately at startup, before any checkpoint is pushed — e.g. a "
                      f"previous run that crashed before reaching --save_steps — or because the "
                      f"checkpoint that IS there is corrupted, e.g. all-NaN weights from an "
                      f"earlier unstable run) — starting fresh. ({type(e).__name__}: {e})")
                # Reload a clean base model rather than reusing `model`: the rejected
                # PeftModel.from_pretrained() attempt above may have already attached
                # LoRA modules onto this same underlying object before the NaN check
                # ran, and get_peft_model() below expects an unwrapped base model, not
                # one PEFT has already touched.
                model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, trust_remote_code=True, **quant_kwargs)
        if not loaded_from_hub:
            model = get_peft_model(model, build_lora_config(args))
    model.print_trainable_parameters()

    is_indictrans2 = "indictrans2" in args.model_name.lower()
    is_nllb = "nllb" in args.model_name.lower()
    src_lang_code, tgt_lang_code = direction_lang_codes(args.direction)

    indic_processor = (get_indic_processor(inference=False)
                       if is_indictrans2 and not args.disable_indic_toolkit else None)
    if is_indictrans2 and indic_processor is None:
        reason = _INDIC_TOOLKIT_IMPORT_ERROR or "unknown (import raised no captured exception, unexpected)"
        print("WARNING: IndicTransToolkit unavailable — falling back to plain tokenization for IndicTrans2. "
              f"Actual import failure: {reason}. "
              "Install with `pip install IndicTransToolkit` for AI4Bharat's recommended preprocessing "
              "(note: this package compiles a Cython extension on install, requiring a working C compiler "
              "in whatever environment it's installed in — see logs/phase_6_status.md for the full "
              "investigation of why this can fail).")

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
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=1.0,
        fp16=not args.no_fp16,
        gradient_checkpointing=not args.no_gradient_checkpointing,
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
