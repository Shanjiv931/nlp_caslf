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


def build_preprocess_fn(tokenizer, direction, max_len, indic_processor):
    src_lang_code = "ben_Beng" if direction == "en2bn" else "eng_Latn"
    tgt_lang_code = "eng_Latn" if direction == "en2bn" else "ben_Beng"

    def preprocess(batch):
        sources = batch["source"]
        targets = batch["target"]
        if indic_processor is not None:
            sources = indic_processor.preprocess_batch(sources, src_lang=src_lang_code, tgt_lang=tgt_lang_code)
            targets = indic_processor.preprocess_batch(targets, src_lang=tgt_lang_code, tgt_lang=src_lang_code)
        model_inputs = tokenizer(sources, max_length=max_len, truncation=True)
        labels = tokenizer(text_target=targets, max_length=max_len, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return preprocess


def main():
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

    indic_processor = IndicProcessor(inference=False) if (_HAS_INDIC_TOOLKIT and "indictrans2" in args.model_name.lower()) else None
    if "indictrans2" in args.model_name.lower() and indic_processor is None:
        print("WARNING: IndicTransToolkit not installed — falling back to plain tokenization for IndicTrans2. "
              "Install with `pip install IndicTransToolkit` for AI4Bharat's recommended preprocessing.")

    train_ds = load_direction_dataset(args.train_file, args.direction, args.max_train_rows)
    val_ds = load_direction_dataset(args.val_file, args.direction, max_rows=2000)

    preprocess_fn = build_preprocess_fn(tokenizer, args.direction, args.max_len, indic_processor)
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
