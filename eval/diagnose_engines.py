"""Phase 8 diagnostics — diagnose_engines.py: cheap, small-scale tracing to
find WHY indictrans2 and banglat5 scored near-zero BLEU in a completed
full-dataset eval run (results_bn2en_indictrans2.json: 0.0 both pretrained
and fine-tuned; results_bn2en_banglat5.json: 1.18 BLEU, bit-for-bit
IDENTICAL between pretrained and fine-tuned) despite both engines' training
runs completing without error (logs/phase_6_status.md). Rather than
guessing and spending another full 200-example GPU run to find out, this
traces a handful of examples end-to-end and prints every intermediate
stage, plus an adapter-on vs adapter-off comparison, so a bug like "the
adapter never actually changes generation" or "postprocessing corrupts
otherwise-correct output" is visible directly instead of inferred from one
aggregate BLEU number. Deliberately tiny (--max_examples defaults to 5) —
this is meant to cost minutes of GPU time, not a full run.

Usage (Kaggle, after `git pull`, from caslf/, same HF_HUB_OFFLINE=1
convention as eval/evaluate.py since every model here should already be
cached from prior runs):
    HF_HUB_OFFLINE=1 python eval/diagnose_engines.py --direction bn2en \
        --engines indictrans2,banglat5 --max_examples 5
"""
import argparse
import contextlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline"))

from evaluate import PROCESSED, load_test_pairs  # reuses eval's exact test-set convention


def _adapter_diagnostics(model, engine_key):
    """Prints whatever structural evidence is available that a LoRA adapter
    is actually loaded with nonzero trainable weights -- confirms the
    adapter attached at all, independent of whether it's *good*."""
    print(f"[{engine_key}] model class: {type(model).__name__}")
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    if hasattr(model, "peft_config"):
        for name, cfg in model.peft_config.items():
            print(f"[{engine_key}] peft_config[{name!r}]: r={getattr(cfg, 'r', '?')} "
                  f"target_modules={getattr(cfg, 'target_modules', '?')}")
    if hasattr(model, "active_adapters"):
        try:
            print(f"[{engine_key}] active_adapters: {model.active_adapters()}")
        except Exception as e:
            print(f"[{engine_key}] active_adapters(): {type(e).__name__}: {e}")


def trace_example(ctx, source_text, reference, index):
    import torch

    from ensemble import postprocess_output, preprocess_source

    processed = preprocess_source(ctx, source_text)
    inputs = ctx.tokenizer(processed, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(ctx.device) for k, v in inputs.items()}

    def _generate(disable_adapter):
        cm = (
            ctx.model.disable_adapter()
            if disable_adapter and hasattr(ctx.model, "disable_adapter")
            else contextlib.nullcontext()
        )
        with torch.no_grad(), cm:
            out = ctx.model.generate(**inputs, num_beams=4, max_new_tokens=128, do_sample=False)
        return ctx.tokenizer.decode(out[0], skip_special_tokens=True)

    with_adapter = _generate(disable_adapter=False)
    without_adapter = _generate(disable_adapter=False) if not hasattr(ctx.model, "disable_adapter") else _generate(disable_adapter=True)
    postprocessed = postprocess_output(ctx, with_adapter)

    print(f"--- {ctx.engine_key} example {index} ---")
    print(f"  SOURCE:                   {source_text}")
    print(f"  REFERENCE:                {reference}")
    print(f"  PREPROCESSED INPUT:       {processed!r}")
    print(f"  RAW DECODE (adapter ON):  {with_adapter!r}")
    print(f"  RAW DECODE (adapter OFF): {without_adapter!r}")
    print(f"  POSTPROCESSED OUTPUT:     {postprocessed!r}")
    print(f"  adapter changed output:  {with_adapter != without_adapter}")
    print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test_file", default=os.path.join(PROCESSED, "test.jsonl"))
    p.add_argument("--direction", required=True, choices=["bn2en", "en2bn"])
    p.add_argument("--engines", default="indictrans2,banglat5",
                    help="comma-separated engine keys to trace (default: the two suspected-broken ones)")
    p.add_argument("--max_examples", type=int, default=5)
    args = p.parse_args()

    pairs = load_test_pairs(args.test_file, args.direction, args.max_examples)
    print(f"loaded {len(pairs)} test pairs for direction={args.direction}")

    from ensemble import load_engine

    for engine_key in args.engines.split(","):
        engine_key = engine_key.strip()
        print(f"\n========== engine: {engine_key} ==========")
        ctx = load_engine(engine_key, args.direction)
        _adapter_diagnostics(ctx.model, engine_key)
        for i, pair in enumerate(pairs):
            trace_example(ctx, pair["source"], pair["reference"], i)


if __name__ == "__main__":
    main()
