"""Phase 8 — evaluate.py: runs inference on data/processed/test.jsonl three
ways and computes BLEU/chrF++/TER (sacrebleu) + COMET + Platform-Fidelity
Score for each, explicitly reporting the delta between the full quality
pipeline and a single fine-tuned model alone — proving the Phase 7
architecture is worth its extra complexity is the entire point of this
script, per the PRD.

Three evaluation modes, run over the same held-out test pairs:
  (a) single_model_baseline — one fine-tuned engine, beam search only, no
      ensemble/QE/LLM-postedit/round-trip-verify. Uses NLLB by default
      (fastest of the 3, and the PRD's second choice after IndicTrans2 —
      chosen as the baseline specifically because it's a reasonable single
      model on its own, not a weak strawman).
  (b) full_pipeline — quality_pipeline.produce_translation(), the complete
      Phase 7 chain.
  (c) pretrained_baseline — the same base model with NO LoRA adapter at
      all, i.e. what AI4Bharat/Meta/csebuetnlp shipped, never fine-tuned
      on this project's Twitter-domain data. Establishes how much the
      fine-tuning itself contributed, separately from how much the
      quality layer contributes on top of it.

This machine cannot run torch/transformers (see logs/phase_6_status.md and
phase_7_status.md), so this script is unverified end-to-end here. What IS
verified locally: `compute_metrics()` actually runs (sacrebleu is pure
Python, no torch needed) against hand-crafted examples — see
eval/tests/test_evaluate.py. `compute_comet()` and the three `translate_*`
functions need real models and are correct-by-construction, unverified.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline"))

import sacrebleu
from tqdm import tqdm

from platform_fidelity import score_dataset

PROCESSED = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
EVAL_DIR = os.path.dirname(__file__)


def load_test_pairs(path, direction, max_rows=None):
    """Same bn_en_translation-only, sanitized-text loading convention as
    train.py's load_direction_dataset, kept independent (not imported) since
    eval intentionally never depends on training-time-only concerns (LoRA
    config, checkpoint resume) — but the row-selection logic must agree, or
    the test set evaluated here silently diverges from what train.py itself
    would consider "bn_en_translation" test data."""
    from mt_compat import sanitize_text

    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("pair_type") != "bn_en_translation":
                continue
            if not row.get("text") or not row.get("parallel_text"):
                continue
            if direction == "bn2en":
                src, tgt = row["text"], row["parallel_text"]
            else:
                src, tgt = row["parallel_text"], row["text"]
            src, tgt = sanitize_text(src), sanitize_text(tgt)
            if not src or not tgt:
                continue
            rows.append({"source": src, "reference": tgt})
            if max_rows and len(rows) >= max_rows:
                break
    return rows


def translate_single_model_baseline(source_text, direction, engine_key="nllb"):
    # shared with catla.py's "fast mode" (Phase 11's demo toggle) via
    # ensemble.translate_single(), so both use the exact same "what does
    # one fine-tuned engine alone produce" definition
    from ensemble import translate_single

    return translate_single(source_text, direction, engine_key=engine_key)


def translate_full_pipeline(source_text, direction):
    from quality_pipeline import produce_translation

    result = produce_translation(source_text, direction)
    return result["translation"] or ""


_PRETRAINED_CTX_CACHE = {}


def _load_pretrained_ctx(engine_key, direction):
    """Cached per (engine_key, direction) -- without this, run_mode()'s
    per-example call to translate_pretrained_baseline() re-triggers
    AutoModelForSeq2SeqLM.from_pretrained() from scratch for EVERY test
    example (200x for a 200-example run). For nllb/banglat5 (small models)
    that was merely wasteful; for indictrans2 (4GB, 1B params) it meant
    reloading the entire model 200 times in a row and never finishing in
    any reasonable time -- a real bug, confirmed 2026-09-05 by a Kaggle
    run stuck cycling `Loading weights: 100%|763/763` endlessly on this
    exact mode/engine. Mirrors ensemble.py's load_engine() cache."""
    cache_key = (engine_key, direction)
    if cache_key in _PRETRAINED_CTX_CACHE:
        return _PRETRAINED_CTX_CACHE[cache_key]

    import mt_compat as mc

    mc.ensure_all_compat_shims()
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model_name = mc.base_model_name(engine_key, direction)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
    model.eval()
    # Deliberately NOT moved to .cuda() here: generate_candidates_for_engine()
    # never moves its tokenizer output off CPU, so putting the model on GPU
    # while inputs stay on CPU would crash with a device mismatch. This
    # matches ensemble.py's load_engine(), which has the same CPU-only
    # behavior for the same reason -- generation-side GPU placement is a
    # real, separate optimization opportunity, not fixed here to avoid
    # shipping an untested device-mismatch bug alongside the caching fix.

    is_indictrans2 = engine_key == "indictrans2"
    is_nllb = engine_key == "nllb"
    src_lang, tgt_lang = mc.direction_lang_codes(direction)
    indic_processor = mc.get_indic_processor(inference=True) if is_indictrans2 else None
    if is_nllb:
        if hasattr(tokenizer, "src_lang"):
            tokenizer.src_lang = src_lang
        if hasattr(tokenizer, "tgt_lang"):
            tokenizer.tgt_lang = tgt_lang

    from ensemble import EngineContext

    ctx = EngineContext(engine_key=engine_key, direction=direction, model=model, tokenizer=tokenizer,
                         indic_processor=indic_processor, is_indictrans2=is_indictrans2, is_nllb=is_nllb,
                         src_lang=src_lang, tgt_lang=tgt_lang)
    _PRETRAINED_CTX_CACHE[cache_key] = ctx
    return ctx


def translate_pretrained_baseline(source_text, direction, engine_key="nllb"):
    """The base model with NO LoRA adapter — never fine-tuned on this
    project's data at all. Loaded independently of ensemble.load_engine()
    (which always attaches a LoRA adapter) since this mode specifically
    needs the adapter-free model. Model/tokenizer load itself is cached
    (see _load_pretrained_ctx) -- only generation happens per call."""
    from ensemble import generate_candidates_for_engine

    ctx = _load_pretrained_ctx(engine_key, direction)
    candidates = generate_candidates_for_engine(ctx, source_text, num_beam_candidates=1, num_sample_candidates=0)
    return candidates[0].text if candidates else ""


def compute_metrics(hypotheses, references):
    """references: list of single reference strings (one per hypothesis) —
    sacrebleu's corpus_* functions want a list-of-reference-lists (multiple
    references per system supported), so this wraps into that shape.
    Pure Python, no torch — genuinely runnable and tested locally."""
    ref_lists = [references]
    bleu = sacrebleu.corpus_bleu(hypotheses, ref_lists)
    chrf_pp = sacrebleu.corpus_chrf(hypotheses, ref_lists, word_order=2)  # word_order=2 is chrF++
    ter = sacrebleu.corpus_ter(hypotheses, ref_lists)
    return {"bleu": bleu.score, "chrf++": chrf_pp.score, "ter": ter.score}


def _comet_gpu_count():
    """1 if a CUDA GPU is available, else 0. `model.predict(..., gpus=N)`
    was hardcoded to `gpus=0` here (and in qe_rerank.py) unconditionally --
    real bug, not a deliberate CPU-only choice: it silently ran COMET on
    CPU even on a GPU-equipped Kaggle session, and was a real driver of an
    8+ hour eval run (2026-09-05) that should have taken a fraction of
    that. `qe_rerank.py`'s CometKiwi call in particular runs once per test
    example inside full_pipeline mode, so this one flag has an outsized
    effect on total runtime."""
    try:
        import torch
        return 1 if torch.cuda.is_available() else 0
    except ImportError:
        return 0


def compute_comet(sources, hypotheses, references, checkpoint="Unbabel/wmt22-comet-da"):
    """Reference-based COMET (distinct from qe_rerank.py's reference-free
    CometKiwi) — needs the gold reference, appropriate for evaluation where
    references exist, unlike inference-time QE reranking where they don't."""
    from comet import download_model, load_from_checkpoint

    model = load_from_checkpoint(download_model(checkpoint))
    data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(sources, hypotheses, references)]
    output = model.predict(data, batch_size=8, gpus=_comet_gpu_count())
    return output.system_score if hasattr(output, "system_score") else sum(output["scores"]) / len(output["scores"])


TRANSLATE_FNS = {
    "single_model_baseline": translate_single_model_baseline,
    "full_pipeline": lambda src, direction: translate_full_pipeline(src, direction),
    "pretrained_baseline": translate_pretrained_baseline,
}


def run_mode(mode_name, pairs, direction, translate_fn):
    # No progress output here used to mean total silence for the entire
    # per-example loop (200 examples, no print/tqdm at all) -- looked
    # indistinguishable from a hang, especially for pretrained_baseline/
    # single_model_baseline's CPU-only generation. Confirmed for real
    # 2026-09-06: a run sitting quietly for 5+ minutes after model download
    # finished was actually just working through this exact loop.
    #
    # try/except added 2026-09-05: this list comprehension used to call
    # translate_fn() completely unguarded -- a single example raising for
    # ANY reason (a transient network blip, an edge-case input, anything
    # inside full_pipeline's ensemble+QE+LLM-postedit+round-trip chain)
    # crashed the entire multi-hour run and wrote nothing to disk, discarding
    # every one of the other ~199 already-computed examples along with it.
    # Paired with mt_compat.py's new socket.setdefaulttimeout(120): a stalled
    # network call now actually raises (instead of hanging forever) and that
    # raise is now caught here instead of taking down the whole run. A failed
    # example gets an empty-string hypothesis (scores ~0 against its
    # reference in corpus-level BLEU/chrF++/TER -- an honest, visible
    # penalty, not silently dropped or excluded) and a printed warning naming
    # which example and why, never a swallowed failure.
    hypotheses = []
    n_failed = 0
    for p in tqdm(pairs, desc=f"{mode_name} ({direction})"):
        try:
            hypotheses.append(translate_fn(p["source"], direction))
        except Exception as e:
            n_failed += 1
            hypotheses.append("")
            print(f"WARNING: {mode_name} ({direction}) failed on example "
                  f"{len(hypotheses)}/{len(pairs)} ({type(e).__name__}: {e}) "
                  f"-- scored as an empty hypothesis, continuing")
    if n_failed:
        print(f"{mode_name} ({direction}): {n_failed}/{len(pairs)} examples failed "
              f"and were scored as empty hypotheses -- see WARNING lines above")
    references = [p["reference"] for p in pairs]
    sources = [p["source"] for p in pairs]

    metrics = compute_metrics(hypotheses, references)
    metrics["n_failed"] = n_failed
    try:
        metrics["comet"] = compute_comet(sources, hypotheses, references)
    except Exception as e:
        # Full traceback, not just str(e) -- confirmed for real 2026-09-06:
        # a "ValueError: not enough values to unpack (expected 3, got 2)"
        # persisted even after pinning CUDA_VISIBLE_DEVICES=0 (ruling out
        # the dual-GPU theory), meaning it's a real version-compatibility
        # crack between unbabel-comet==2.2.7 (pinned to torchmetrics<0.11,
        # written for pytorch-lightning ~1.6-era APIs) and the much newer
        # stack actually installed (pytorch-lightning auto-upgrades
        # checkpoints to v2.6.5, torchmetrics 1.9.0, transformers 5.0.0).
        # str(e) alone doesn't say which library's internals the unpack
        # happens in -- the full traceback does, and is needed to fix this
        # properly instead of guessing.
        import traceback
        metrics["comet"] = None
        metrics["comet_error"] = f"{type(e).__name__}: {e}"
        metrics["comet_traceback"] = traceback.format_exc()
        print(f"  [comet failed] {metrics['comet_error']}\n{metrics['comet_traceback']}")

    pfs_input = [{"source_text": s, "translated_text": h} for s, h in zip(sources, hypotheses)]
    pfs = score_dataset(pfs_input)

    return {"mode": mode_name, "metrics": metrics, "pfs": pfs, "hypotheses": hypotheses,
            "sources": sources, "references": references}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test_file", default=os.path.join(PROCESSED, "test.jsonl"))
    p.add_argument("--direction", required=True, choices=["bn2en", "en2bn"])
    p.add_argument("--max_examples", type=int, default=200,
                    help="Full pipeline mode calls 3 fine-tuned models + a 7B LLM per example "
                         "(see quality_pipeline.py) -- default caps this for free-tier feasibility, "
                         "override for a full test-set run once time/compute allows.")
    p.add_argument("--baseline_engine", default="nllb")
    p.add_argument("--output_json", default=None)
    p.add_argument("--modes", default=",".join(TRANSLATE_FNS.keys()),
                    help="Comma-separated subset of modes to run: "
                         f"{list(TRANSLATE_FNS.keys())}. `full_pipeline` always evaluates "
                         "all 3 ensemble engines regardless of --baseline_engine, so its result "
                         "is identical across a per-engine loop -- run it once per direction, "
                         "not once per engine, to avoid tripling the cost of the most expensive "
                         "mode (3 MT engines + CometKiwi reranking + a 7B LLM post-edit + "
                         "round-trip verification, per example) for no new information.")
    args = p.parse_args()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    unknown = [m for m in modes if m not in TRANSLATE_FNS]
    if unknown:
        p.error(f"unknown mode(s) {unknown}, choose from {list(TRANSLATE_FNS.keys())}")

    pairs = load_test_pairs(args.test_file, args.direction, args.max_examples)
    print(f"loaded {len(pairs)} test pairs for direction={args.direction}")

    results = {}
    for mode_name in modes:
        translate_fn = TRANSLATE_FNS[mode_name]
        print(f"running mode: {mode_name} ...")
        if mode_name in ("single_model_baseline", "pretrained_baseline"):
            fn = lambda src, direction, _f=translate_fn: _f(src, direction, engine_key=args.baseline_engine)
        else:
            fn = translate_fn
        results[mode_name] = run_mode(mode_name, pairs, args.direction, fn)
        print(f"  {mode_name}: {results[mode_name]['metrics']}")

    output_json = args.output_json or os.path.join(EVAL_DIR, f"results_{args.direction}.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({k: {"metrics": v["metrics"], "pfs_summary": {kk: vv for kk, vv in v["pfs"].items() if kk != "per_example"}}
                   for k, v in results.items()}, f, ensure_ascii=False, indent=2)
    print(f"wrote {output_json}")

    if "single_model_baseline" in results and "full_pipeline" in results:
        single = results["single_model_baseline"]["metrics"]
        full = results["full_pipeline"]["metrics"]
        print("\n=== Quality-layer delta (full_pipeline - single_model_baseline) ===")
        for metric_name in ["bleu", "chrf++", "ter", "comet"]:
            s, f_ = single.get(metric_name), full.get(metric_name)
            if s is not None and f_ is not None:
                print(f"  {metric_name}: {f_ - s:+.2f}  (baseline={s:.2f}, full_pipeline={f_:.2f})")
    else:
        print(f"\n(skipping quality-layer delta print -- this run only covered modes: {modes})")


if __name__ == "__main__":
    main()
