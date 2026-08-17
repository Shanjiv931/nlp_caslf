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


def translate_pretrained_baseline(source_text, direction, engine_key="nllb"):
    """The base model with NO LoRA adapter — never fine-tuned on this
    project's data at all. Loaded independently of ensemble.load_engine()
    (which always attaches a LoRA adapter) since this mode specifically
    needs the adapter-free model."""
    import mt_compat as mc

    mc.ensure_all_compat_shims()
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model_name = mc.base_model_name(engine_key, direction)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
    model.eval()

    is_indictrans2 = engine_key == "indictrans2"
    is_nllb = engine_key == "nllb"
    src_lang, tgt_lang = mc.direction_lang_codes(direction)
    indic_processor = mc.get_indic_processor(inference=True) if is_indictrans2 else None
    if is_nllb:
        if hasattr(tokenizer, "src_lang"):
            tokenizer.src_lang = src_lang
        if hasattr(tokenizer, "tgt_lang"):
            tokenizer.tgt_lang = tgt_lang

    from ensemble import EngineContext, generate_candidates_for_engine

    ctx = EngineContext(engine_key=engine_key, direction=direction, model=model, tokenizer=tokenizer,
                         indic_processor=indic_processor, is_indictrans2=is_indictrans2, is_nllb=is_nllb,
                         src_lang=src_lang, tgt_lang=tgt_lang)
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


def compute_comet(sources, hypotheses, references, checkpoint="Unbabel/wmt22-comet-da"):
    """Reference-based COMET (distinct from qe_rerank.py's reference-free
    CometKiwi) — needs the gold reference, appropriate for evaluation where
    references exist, unlike inference-time QE reranking where they don't."""
    from comet import download_model, load_from_checkpoint

    model = load_from_checkpoint(download_model(checkpoint))
    data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(sources, hypotheses, references)]
    output = model.predict(data, batch_size=8, gpus=0)
    return output.system_score if hasattr(output, "system_score") else sum(output["scores"]) / len(output["scores"])


TRANSLATE_FNS = {
    "single_model_baseline": translate_single_model_baseline,
    "full_pipeline": lambda src, direction: translate_full_pipeline(src, direction),
    "pretrained_baseline": translate_pretrained_baseline,
}


def run_mode(mode_name, pairs, direction, translate_fn):
    hypotheses = [translate_fn(p["source"], direction) for p in pairs]
    references = [p["reference"] for p in pairs]
    sources = [p["source"] for p in pairs]

    metrics = compute_metrics(hypotheses, references)
    try:
        metrics["comet"] = compute_comet(sources, hypotheses, references)
    except Exception as e:
        metrics["comet"] = None
        metrics["comet_error"] = f"{type(e).__name__}: {e}"

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
    args = p.parse_args()

    pairs = load_test_pairs(args.test_file, args.direction, args.max_examples)
    print(f"loaded {len(pairs)} test pairs for direction={args.direction}")

    results = {}
    for mode_name, translate_fn in TRANSLATE_FNS.items():
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

    single = results["single_model_baseline"]["metrics"]
    full = results["full_pipeline"]["metrics"]
    print("\n=== Quality-layer delta (full_pipeline - single_model_baseline) ===")
    for metric_name in ["bleu", "chrf++", "ter", "comet"]:
        s, f_ = single.get(metric_name), full.get(metric_name)
        if s is not None and f_ is not None:
            print(f"  {metric_name}: {f_ - s:+.2f}  (baseline={s:.2f}, full_pipeline={f_:.2f})")


if __name__ == "__main__":
    main()
