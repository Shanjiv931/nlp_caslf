"""Phase 7 — ensemble.py: multi-candidate generation across all 3
LoRA-fine-tuned engines (IndicTrans2, NLLB-200-distilled-600M, BanglaT5).

Per the PRD's design philosophy (Section 0, principle 2 — "never trust a
single decode"): for a given source text + direction, generates 2-4
candidates from EACH engine using both beam search and nucleus sampling
(diversity of decoding strategy, not just diversity of model), yielding up
to ~10 candidates total. qe_rerank.py picks the best one; this module's
only job is producing a diverse, honestly-labeled candidate pool.

This machine cannot run torch/transformers (see logs/phase_6_status.md —
no CUDA GPU, and a Device Guard policy blocks even CPU torch). The model-
loading and `.generate()` code paths below are written correctly against
the real APIs and mirror train.py's already-verified-on-Kaggle preprocessing
exactly (via mt_compat.py, the shared module), but have NOT been executed
here — only the pure-logic pieces (preprocessing/postprocessing text
transforms, deduplication) are unit-tested locally, using mock objects that
don't require torch. Real end-to-end verification needs Kaggle/Colab or
another working-torch environment, same as Phase 6.
"""
import signal
from dataclasses import dataclass
from typing import List, Optional

import mt_compat as mc

# Real hang found and confirmed for real 2026-09-05: with the GPU-placement
# and per-step debug checkpoints already in place, a full_pipeline run's
# indictrans2 beam candidate 0 postprocessed instantly, but candidate 1's
# IndicProcessor.postprocess_batch() call (IndicTransToolkit, third-party)
# hung indefinitely on that specific piece of generated text -- ruling out
# "first call is slow" (candidate 0 already proved the IndicProcessor is
# warm and fast) and pointing at a genuine pathological-input bug in the
# library itself (most likely quadratic/catastrophic-backtracking regex
# behavior on degenerate output from an early or undertrained decode, e.g.
# repeated tokens). This is pure CPU computation, not network I/O, so
# mt_compat.py's socket.setdefaulttimeout(120) cannot catch it -- a
# SIGALRM-based wall-clock timeout is the only mechanism that can, and one
# specific enough to wrap only this call rather than the whole pipeline.
_POSTPROCESS_TIMEOUT_SECONDS = 30


class _PostprocessTimeout(Exception):
    pass


def _postprocess_with_timeout(ctx, decoded_text, label, timeout_seconds=_POSTPROCESS_TIMEOUT_SECONDS):
    """Runs postprocess_output() under a hard wall-clock timeout, falling
    back to the plain decoded/stripped text (never crashing or hanging the
    whole run over one degenerate candidate) if it doesn't return in time.
    SIGALRM only exists on POSIX -- the real execution environment for this
    pipeline is always a Linux Kaggle GPU session, never this Windows dev
    machine, so on Windows (where the unit tests run, against fast fake
    IndicProcessor objects) this just calls postprocess_output() directly
    with no timeout at all."""
    if not hasattr(signal, "SIGALRM"):
        return postprocess_output(ctx, decoded_text)

    def _handler(signum, frame):
        raise _PostprocessTimeout()

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout_seconds)
    try:
        return postprocess_output(ctx, decoded_text)
    except _PostprocessTimeout:
        print(f"WARNING: [generate:{ctx.engine_key}] postprocessing {label} exceeded "
              f"{timeout_seconds}s (likely a pathological-input bug in IndicProcessor / "
              f"IndicTransToolkit, not a network or GPU issue) -- falling back to the "
              f"plain decoded text for this candidate, un-postprocessed, and continuing.")
        return decoded_text.strip()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

ALL_ENGINES = ["indictrans2", "nllb", "banglat5"]


@dataclass
class EngineContext:
    engine_key: str
    direction: str
    model: object
    tokenizer: object
    indic_processor: object
    is_indictrans2: bool
    is_nllb: bool
    src_lang: str
    tgt_lang: str
    device: str = "cpu"


@dataclass
class Candidate:
    text: str
    engine: str
    method: str  # "beam" or "sample"
    source_text: str
    direction: str
    raw_score: Optional[float] = None  # length-normalized log-prob, beam-search candidates only
    # (`model.generate(..., output_scores=True, return_dict_in_generate=True)`
    # returns `sequences_scores` for free during beam search — captured here
    # as qe_rerank.py's fallback ranking signal if a QE model can't be
    # loaded. Not populated for nucleus-sampled candidates: a comparable
    # score for those needs a separate teacher-forced rescoring pass this
    # module doesn't do, so sampled candidates are ranked as "unscored"
    # in the fallback path rather than given a misleading number.


_MODEL_CACHE = {}


def load_engine(engine_key, direction, adapter_prefix=mc.DEFAULT_HF_ADAPTER_PREFIX, use_4bit=False):
    """Load base model + LoRA adapter for one engine+direction, applying the
    same compat shims and language setup train.py used to fine-tune it —
    inference-time preprocessing that drifts from training-time preprocessing
    is a classic, real source of quality loss in MT systems, which is
    exactly why this reuses mt_compat.py instead of reimplementing it here.
    Cached per (engine_key, direction, adapter_prefix): loading is expensive
    (full base model + adapter download) and produce_translation() may need
    the same engine repeatedly across many user requests.
    """
    cache_key = (engine_key, direction, adapter_prefix, use_4bit)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    mc.ensure_all_compat_shims()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model_name = mc.base_model_name(engine_key, direction)
    adapter_repo = mc.adapter_repo_id(engine_key, direction, adapter_prefix)

    # Bug found 2026-09-05 during the "find all possible errors" audit: this
    # function never placed the model (or, further down in
    # generate_candidates_for_engine, its inputs) on the GPU -- from_pretrained
    # with no device_map/`.to()` call leaves everything on CPU by default.
    # Every one of today's "hangs" (0% GPU, low/flat CPU%, no crash, no
    # output for minutes) is consistent with this: beam search + nucleus
    # sampling for a 1B-parameter model, run three times per example (one
    # per engine) on CPU alone, is genuinely slow -- not stuck. This was
    # never a network stall like the CometKiwi/Qwen/LaBSE downloads; it's
    # the actual translation engines silently never touching the GPU that
    # Kaggle session-hours are being spent on. use_4bit is never True for
    # these ensemble engines in this codebase (only llm_postedit.py's
    # separate 7B LLM uses it), so `.to(device)` below is always safe --
    # bitsandbytes 4-bit models can't use `.to()` and need `device_map`
    # instead, which is why this is skipped when use_4bit is set.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    quant_kwargs = {}
    if use_4bit:
        from transformers import BitsAndBytesConfig
        quant_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype="bfloat16", bnb_4bit_quant_type="nf4",
        )
        quant_kwargs["device_map"] = "auto"

    # Debug checkpoints added 2026-09-05: a full_pipeline run on Kaggle hung
    # with ~1% CPU / 0% GPU right after indictrans2's base model finished
    # loading, even with HF_HUB_OFFLINE=1 set (which only silences
    # huggingface_hub -- it does nothing for a stall in a *different*
    # library's own network/resource-loading code, e.g. IndicTransToolkit's
    # IndicProcessor). Prints below pin down exactly which of these four
    # steps is actually the one that never returns, instead of guessing
    # again. print(..., flush=True) since this runs inside a piped `!python`
    # subprocess where stdout can be block-buffered.
    print(f"[load_engine:{engine_key}] loading tokenizer ({model_name}) ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print(f"[load_engine:{engine_key}] tokenizer loaded; loading base model ...", flush=True)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True, **quant_kwargs)
    print(f"[load_engine:{engine_key}] base model loaded; applying adapter ({adapter_repo}) ...", flush=True)
    model = PeftModel.from_pretrained(base_model, adapter_repo)
    if not use_4bit:
        print(f"[load_engine:{engine_key}] moving model to {device} ...", flush=True)
        model = model.to(device)
    model.eval()
    print(f"[load_engine:{engine_key}] adapter applied (device={device}).", flush=True)

    is_indictrans2 = engine_key == "indictrans2"
    is_nllb = engine_key == "nllb"
    src_lang, tgt_lang = mc.direction_lang_codes(direction)

    if is_indictrans2:
        print(f"[load_engine:{engine_key}] constructing IndicProcessor ...", flush=True)
    indic_processor = mc.get_indic_processor(inference=True) if is_indictrans2 else None
    if is_indictrans2:
        print(f"[load_engine:{engine_key}] IndicProcessor ready.", flush=True)
    if is_nllb:
        if hasattr(tokenizer, "src_lang"):
            tokenizer.src_lang = src_lang
        if hasattr(tokenizer, "tgt_lang"):
            tokenizer.tgt_lang = tgt_lang

    ctx = EngineContext(
        engine_key=engine_key, direction=direction, model=model, tokenizer=tokenizer,
        indic_processor=indic_processor, is_indictrans2=is_indictrans2, is_nllb=is_nllb,
        src_lang=src_lang, tgt_lang=tgt_lang, device=device,
    )
    _MODEL_CACHE[cache_key] = ctx
    return ctx


def preprocess_source(ctx, text):
    """Pure-logic text transform (no model calls) — testable without torch."""
    text = mc.sanitize_text(text)
    if ctx.indic_processor is not None:
        return ctx.indic_processor.preprocess_batch([text], src_lang=ctx.src_lang, tgt_lang=ctx.tgt_lang)[0]
    if ctx.is_indictrans2:
        return mc.tag_indictrans2_text(text, ctx.src_lang, ctx.tgt_lang)
    return text


def postprocess_output(ctx, decoded_text):
    """Pure-logic text transform (no model calls) — testable without torch."""
    decoded_text = decoded_text.strip()
    if ctx.indic_processor is not None:
        return ctx.indic_processor.postprocess_batch([decoded_text], lang=ctx.tgt_lang)[0]
    return decoded_text


def dedupe_candidates(candidates: List[Candidate]) -> List[Candidate]:
    """Drop exact-text duplicates (common between beam search's top
    sequences and nucleus sampling, especially on short/simple inputs),
    keeping the first occurrence's method label. Also drops empty strings.
    Pure logic, no model calls — fully unit-testable."""
    seen = set()
    deduped = []
    for c in candidates:
        if c.text and c.text not in seen:
            seen.add(c.text)
            deduped.append(c)
    return deduped


def generate_candidates_for_engine(ctx, source_text, num_beam_candidates=2, num_sample_candidates=2, max_new_tokens=128):
    import torch

    # Debug checkpoints added 2026-09-05: after the GPU-placement fix, a
    # full_pipeline run went silent again right after "IndicProcessor
    # ready." with no change even after the model was confirmed on cuda
    # (GPU Memory climbed, confirming the earlier fix took effect) -- this
    # function's own steps (IndicProcessor preprocessing, tokenization,
    # device transfer, the two generate() calls) had zero prints, so there
    # was no way to tell which one, if any, was actually the new stall
    # point versus just legitimately-slow-but-progressing work. Same
    # rationale as load_engine's checkpoints: pin the exact location down
    # empirically instead of guessing again.
    print(f"[generate:{ctx.engine_key}] preprocessing source ...", flush=True)
    processed_source = preprocess_source(ctx, source_text)
    print(f"[generate:{ctx.engine_key}] tokenizing ...", flush=True)
    inputs = ctx.tokenizer(processed_source, return_tensors="pt", truncation=True, max_length=128)
    # Companion fix to load_engine's model placement above: a model on GPU
    # with CPU-resident inputs raises "Expected all tensors to be on the
    # same device" the moment .generate() is called, so this has to move in
    # lockstep with that change.
    inputs = {k: v.to(ctx.device) for k, v in inputs.items()}
    print(f"[generate:{ctx.engine_key}] inputs on {ctx.device}; starting beam search ...", flush=True)

    gen_kwargs_common = {}
    if ctx.is_nllb:
        # NLLB needs forced_bos_token_id set to the target-language token,
        # or generation defaults to whatever the base model's own default
        # target language is rather than the direction we actually want.
        try:
            gen_kwargs_common["forced_bos_token_id"] = ctx.tokenizer.convert_tokens_to_ids(ctx.tgt_lang)
        except Exception:
            pass

    raw_candidates = []
    with torch.no_grad():
        beam_out = ctx.model.generate(
            **inputs,
            num_beams=max(4, num_beam_candidates),
            num_return_sequences=num_beam_candidates,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
            **gen_kwargs_common,
        )
        print(f"[generate:{ctx.engine_key}] beam search done.", flush=True)
        beam_scores = beam_out.sequences_scores.tolist() if beam_out.sequences_scores is not None else [None] * len(beam_out.sequences)
        # Confirmed for real 2026-09-05: beam candidate 0 postprocessed
        # instantly, but candidate 1's postprocess_output() call hung
        # indefinitely -- a genuine pathological-input bug in
        # IndicTransToolkit's IndicProcessor.postprocess_batch() (CPU-only,
        # not caught by mt_compat.py's socket timeout or the GPU-placement
        # fix), not a "first call is slow" issue. _postprocess_with_timeout
        # bounds this call and degrades to the plain decoded text on
        # timeout rather than losing the whole run to one bad candidate.
        for i, (seq, score) in enumerate(zip(beam_out.sequences, beam_scores)):
            print(f"[generate:{ctx.engine_key}] decoding beam candidate {i} ...", flush=True)
            decoded = ctx.tokenizer.decode(seq, skip_special_tokens=True)
            print(f"[generate:{ctx.engine_key}] beam candidate {i} decoded; postprocessing ...", flush=True)
            text = _postprocess_with_timeout(ctx, decoded, label=f"beam candidate {i}")
            print(f"[generate:{ctx.engine_key}] beam candidate {i} postprocessed.", flush=True)
            raw_candidates.append(Candidate(text=text, engine=ctx.engine_key, method="beam",
                                             source_text=source_text, direction=ctx.direction,
                                             raw_score=score))

        if num_sample_candidates > 0:
            # Isolated in its own try/except, deliberately separate from the
            # beam-search block above: nucleus sampling is strictly a
            # diversity bonus on top of beam search, not a required source
            # of candidates, so a sampling-specific failure must never
            # discard beam candidates already collected. Hit for real on
            # Kaggle running eval/evaluate.py, 2026-08-17: IndicTrans2
            # (1B, LoRA-adapted) raised `RuntimeError: probability tensor
            # contains either inf, nan or element < 0` from `do_sample=True`
            # generation while beam search on the same input/model had
            # already succeeded moments earlier in this same function call —
            # before this fix, that exception propagated out of this
            # function entirely, and generate_ensemble_candidates' per-engine
            # try/except then discarded the already-good beam candidates
            # along with it, silently dropping IndicTrans2 from the ensemble
            # for that example instead of degrading to beam-only.
            try:
                print(f"[generate:{ctx.engine_key}] starting nucleus sampling ...", flush=True)
                sample_out = ctx.model.generate(
                    **inputs,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.9,
                    num_return_sequences=num_sample_candidates,
                    max_new_tokens=max_new_tokens,
                    **gen_kwargs_common,
                )
                print(f"[generate:{ctx.engine_key}] nucleus sampling done.", flush=True)
                for i, seq in enumerate(sample_out):
                    decoded = ctx.tokenizer.decode(seq, skip_special_tokens=True)
                    text = _postprocess_with_timeout(ctx, decoded, label=f"sample candidate {i}")
                    raw_candidates.append(Candidate(text=text, engine=ctx.engine_key, method="sample",
                                                     source_text=source_text, direction=ctx.direction))
            except Exception as e:
                print(f"WARNING: nucleus sampling failed for engine '{ctx.engine_key}' "
                      f"({type(e).__name__}: {e}) — continuing with beam-search candidates only")

    return dedupe_candidates(raw_candidates)


def generate_ensemble_candidates(source_text, direction, engines: Optional[List[str]] = None,
                                  adapter_prefix=mc.DEFAULT_HF_ADAPTER_PREFIX, use_4bit=False) -> List[Candidate]:
    """Top-level entry point: generate candidates across all requested
    engines (default: all 3), continuing past any single engine's failure
    rather than aborting the whole ensemble — one engine having a problem
    shouldn't deny the user a translation from the other two."""
    engines = engines or ALL_ENGINES
    all_candidates = []
    for engine_key in engines:
        try:
            ctx = load_engine(engine_key, direction, adapter_prefix=adapter_prefix, use_4bit=use_4bit)
            all_candidates.extend(generate_candidates_for_engine(ctx, source_text))
        except Exception as e:
            print(f"WARNING: engine '{engine_key}' failed to generate candidates "
                  f"({type(e).__name__}: {e}) — continuing with remaining engines")
    return all_candidates


def translate_single(source_text, direction, engine_key="nllb",
                      adapter_prefix=mc.DEFAULT_HF_ADAPTER_PREFIX, use_4bit=False) -> str:
    """Single-engine, beam-search-only translation — no ensemble diversity,
    no QE reranking, no LLM post-edit, no round-trip verification. Shared
    by `catla.py`'s "fast mode" (Phase 11's demo toggle) and
    `eval/evaluate.py`'s single_model_baseline mode, so both use the exact
    same "what does one fine-tuned engine alone produce" definition rather
    than two subtly different reimplementations of the same idea."""
    ctx = load_engine(engine_key, direction, adapter_prefix=adapter_prefix, use_4bit=use_4bit)
    candidates = generate_candidates_for_engine(ctx, source_text, num_beam_candidates=1, num_sample_candidates=0)
    return candidates[0].text if candidates else ""


if __name__ == "__main__":
    import sys
    text = sys.argv[1] if len(sys.argv) > 1 else "আমি ভালো আছি"
    direction = sys.argv[2] if len(sys.argv) > 2 else "bn2en"
    for c in generate_ensemble_candidates(text, direction):
        print(f"[{c.engine}/{c.method}] {c.text}")
