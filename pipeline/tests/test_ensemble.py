"""Unit tests for pipeline/ensemble.py's pure-logic functions
(preprocessing/postprocessing text transforms, deduplication) using mock
EngineContext objects — no torch/transformers required. Model loading and
`.generate()` itself cannot be exercised on this machine (see
logs/phase_6_status.md); those need real verification on Kaggle/Colab.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ensemble import Candidate, EngineContext, dedupe_candidates, postprocess_output, preprocess_source


def make_ctx(is_indictrans2=False, is_nllb=False, indic_processor=None, direction="bn2en"):
    src, tgt = ("ben_Beng", "eng_Latn") if direction == "bn2en" else ("eng_Latn", "ben_Beng")
    return EngineContext(
        engine_key="test", direction=direction, model=None, tokenizer=None,
        indic_processor=indic_processor, is_indictrans2=is_indictrans2, is_nllb=is_nllb,
        src_lang=src, tgt_lang=tgt,
    )


class FakeIndicProcessor:
    """Mimics IndicTransToolkit's IndicProcessor interface just enough to
    test the plumbing — real script normalization/NER masking isn't
    reproduced, just the batch-in/batch-out contract."""

    def preprocess_batch(self, texts, src_lang, tgt_lang):
        return [f"[PRE:{src_lang}->{tgt_lang}] {t}" for t in texts]

    def postprocess_batch(self, texts, lang):
        return [f"[POST:{lang}] {t}" for t in texts]


# ---- preprocess_source ----

def test_preprocess_source_sanitizes_before_anything_else():
    ctx = make_ctx()
    out = preprocess_source(ctx, "hello <unk> world")
    assert "<unk>" not in out


def test_preprocess_source_uses_indic_processor_when_available():
    ctx = make_ctx(is_indictrans2=True, indic_processor=FakeIndicProcessor(), direction="bn2en")
    out = preprocess_source(ctx, "ami bhalo achi")
    assert out == "[PRE:ben_Beng->eng_Latn] ami bhalo achi"


def test_preprocess_source_falls_back_to_manual_tagging_for_indictrans2():
    ctx = make_ctx(is_indictrans2=True, indic_processor=None, direction="bn2en")
    out = preprocess_source(ctx, "ami bhalo achi")
    assert out == "ben_Beng eng_Latn ami bhalo achi"
    assert len(out.split(" ", 2)) == 3  # must survive the real tokenizer's split(" ", 2)


def test_preprocess_source_plain_for_non_indictrans2():
    ctx = make_ctx(is_indictrans2=False, is_nllb=True, indic_processor=None)
    out = preprocess_source(ctx, "ami bhalo achi")
    assert out == "ami bhalo achi"  # no tagging needed — NLLB uses tokenizer.src_lang instead


# ---- postprocess_output ----

def test_postprocess_output_uses_indic_processor_when_available():
    ctx = make_ctx(is_indictrans2=True, indic_processor=FakeIndicProcessor())
    out = postprocess_output(ctx, "  I am fine  ")
    assert out == "[POST:eng_Latn] I am fine"  # stripped before being handed to postprocess_batch


def test_postprocess_output_plain_strip_without_indic_processor():
    ctx = make_ctx(is_indictrans2=False, indic_processor=None)
    assert postprocess_output(ctx, "  I am fine  ") == "I am fine"


# ---- dedupe_candidates ----

def test_dedupe_removes_exact_text_duplicates():
    candidates = [
        Candidate(text="I am fine", engine="nllb", method="beam", source_text="x", direction="bn2en"),
        Candidate(text="I am fine", engine="nllb", method="sample", source_text="x", direction="bn2en"),
        Candidate(text="I am well", engine="nllb", method="sample", source_text="x", direction="bn2en"),
    ]
    deduped = dedupe_candidates(candidates)
    assert len(deduped) == 2
    assert deduped[0].method == "beam"  # first occurrence kept


def test_dedupe_drops_empty_strings():
    candidates = [
        Candidate(text="", engine="nllb", method="beam", source_text="x", direction="bn2en"),
        Candidate(text="I am fine", engine="nllb", method="sample", source_text="x", direction="bn2en"),
    ]
    deduped = dedupe_candidates(candidates)
    assert len(deduped) == 1


def test_dedupe_empty_list():
    assert dedupe_candidates([]) == []
