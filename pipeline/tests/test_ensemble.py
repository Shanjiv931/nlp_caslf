"""Unit tests for pipeline/ensemble.py's pure-logic functions
(preprocessing/postprocessing text transforms, deduplication) using mock
EngineContext objects — no torch/transformers required. Model loading and
`.generate()` itself cannot be exercised on this machine (see
logs/phase_6_status.md); those need real verification on Kaggle/Colab.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ensemble import Candidate, EngineContext, dedupe_candidates, generate_candidates_for_engine, postprocess_output, preprocess_source


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


# ---- generate_candidates_for_engine ----
# `import torch` happens *inside* this function specifically so it stays
# importable on this machine at module load time; these tests inject a
# minimal fake `torch` module (just enough of the `no_grad()` context
# manager contract) into sys.modules before calling it, and a fake
# model/tokenizer -- letting the REAL function body run, exercising its
# actual control flow (not a reimplementation of it), without needing a
# real torch install. Same technique used in phase_6_status.md to verify
# the transformers.onnx shim's import mechanics without real transformers.

class _FakeNoGrad:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _install_fake_torch(monkeypatch):
    import types
    fake_torch = types.ModuleType("torch")
    fake_torch.no_grad = lambda: _FakeNoGrad()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)


def _fake_tokenizer():
    class FakeTokenizer:
        def __call__(self, text, return_tensors="pt", truncation=True, max_length=128):
            return {}

        def decode(self, seq, skip_special_tokens=True):
            return seq

        def convert_tokens_to_ids(self, tok):
            return 0

    return FakeTokenizer()


class _BeamOut:
    def __init__(self, sequences):
        self.sequences = sequences
        self.sequences_scores = None


def _fake_model(sample_raises=False, sample_error=None):
    class FakeModel:
        def generate(self, **kwargs):
            if kwargs.get("do_sample"):
                if sample_raises:
                    raise (sample_error or RuntimeError(
                        "probability tensor contains either `inf`, `nan` or element < 0"))
                return ["sampled A", "sampled B"]
            return _BeamOut(["beam A", "beam B"])

    return FakeModel()


def test_generate_candidates_sampling_failure_keeps_beam_candidates(monkeypatch):
    # Real bug hit running eval/evaluate.py on Kaggle, 2026-08-17:
    # IndicTrans2's nucleus-sampling generate() call raised
    # `RuntimeError: probability tensor contains either inf, nan or
    # element < 0` *after* beam search on the same input had already
    # succeeded moments earlier in the same function call. Before the fix,
    # that exception propagated out of the whole function, and the caller's
    # per-engine try/except (generate_ensemble_candidates) then discarded
    # the already-good beam candidates along with it -- dropping the
    # engine from the ensemble entirely instead of degrading to beam-only.
    _install_fake_torch(monkeypatch)
    ctx = make_ctx()
    ctx.model = _fake_model(sample_raises=True)
    ctx.tokenizer = _fake_tokenizer()

    candidates = generate_candidates_for_engine(ctx, "source text",
                                                  num_beam_candidates=2, num_sample_candidates=2)

    assert len(candidates) == 2
    assert all(c.method == "beam" for c in candidates)


def test_generate_candidates_happy_path_includes_both_methods(monkeypatch):
    _install_fake_torch(monkeypatch)
    ctx = make_ctx()
    ctx.model = _fake_model(sample_raises=False)
    ctx.tokenizer = _fake_tokenizer()

    candidates = generate_candidates_for_engine(ctx, "source text",
                                                  num_beam_candidates=2, num_sample_candidates=2)

    assert {c.method for c in candidates} == {"beam", "sample"}
    assert len(candidates) == 4


def test_generate_candidates_skips_sampling_call_entirely_when_zero_requested(monkeypatch):
    _install_fake_torch(monkeypatch)
    calls = []

    class TrackingModel:
        def generate(self, **kwargs):
            calls.append(kwargs.get("do_sample"))
            if kwargs.get("do_sample"):
                return []
            return _BeamOut(["beam A"])

    ctx = make_ctx()
    ctx.model = TrackingModel()
    ctx.tokenizer = _fake_tokenizer()

    candidates = generate_candidates_for_engine(ctx, "source text",
                                                  num_beam_candidates=1, num_sample_candidates=0)

    assert calls == [False]  # sampling generate() call never made at all
    assert len(candidates) == 1
