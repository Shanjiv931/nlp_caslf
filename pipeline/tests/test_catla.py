"""Unit tests for pipeline/catla.py — looks_fully_romanized_bengali (pure
logic) and translate_tweet() orchestration with mocked quality_pipeline/
ensemble calls. No torch required; real model calls need Kaggle/Colab, same
as every other model-calling function in this project.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import catla
from catla import looks_fully_romanized_bengali, translate_tweet


# ---- looks_fully_romanized_bengali ----

def test_fully_romanized_text_detected():
    assert looks_fully_romanized_bengali("ami bhalo achi") is True


def test_bengali_script_text_not_flagged():
    assert looks_fully_romanized_bengali("আমি ভালো আছি") is False


def test_code_mixed_text_not_flagged():
    # has real Bengali script AND English words -- must NOT be
    # transliterated wholesale, or the English content gets corrupted
    assert looks_fully_romanized_bengali("আমি feeling bhalo") is False


def test_pure_english_also_flagged_true_but_only_matters_for_bn2en_direction():
    # looks_fully_romanized_bengali can't distinguish "romanized Bengali"
    # from "genuinely just English" by script alone -- that's why
    # translate_tweet() only ever applies this check for the bn2en
    # direction, where the caller has already asserted the input is
    # Bengali. Documented here rather than silently assumed.
    assert looks_fully_romanized_bengali("hello world") is True


def test_empty_string_not_flagged():
    assert looks_fully_romanized_bengali("") is False


def test_numbers_and_punctuation_only_not_flagged():
    assert looks_fully_romanized_bengali("123 !!! ...") is False


# ---- translate_tweet() orchestration, mocked ----

def _patch_pipeline(monkeypatch, translation="I am doing well", processed_text_seen=None):
    def fake_produce_translation(processed, direction, engines=None, roundtrip_engine="nllb"):
        if processed_text_seen is not None:
            processed_text_seen.append(processed)
        return {
            "translation": translation,
            "winning_engine": "nllb",
            "qe_score": 0.9,
            "roundtrip_similarity": 0.95,
            "low_confidence": False,
        }

    monkeypatch.setattr(catla.quality_pipeline, "produce_translation", fake_produce_translation)


def test_translate_tweet_bn2en_bengali_script_input_not_transliterated(monkeypatch):
    seen = []
    _patch_pipeline(monkeypatch, processed_text_seen=seen)
    result = translate_tweet("আমি ভালো আছি", "bn2en")
    assert result["transliterated"] is False
    assert seen[0] == "আমি ভালো আছি"  # passed through unchanged (already proper Bengali script)


def test_translate_tweet_bn2en_romanized_input_gets_transliterated(monkeypatch):
    seen = []
    _patch_pipeline(monkeypatch, processed_text_seen=seen)
    result = translate_tweet("ami bhalo achi", "bn2en")
    assert result["transliterated"] is True
    assert seen[0] != "ami bhalo achi"  # was transliterated before hitting the pipeline
    assert any(0x0980 <= ord(c) <= 0x09FF for c in seen[0])  # now contains Bengali script


def test_translate_tweet_en2bn_never_transliterates(monkeypatch):
    seen = []
    _patch_pipeline(monkeypatch, processed_text_seen=seen)
    result = translate_tweet("hello there", "en2bn")
    assert result["transliterated"] is False
    assert seen[0] == "hello there"


def test_translate_tweet_applies_slang_normalization_before_pipeline(monkeypatch):
    seen = []
    _patch_pipeline(monkeypatch, processed_text_seen=seen)
    result = translate_tweet("u r great", "en2bn")
    assert result["normalized_text"] == "you are great"
    assert seen[0] == "you are great"


def test_translate_tweet_returns_none_translation_when_pipeline_fails(monkeypatch):
    def fake_produce_translation(processed, direction, engines=None, roundtrip_engine="nllb"):
        return {"translation": None, "low_confidence": True, "error": "no candidates"}

    monkeypatch.setattr(catla.quality_pipeline, "produce_translation", fake_produce_translation)
    result = translate_tweet("some text", "bn2en")
    assert result["translation"] is None
    assert result["low_confidence"] is True


def test_translate_tweet_result_includes_original_and_direction(monkeypatch):
    _patch_pipeline(monkeypatch)
    result = translate_tweet("ami bhalo achi", "bn2en")
    assert result["original_text"] == "ami bhalo achi"
    assert result["direction"] == "bn2en"


def test_translate_tweet_preserves_hashtag_via_reassembly(monkeypatch):
    _patch_pipeline(monkeypatch, translation="watched it #EndGame")
    # mock the word-level translator itself, not just the ensemble call it
    # wraps -- avoids a real (slow) network attempt inside load_engine
    # before its try/except catches the failure and falls back
    monkeypatch.setattr(catla, "make_word_translator", lambda direction, engine_key="nllb": (lambda w: w))
    result = translate_tweet("dekhlam eta #EndGame", "bn2en")
    assert "#EndGame" in result["translation"]


def test_translate_tweet_word_translator_falls_back_on_failure(monkeypatch):
    # this one deliberately exercises the REAL make_word_translator's
    # try/except path (ensemble.load_engine raising) to confirm it fails
    # safe (keeps the original word) rather than crashing hashtag rebuilding
    _patch_pipeline(monkeypatch, translation="watched it #EndGame")
    monkeypatch.setattr(catla.ensemble, "load_engine",
                         lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("simulated model load failure")))
    result = translate_tweet("dekhlam eta #EndGame", "bn2en")
    assert result["translation"] is not None  # didn't crash
    assert "#EndGame" in result["translation"]  # fell back to the original (untranslated) word


# ---- fast_mode ----

def test_fast_mode_calls_translate_single_not_quality_pipeline(monkeypatch):
    calls = []

    def fake_translate_single(processed, direction, engine_key="nllb"):
        calls.append((processed, direction, engine_key))
        return "I am doing well"

    def fail_if_called(*a, **kw):
        raise AssertionError("quality_pipeline.produce_translation must NOT be called in fast_mode")

    monkeypatch.setattr(catla.ensemble, "translate_single", fake_translate_single)
    monkeypatch.setattr(catla.quality_pipeline, "produce_translation", fail_if_called)
    monkeypatch.setattr(catla, "make_word_translator", lambda direction, engine_key="nllb": (lambda w: w))

    result = translate_tweet("আমি ভালো আছি", "bn2en", fast_mode=True)
    assert result["translation"] == "I am doing well"
    assert len(calls) == 1
    assert calls[0][2] == "nllb"  # default roundtrip_engine used as the fast-mode engine


def test_fast_mode_respects_custom_engine_choice(monkeypatch):
    calls = []
    monkeypatch.setattr(catla.ensemble, "translate_single",
                         lambda processed, direction, engine_key="nllb": (calls.append(engine_key), "ok")[1])
    monkeypatch.setattr(catla, "make_word_translator", lambda direction, engine_key="nllb": (lambda w: w))

    translate_tweet("hello", "en2bn", roundtrip_engine="banglat5", fast_mode=True)
    assert calls == ["banglat5"]


def test_fast_mode_skips_qe_and_roundtrip_fields(monkeypatch):
    monkeypatch.setattr(catla.ensemble, "translate_single", lambda processed, direction, engine_key="nllb": "ok")
    monkeypatch.setattr(catla, "make_word_translator", lambda direction, engine_key="nllb": (lambda w: w))

    result = translate_tweet("hello", "en2bn", fast_mode=True)
    assert result["qe_score"] is None
    assert result["roundtrip_similarity"] is None
    assert result["low_confidence"] is False
    assert result["winning_method"] == "beam_fast_mode"


def test_quality_mode_is_still_the_default(monkeypatch):
    # fast_mode defaults to False -- the full quality pipeline must still
    # run when the caller doesn't opt into fast mode explicitly
    seen = []
    _patch_pipeline(monkeypatch, processed_text_seen=seen)

    def fail_if_called(*a, **kw):
        raise AssertionError("ensemble.translate_single must NOT be called outside fast_mode")

    monkeypatch.setattr(catla.ensemble, "translate_single", fail_if_called)
    result = translate_tweet("আমি ভালো আছি", "bn2en")
    assert len(seen) == 1  # quality_pipeline.produce_translation was called
