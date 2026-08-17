"""Unit tests for pipeline/mt_compat.py's pure-logic functions (the shims
and model-loading helpers need real transformers/torch to exercise
meaningfully and are covered by the standalone mock tests referenced in
logs/phase_6_status.md instead — these test what's testable without them).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mt_compat import (
    adapter_repo_id,
    base_model_name,
    default_target_modules,
    direction_lang_codes,
    sanitize_text,
    tag_indictrans2_text,
)


def test_direction_lang_codes_bn2en():
    assert direction_lang_codes("bn2en") == ("ben_Beng", "eng_Latn")


def test_direction_lang_codes_en2bn():
    assert direction_lang_codes("en2bn") == ("eng_Latn", "ben_Beng")


def test_base_model_name_indictrans2_is_direction_specific():
    bn2en = base_model_name("indictrans2", "bn2en")
    en2bn = base_model_name("indictrans2", "en2bn")
    assert bn2en != en2bn
    assert "indic-en" in bn2en
    assert "en-indic" in en2bn


def test_base_model_name_nllb_same_both_directions():
    assert base_model_name("nllb", "bn2en") == base_model_name("nllb", "en2bn")


def test_base_model_name_banglat5_same_both_directions():
    assert base_model_name("banglat5", "bn2en") == base_model_name("banglat5", "en2bn")


def test_adapter_repo_id_format():
    assert adapter_repo_id("nllb", "bn2en") == "shanjivkr/catla-nllb-bn2en"
    assert adapter_repo_id("indictrans2", "en2bn", prefix="someone/other") == "someone/other-indictrans2-en2bn"


def test_default_target_modules_t5_family():
    assert default_target_modules("csebuetnlp/banglat5") == ["q", "k", "v", "o"]


def test_default_target_modules_m2m100_family():
    modules = default_target_modules("facebook/nllb-200-distilled-600M")
    assert "q_proj" in modules and "q" not in modules


def test_sanitize_text_strips_special_token_literals():
    assert sanitize_text("Notok kom koro priyo <unk>") == "Notok kom koro priyo"


def test_sanitize_text_collapses_whitespace():
    assert sanitize_text("hello   <pad>   world") == "hello world"


def test_sanitize_text_noop_on_clean_text():
    assert sanitize_text("ami bhalo achi") == "ami bhalo achi"


def test_tag_indictrans2_text_single_string():
    tagged = tag_indictrans2_text("ami bhalo achi", "ben_Beng", "eng_Latn")
    assert tagged == "ben_Beng eng_Latn ami bhalo achi"
    # must round-trip through the exact split the real tokenizer does
    assert tagged.split(" ", 2) == ["ben_Beng", "eng_Latn", "ami bhalo achi"]


def test_tag_indictrans2_text_list():
    tagged = tag_indictrans2_text(["a", "b"], "eng_Latn", "ben_Beng")
    assert tagged == ["eng_Latn ben_Beng a", "eng_Latn ben_Beng b"]


def test_tag_prefix_survives_sanitization_pipeline():
    # the exact scenario that broke IndicTrans2 training on 2026-08-15:
    # sanitize first, then tag -- must never leave a fragment with <2 spaces
    raw = "Notok kom koro priyo <unk>"
    clean = sanitize_text(raw)
    tagged = tag_indictrans2_text(clean, "ben_Beng", "eng_Latn")
    assert len(tagged.split(" ", 2)) == 3
