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


# ---- ensure_transformers_onnx_shim() ----
# Unlike ensure_tie_weights_compat() (needs transformers.modeling_utils,
# which imports torch -- unavailable on this machine), this shim is pure
# module-registration logic with no torch dependency at all, and
# `transformers` itself (without torch) genuinely imports here -- so this
# gets real, not just correct-by-construction, verification.

import inspect

import mt_compat


def _fresh_onnx_shim():
    # Each call to ensure_transformers_onnx_shim() early-returns if
    # transformers.onnx.utils is already importable, so force a fresh
    # shim by clearing any previous run's entries first.
    sys.modules.pop("transformers.onnx", None)
    sys.modules.pop("transformers.onnx.utils", None)
    mt_compat.ensure_transformers_onnx_shim()
    return sys.modules["transformers.onnx"], sys.modules["transformers.onnx.utils"]


def test_onnx_shim_file_attribute_is_a_real_string():
    # Root cause of a real bug hit on Kaggle 2026-08-17: the shim's
    # __getattr__ used to auto-synthesize a *class* for __file__ (and any
    # other dunder), which crashed unrelated code (Python's own `inspect`
    # module, invoked transitively by torch._dynamo while just importing
    # transformers.modeling_utils) with `AttributeError: type object
    # '__file__' has no attribute 'endswith'`.
    onnx_shim, utils_shim = _fresh_onnx_shim()
    assert isinstance(onnx_shim.__file__, str)
    assert isinstance(utils_shim.__file__, str)


def test_onnx_shim_survives_real_inspect_getsourcefile():
    # This is the literal function that crashed in the real traceback
    # (inspect.getsourcefile -> filename.endswith(...) on a synthesized
    # class instead of a string) -- exercised for real here, not mocked.
    onnx_shim, utils_shim = _fresh_onnx_shim()
    assert inspect.getsourcefile(onnx_shim) is None or isinstance(inspect.getsourcefile(onnx_shim), str)
    assert inspect.getabsfile(onnx_shim)  # must not raise


def test_onnx_shim_unset_dunder_attributes_report_false_via_hasattr():
    # Any dunder the shim never explicitly sets, and that a bare
    # types.ModuleType doesn't define by default either, must behave like a
    # normal module that lacks it (hasattr -> False) -- not silently
    # synthesize a fake class for it. (__loader__/__spec__/__package__ are
    # real attributes types.ModuleType defines as None by default, so
    # they're excluded here; __file__ is handled explicitly by the shim,
    # covered by test_onnx_shim_file_attribute_is_a_real_string above.)
    onnx_shim, utils_shim = _fresh_onnx_shim()
    assert hasattr(onnx_shim, "__path__") is True  # explicitly set by the shim itself
    assert hasattr(utils_shim, "__path__") is False  # never set on the .utils submodule
    assert hasattr(onnx_shim, "__nonsense_dunder_nobody_asked_for__") is False


def test_onnx_shim_still_auto_synthesizes_non_dunder_attributes():
    # Regression guard: the actual purpose of this shim is letting
    # IndicTrans2's remote-code `from transformers.onnx.utils import
    # SomeClass` style imports succeed with a permissive placeholder --
    # that behavior must be untouched by the dunder fix above.
    onnx_shim, utils_shim = _fresh_onnx_shim()
    synthesized = utils_shim.SomeRandomExportClassIndicTrans2MightImport
    assert isinstance(synthesized, type)
    synthesized_instance = synthesized(1, 2, keyword="anything")  # _Permissive accepts any args
    assert synthesized_instance is not None


def test_onnx_shim_is_idempotent_and_safe_to_call_repeatedly():
    mt_compat.ensure_transformers_onnx_shim()
    mt_compat.ensure_transformers_onnx_shim()
    import transformers.onnx.utils  # noqa: F401 -- must not raise on repeated calls
