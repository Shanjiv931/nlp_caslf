"""Unit tests for pipeline/ocr_cleanup.py's rule-based cleanup (pure text
processing, genuinely runs and is tested for real — no OCR engine or LLM
needed). The LLM-assisted pass (`llm_fix_span`) and `cleanup_ocr_result`'s
orchestration around it are tested separately with mocks, matching this
project's established pattern for model-calling code.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ocr_cleanup import cleanup_ocr_result, rule_based_cleanup


def test_danda_misread_fixed_when_adjacent_to_bengali():
    result = rule_based_cleanup("আমি ভালো আছি | সে ভালো")
    assert "।" in result
    assert "|" not in result


def test_pipe_not_touched_in_pure_english_text():
    # must not corrupt genuine English "|" (e.g. a table separator)
    assert rule_based_cleanup("hello | world") == "hello | world"


def test_pipe_not_touched_when_isolated_from_bengali():
    result = rule_based_cleanup("this is english | not bengali")
    assert "|" in result  # unchanged, no Bengali adjacency


def test_whitespace_collapsed():
    result = rule_based_cleanup("নমস্কার   \t\t সবাই")
    assert "  " not in result
    assert "\t" not in result


def test_excess_newlines_collapsed():
    result = rule_based_cleanup("line one\n\n\n\n\nline two")
    assert "\n\n\n" not in result


def test_control_characters_stripped():
    result = rule_based_cleanup("hello\x00\x01world")
    assert "\x00" not in result
    assert "\x01" not in result


def test_nfc_normalization_applied():
    import unicodedata

    # an NFD-decomposed Bengali string (base + combining marks separately)
    # should come out NFC-composed, matching the rest of this project's
    # data (Phase 3's unified.jsonl etc. is NFC throughout)
    nfd_text = unicodedata.normalize("NFD", "আমি ভালো আছি")
    result = rule_based_cleanup(nfd_text)
    assert unicodedata.is_normalized("NFC", result)


def test_cleanup_is_idempotent():
    # running cleanup twice should be a no-op the second time -- a good
    # sanity property for any text-normalization function
    once = rule_based_cleanup("আমি ভালো আছি | সে ভালো   বাই")
    twice = rule_based_cleanup(once)
    assert once == twice


def test_empty_string():
    assert rule_based_cleanup("") == ""


# ---- cleanup_ocr_result orchestration, mocked LLM ----

def test_cleanup_ocr_result_applies_rule_based_to_all_lines():
    ocr_result = {
        "text": "আমি ভালো আছি | সে",
        "lines": [
            {"text": "আমি ভালো আছি |", "confidence": 0.95, "bbox": None},
            {"text": "সে", "confidence": 0.98, "bbox": None},
        ],
        "avg_confidence": 0.96,
        "engine": "tesseract",
    }
    result = cleanup_ocr_result(ocr_result, use_llm=False)
    assert "|" not in result["lines"][0]["text"]
    assert result["llm_cleanup_applied_count"] == 0


def test_cleanup_ocr_result_only_applies_llm_below_threshold(monkeypatch):
    import ocr_cleanup

    monkeypatch.setattr(ocr_cleanup, "llm_fix_span", lambda text, model_names=None: f"FIXED:{text}")

    ocr_result = {
        "text": "high conf line low conf line",
        "lines": [
            {"text": "high conf line", "confidence": 0.9, "bbox": None},
            {"text": "low conf line", "confidence": 0.3, "bbox": None},
        ],
        "avg_confidence": 0.6,
        "engine": "tesseract",
    }
    result = cleanup_ocr_result(ocr_result, low_confidence_threshold=0.6, use_llm=True)
    assert result["lines"][0]["text"] == "high conf line"  # untouched by LLM
    assert result["lines"][1]["text"].startswith("FIXED:")
    assert result["llm_cleanup_applied_count"] == 1


def test_cleanup_ocr_result_llm_failure_falls_back_to_rule_based(monkeypatch):
    import ocr_cleanup

    def broken_llm_fix(text, model_names=None):
        raise RuntimeError("simulated LLM failure")

    monkeypatch.setattr(ocr_cleanup, "llm_fix_span", broken_llm_fix)

    ocr_result = {
        "text": "low conf",
        "lines": [{"text": "low conf | line", "confidence": 0.2, "bbox": None}],
        "avg_confidence": 0.2,
        "engine": "tesseract",
    }
    result = cleanup_ocr_result(ocr_result, use_llm=True)
    assert result["lines"][0]["text"]  # didn't crash, still has rule-based-cleaned text
