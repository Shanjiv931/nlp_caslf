"""Unit tests for pipeline/image_to_translation.py. OCR (needs PaddleOCR/
Tesseract, neither available on this machine) and catla.translate_tweet
(needs real models) are mocked; the language gate is REAL (fastText
genuinely runs here, see test_language_gate.py) since it's cheap and this
gives real end-to-end confidence for the gate/routing logic specifically.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import image_to_translation
from image_to_translation import UNSUPPORTED_LANGUAGE_MESSAGE, _hash_image, handle_uploaded_image, log_decision


def _fake_ocr_result(text, confidence=0.9):
    return {
        "engine": "tesseract",
        "text": text,
        "lines": [{"text": text, "confidence": confidence, "bbox": None}],
        "avg_confidence": confidence,
    }


def test_hash_image_consistent_for_same_content(tmp_path):
    f1 = tmp_path / "a.png"
    f2 = tmp_path / "b.png"
    f1.write_bytes(b"same bytes here")
    f2.write_bytes(b"same bytes here")
    assert _hash_image(str(f1)) == _hash_image(str(f2))


def test_hash_image_different_for_different_content(tmp_path):
    f1 = tmp_path / "a.png"
    f2 = tmp_path / "b.png"
    f1.write_bytes(b"content one")
    f2.write_bytes(b"content two")
    assert _hash_image(str(f1)) != _hash_image(str(f2))


def test_hash_image_missing_file_returns_none():
    assert _hash_image("/nonexistent/path/does/not/exist.png") is None


def test_log_decision_writes_valid_jsonl(tmp_path):
    log_path = str(tmp_path / "gate_log.jsonl")
    gate_result = {"language": "bn", "confidence": 0.95, "fasttext_language": "bn", "bengali_unicode_ratio": 1.0}
    log_decision("some/image.png", "আমি ভালো আছি", gate_result, log_path=log_path)

    with open(log_path, encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["detected_language"] == "bn"
    assert entry["routing_decision"] == "bn"
    assert "image_hash" in entry
    assert entry["cleaned_text"] == "আমি ভালো আছি"


def test_log_decision_appends_not_overwrites(tmp_path):
    log_path = str(tmp_path / "gate_log.jsonl")
    gate_result = {"language": "en", "confidence": 0.9, "fasttext_language": "en", "bengali_unicode_ratio": 0.0}
    log_decision("img1.png", "text one", gate_result, log_path=log_path)
    log_decision("img2.png", "text two", gate_result, log_path=log_path)
    with open(log_path, encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 2


def test_unsupported_language_returns_alert_and_never_translates(monkeypatch, tmp_path):
    monkeypatch.setattr(image_to_translation, "extract_text", lambda path, engine="auto": _fake_ocr_result("estoy bien hoy"))
    monkeypatch.setattr(image_to_translation, "cleanup_ocr_result", lambda ocr_result, use_llm=True: {**ocr_result})

    def fail_if_called(*a, **kw):
        raise AssertionError("translate_tweet must NOT be called for unsupported language")

    monkeypatch.setattr(image_to_translation.catla, "translate_tweet", fail_if_called)

    result = handle_uploaded_image("fake.png", log_path=str(tmp_path / "log.jsonl"))
    assert result["supported"] is False
    assert result["message"] == UNSUPPORTED_LANGUAGE_MESSAGE
    assert result["translation"] is None
    assert result["detected_language"] == "unsupported"


def test_bengali_image_routes_to_bn2en(monkeypatch, tmp_path):
    monkeypatch.setattr(image_to_translation, "extract_text", lambda path, engine="auto": _fake_ocr_result("আমি ভালো আছি"))
    monkeypatch.setattr(image_to_translation, "cleanup_ocr_result", lambda ocr_result, use_llm=True: {**ocr_result})
    monkeypatch.setattr(image_to_translation.catla, "translate_tweet",
                         lambda text, direction: {"translation": "I am doing well", "direction": direction})

    result = handle_uploaded_image("fake.png", log_path=str(tmp_path / "log.jsonl"))
    assert result["supported"] is True
    assert result["direction"] == "bn2en"
    assert result["translation"] == "I am doing well"


def test_english_image_routes_to_en2bn(monkeypatch, tmp_path):
    monkeypatch.setattr(image_to_translation, "extract_text", lambda path, engine="auto": _fake_ocr_result("I am doing well today"))
    monkeypatch.setattr(image_to_translation, "cleanup_ocr_result", lambda ocr_result, use_llm=True: {**ocr_result})
    monkeypatch.setattr(image_to_translation.catla, "translate_tweet",
                         lambda text, direction: {"translation": "আমি ভালো আছি", "direction": direction})

    result = handle_uploaded_image("fake.png", log_path=str(tmp_path / "log.jsonl"))
    assert result["supported"] is True
    assert result["direction"] == "en2bn"


def test_every_call_logs_a_decision(monkeypatch, tmp_path):
    monkeypatch.setattr(image_to_translation, "extract_text", lambda path, engine="auto": _fake_ocr_result("hello there world"))
    monkeypatch.setattr(image_to_translation, "cleanup_ocr_result", lambda ocr_result, use_llm=True: {**ocr_result})
    monkeypatch.setattr(image_to_translation.catla, "translate_tweet", lambda text, direction: {"translation": "x", "direction": direction})

    log_path = str(tmp_path / "log.jsonl")
    handle_uploaded_image("fake.png", log_path=log_path)
    assert os.path.exists(log_path)
    with open(log_path, encoding="utf-8") as f:
        assert len(f.readlines()) == 1
