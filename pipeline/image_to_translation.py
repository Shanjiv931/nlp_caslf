"""Phase 10 — image_to_translation.py: wires ocr.py -> ocr_cleanup.py ->
language_gate.py -> (if supported) catla.translate_tweet() into one
function, `handle_uploaded_image()`. The Gradio demo (Phase 11) calls this
directly.

If the detected language isn't Bengali or English, returns the PRD's
required alert message and does NOT call the translation pipeline at all
— a correctness requirement (never mistranslate or hallucinate on
unsupported input), not just UX polish.

Every decision (image hash, cleaned OCR text, detected language,
confidence, routing decision) is logged to
`logs/language_gate_log.jsonl` per the PRD, for auditing and for building
a labeled dataset to keep improving the gate. Logs a SHA-256 hash of the
image bytes, not the file path or the image itself — basic privacy
hygiene, since this log may be reviewed later and shouldn't leak local
filesystem structure or require retaining image content.
"""
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))

import catla
from language_gate import detect_and_route
from ocr import extract_text
from ocr_cleanup import cleanup_ocr_result

UNSUPPORTED_LANGUAGE_MESSAGE = (
    "⚠️ This language is not currently supported. CATLA currently supports Bengali and English only."
)

LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "logs", "language_gate_log.jsonl")


def _hash_image(image_path):
    try:
        with open(image_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    except Exception:
        return None


def log_decision(image_path, cleaned_text, gate_result, log_path=LOG_PATH):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "image_hash": _hash_image(image_path),
        "cleaned_text": cleaned_text,
        "detected_language": gate_result["language"],
        "confidence": gate_result["confidence"],
        "fasttext_language": gate_result["fasttext_language"],
        "bengali_unicode_ratio": gate_result["bengali_unicode_ratio"],
        "routing_decision": gate_result["language"] if gate_result["language"] != "unsupported" else "rejected",
    }
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def handle_uploaded_image(image_path, ocr_engine="auto", use_llm_ocr_cleanup=True, log_path=LOG_PATH,
                           fast_mode=False, roundtrip_engine="nllb") -> dict:
    ocr_result = extract_text(image_path, engine=ocr_engine)
    cleaned = cleanup_ocr_result(ocr_result, use_llm=use_llm_ocr_cleanup)
    cleaned_text = cleaned["text"]

    gate_result = detect_and_route(cleaned_text)
    log_decision(image_path, cleaned_text, gate_result, log_path=log_path)

    result = {
        "ocr_text": ocr_result["text"],
        "ocr_confidence": ocr_result["avg_confidence"],
        "ocr_engine": ocr_result["engine"],
        "cleaned_text": cleaned_text,
        "detected_language": gate_result["language"],
        "language_confidence": gate_result["confidence"],
        "supported": gate_result["language"] != "unsupported",
        "message": None,
        "direction": None,
        "translation": None,
        "translation_result": None,
    }

    if not result["supported"]:
        result["message"] = UNSUPPORTED_LANGUAGE_MESSAGE
        return result

    direction = "bn2en" if gate_result["language"] == "bn" else "en2bn"
    translation_result = catla.translate_tweet(cleaned_text, direction, roundtrip_engine=roundtrip_engine,
                                                 fast_mode=fast_mode)
    result["direction"] = direction
    result["translation_result"] = translation_result
    result["translation"] = translation_result.get("translation")
    return result


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "sample.png"
    result = handle_uploaded_image(path)
    for k, v in result.items():
        if k != "translation_result":
            print(f"{k}: {v}")
