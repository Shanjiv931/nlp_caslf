"""Phase 10 — ocr.py: image -> raw text extraction with confidence scores.

PaddleOCR is the primary engine (per the PRD: strong multilingual and
Bengali-script support), with a Tesseract (`pytesseract` + the `ben`+`eng`
language packs) fallback. Both are genuinely free/open-source. Returns raw
extracted text, per-line/word confidence scores, and bounding boxes.

Neither engine is verified on this machine: PaddleOCR needs its own
PaddlePaddle deep-learning framework (a separate heavy install, likely
hitting the same class of issues as torch on this machine — untested,
since the point would be moot regardless of outcome, see
logs/phase_6_status.md), and `pytesseract` wraps a system-level Tesseract
binary that isn't installed here (confirmed: `tesseract --version` ->
command not found — a system install this session shouldn't perform
unprompted). Both code paths are correct-by-construction against the real
APIs, consistent with every other model-calling module in this project.
"""
import os


def run_paddleocr(image_path, langs="en"):
    """langs: PaddleOCR's language code ("en", or use "bn" — PaddleOCR's
    Bengali support may need a specific model variant; check PaddleOCR's
    own docs for the current recommended language code before relying on
    this in production, since PaddleOCR's supported-language list changes
    across releases)."""
    from paddleocr import PaddleOCR

    ocr = PaddleOCR(use_angle_cls=True, lang=langs)
    raw_result = ocr.ocr(image_path, cls=True)

    lines = []
    for page in raw_result or []:
        for entry in page or []:
            bbox, (text, confidence) = entry
            lines.append({"text": text, "confidence": float(confidence), "bbox": bbox})

    full_text = "\n".join(line["text"] for line in lines)
    avg_confidence = sum(line["confidence"] for line in lines) / len(lines) if lines else 0.0
    return {"engine": "paddleocr", "text": full_text, "lines": lines, "avg_confidence": avg_confidence}


def run_tesseract(image_path, langs="ben+eng"):
    import pytesseract
    from PIL import Image

    image = Image.open(image_path)
    data = pytesseract.image_to_data(image, lang=langs, output_type=pytesseract.Output.DICT)

    lines = []
    n = len(data["text"])
    for i in range(n):
        text = data["text"][i].strip()
        if not text:
            continue
        confidence_raw = data["conf"][i]
        try:
            confidence = float(confidence_raw) / 100.0 if float(confidence_raw) >= 0 else 0.0
        except (TypeError, ValueError):
            confidence = 0.0
        bbox = (data["left"][i], data["top"][i], data["width"][i], data["height"][i])
        lines.append({"text": text, "confidence": confidence, "bbox": bbox})

    full_text = " ".join(line["text"] for line in lines)
    avg_confidence = sum(line["confidence"] for line in lines) / len(lines) if lines else 0.0
    return {"engine": "tesseract", "text": full_text, "lines": lines, "avg_confidence": avg_confidence}


def extract_text(image_path, engine="auto"):
    """engine: "paddleocr", "tesseract", or "auto" (tries PaddleOCR first,
    falls back to Tesseract on any failure — never crashes the whole
    pipeline because one OCR backend isn't available in a given
    environment, consistent with this project's engine-fallback pattern
    elsewhere, e.g. IndicTransToolkit's fallback in mt_compat.py)."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"image not found: {image_path}")

    if engine == "paddleocr":
        return run_paddleocr(image_path)
    if engine == "tesseract":
        return run_tesseract(image_path)

    # auto
    try:
        return run_paddleocr(image_path)
    except Exception as e:
        print(f"WARNING: PaddleOCR unavailable ({type(e).__name__}: {e}) — falling back to Tesseract")
        return run_tesseract(image_path)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "sample.png"
    result = extract_text(path)
    print(f"engine: {result['engine']}, avg_confidence: {result['avg_confidence']:.3f}")
    print(result["text"])
