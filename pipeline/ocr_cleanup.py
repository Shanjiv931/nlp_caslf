"""Phase 10 — ocr_cleanup.py: rule-based + LLM-assisted OCR error
correction, applied before language detection.

Rule-based pass (pure text processing, no model — genuinely testable and
tested locally, unlike almost everything else model-adjacent in this
project):
1. Unicode NFC normalization — a real, meaningful fix for a specific,
   common category of "broken conjuncts": many apparent conjunct-splitting
   issues in OCR'd Bengali text are actually the *same* visual glyph
   represented in decomposed (NFD) form rather than composed (NFC) form,
   which downstream tokenizers/models (all trained on NFC-normalized
   corpora, like the rest of this project's data) don't handle the same
   way. This is disclosed honestly as covering *one* real category of
   conjunct issues, not a general Bengali-OCR-error-corrector — genuinely
   garbled conjuncts (wrong character substituted, not just wrong
   normalization form) need either a curated correction-pattern list built
   from real OCR error data (not fabricated here) or the LLM-assisted pass
   below.
2. Common Bengali danda (।) misreads — OCR engines not well-tuned for
   Bengali script frequently misread the Bengali full-stop/danda as an
   ASCII "|", "I", or "l" when it appears directly adjacent to Bengali
   script (no misread-danda pattern fires on text that's mostly Latin
   script, to avoid corrupting genuine English "I"/"l" characters).
3. Whitespace/control-character cleanup.

LLM-assisted pass: for spans OCR reported low confidence on, passes them
through the same open LLM `llm_postedit.py` uses, with a narrow prompt
("fix obvious OCR typos only, do not translate or rephrase") — reuses
`llm_postedit.load_llm()` rather than loading a second copy of the model.
"""
import re
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

BENGALI_RANGE = re.compile(r"[ঀ-৿]")
# ASCII chars OCR commonly substitutes for the Bengali danda (।) when
# scanning Bengali script — only corrected when genuinely adjacent to
# Bengali characters, never in isolation (would corrupt real English text).
DANDA_MISREAD_NEAR_BENGALI = re.compile(r"(?<=[ঀ-৿])\s*[|Il1]\s*(?=[ঀ-৿]|\s|$)")

CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
MULTI_SPACE_RE = re.compile(r"[ \t]+")
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def rule_based_cleanup(text: str) -> str:
    import unicodedata

    text = unicodedata.normalize("NFC", text)
    text = CONTROL_CHAR_RE.sub("", text)
    text = DANDA_MISREAD_NEAR_BENGALI.sub("।", text)
    text = MULTI_SPACE_RE.sub(" ", text)
    text = MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


OCR_FIX_SYSTEM_PROMPT = (
    "You fix obvious OCR (optical character recognition) errors in short text "
    "snippets. Fix only clear character-recognition mistakes. Do NOT translate, "
    "rephrase, summarize, or change the language. Do NOT add or remove words that "
    "aren't OCR artifacts. Return ONLY the corrected text, with no explanation."
)


def llm_fix_span(text: str, model_names=None) -> str:
    """Runs one low-confidence OCR span through the LLM post-editor's
    model with a narrow OCR-fixing prompt. Reuses llm_postedit.load_llm()
    (same cache) rather than loading a second copy of a multi-GB model."""
    from llm_postedit import generate_postedit, load_llm

    model, tokenizer, _ = load_llm(model_names)
    user_prompt = f"Fix OCR errors in this text:\n{text}"
    return generate_postedit(model, tokenizer, OCR_FIX_SYSTEM_PROMPT, user_prompt, max_new_tokens=128)


def cleanup_ocr_result(ocr_result: dict, low_confidence_threshold: float = 0.6, use_llm: bool = True) -> dict:
    """ocr_result: the dict from ocr.py's extract_text() — {"text", "lines",
    "avg_confidence", ...}. Applies rule-based cleanup to the full text
    always; applies the LLM-assisted pass only to individual `lines` whose
    OCR-reported confidence is below `low_confidence_threshold`, and only
    if `use_llm` (callers on constrained hardware, or evaluating rule-based
    cleanup in isolation, can skip the LLM pass entirely)."""
    cleaned_lines = []
    llm_used_count = 0
    for line in ocr_result.get("lines", []):
        cleaned_text = rule_based_cleanup(line["text"])
        if use_llm and line["confidence"] < low_confidence_threshold and cleaned_text:
            try:
                cleaned_text = llm_fix_span(cleaned_text)
                llm_used_count += 1
            except Exception as e:
                print(f"WARNING: LLM-assisted OCR cleanup failed for a low-confidence span "
                      f"({type(e).__name__}: {e}) — kept rule-based cleanup only for that span")
        cleaned_lines.append({**line, "text": cleaned_text})

    full_cleaned_text = rule_based_cleanup(ocr_result.get("text", ""))
    return {
        **ocr_result,
        "text": full_cleaned_text,
        "lines": cleaned_lines,
        "llm_cleanup_applied_count": llm_used_count,
    }


if __name__ == "__main__":
    sample = "আমি ভালো‌ আছি | সে ভালো"  # NFD-ish artifact + a misread danda
    print(rule_based_cleanup(sample).encode("ascii", "backslashreplace").decode())
