"""Phase 9 — catla.py: the full end-to-end callable,
`translate_tweet(text, direction) -> dict`.

Wires: slang_normalizer.py -> (conditional) transliterator.py ->
quality_pipeline.produce_translation() -> reassembler.py, per the PRD.

Preprocessing order and rationale:
1. `normalize_slang()` always runs first, regardless of direction — its
   combined Banglish+English slang dictionary is safe to apply either way
   (whole-word matching only, case-preserving), and normalizing "kmn" ->
   "kemon" or "u" -> "you" gives every downstream stage cleaner input.
2. Transliteration (`transliterator.latin_to_bn()`) only runs for the
   bn2en direction, and only when the input `looks_fully_romanized_bengali`
   — i.e. Latin-script with *zero* Bengali Unicode presence at all. This is
   deliberately conservative: genuinely code-mixed input (Bengali script
   with some English words/loanwords mixed in, which is common and is what
   this project's fine-tuned engines were actually trained on via BanTH/
   BnSentMix/etc.) is left alone rather than blindly transliterated, which
   would corrupt real English content sitting inside otherwise-Bengali
   text. Only text with *no* Bengali script at all — genuinely fully
   romanized "Banglish" — gets transliterated, to bring it into the
   Bengali-script distribution the MT engines were actually trained on.
3. `quality_pipeline.produce_translation()` does the actual translation.
4. `reassembler.reassemble()` handles hashtag translation (word-level,
   via a lightweight single-engine translator — running the full
   ensemble+QE+LLM+verify pipeline per hashtag word would be far too
   expensive) and the protected-token safety net for mentions/URLs/emoji.

This machine cannot run torch (see logs/phase_6_status.md through
phase_8_status.md) — `translate_tweet()`'s actual model-calling behavior
is unverified here, correct-by-construction against the already-built and
tested pipeline modules it wires together. The pure-logic pieces
(`looks_fully_romanized_bengali`, the preprocessing decisions) and the
full orchestration with mocked stages are unit-tested locally.
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))

import ensemble
import quality_pipeline
from reassembler import reassemble
from slang_normalizer import normalize_slang
from transliterator import latin_to_bn

BENGALI_RANGE = re.compile(r"[ঀ-৿]")
LATIN_ALPHA_RANGE = re.compile(r"[A-Za-z]")

_WORD_TRANSLATE_CACHE = {}


def looks_fully_romanized_bengali(text):
    """True if `text` is entirely Latin-script with zero Bengali Unicode
    presence — suggests fully-romanized "Banglish" input rather than
    code-mixed Bengali-script text with some English words in it.
    Deliberately conservative (see module docstring): only fires when
    there's NO Bengali script at all, so genuinely code-mixed tweets
    aren't blindly transliterated and have real English content corrupted."""
    return bool(LATIN_ALPHA_RANGE.search(text)) and not bool(BENGALI_RANGE.search(text))


def make_word_translator(direction, engine_key="nllb"):
    """Builds a translate_word_fn for reassembler.reassemble() — single
    engine, beam-search-only, word-level translation. NOT the full quality
    pipeline (ensemble+QE+LLM+verify): that's far too expensive to run once
    per hashtag word, and overkill for a single word with no sentence-level
    context to disambiguate. Applies the same romanization-detection as the
    main sentence body for consistency, and fails safe (keeps the original
    word) rather than crashing hashtag rebuilding if translation fails."""

    def translate_word(word):
        cache_key = (engine_key, direction, word.lower())
        if cache_key in _WORD_TRANSLATE_CACHE:
            return _WORD_TRANSLATE_CACHE[cache_key]

        word_to_translate = word
        if direction == "bn2en" and looks_fully_romanized_bengali(word):
            word_to_translate = latin_to_bn(word)

        try:
            ctx = ensemble.load_engine(engine_key, direction)
            candidates = ensemble.generate_candidates_for_engine(
                ctx, word_to_translate, num_beam_candidates=1, num_sample_candidates=0,
            )
            result = candidates[0].text if candidates else word
        except Exception:
            result = word

        _WORD_TRANSLATE_CACHE[cache_key] = result
        return result

    return translate_word


def translate_tweet(text, direction, engines=None, roundtrip_engine="nllb") -> dict:
    """The single end-to-end entry point the Gradio demo (Phase 11) and
    the CLI below both call."""
    target_lang = "en" if direction == "bn2en" else "bn"

    normalized = normalize_slang(text)

    transliterated = False
    processed = normalized
    if direction == "bn2en" and looks_fully_romanized_bengali(normalized):
        processed = latin_to_bn(normalized)
        transliterated = True

    pipeline_result = quality_pipeline.produce_translation(
        processed, direction, engines=engines, roundtrip_engine=roundtrip_engine,
    )

    translation = pipeline_result.get("translation")
    if translation:
        word_translator = make_word_translator(direction, engine_key=roundtrip_engine)
        final_text = reassemble(text, translation, target_lang, translate_word_fn=word_translator)
    else:
        final_text = None

    result = {
        "original_text": text,
        "direction": direction,
        "normalized_text": normalized,
        "transliterated": transliterated,
        "processed_text": processed,
        "translation": final_text,
    }
    result.update({k: v for k, v in pipeline_result.items() if k != "translation"})
    return result


def main():
    p = argparse.ArgumentParser(description="Translate a Bengali/English tweet with the full CATLA pipeline.")
    p.add_argument("text", help="The tweet text to translate.")
    p.add_argument("--direction", required=True, choices=["bn2en", "en2bn"])
    args = p.parse_args()

    result = translate_tweet(args.text, args.direction)
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
