"""Phase 4 — reassembler.

Given the original source tweet and a translated string produced by the
quality pipeline (Phase 7), re-inserts protected spans (emoji, mentions,
URLs) that the translation step was told never to touch, and rebuilds
hashtags in the target language from their segmented+translated words.

Protected non-hashtag tokens (emoji/mentions/URLs) are assumed to already
appear verbatim in the translated text — that's the contract Phase 7's LLM
post-editor prompt enforces (never alter protected tokens). This module's
job is mainly hashtag reconstruction, plus a defensive pass that re-appends
any protected token the translation step dropped, so nothing silently
disappears.

Hashtags are handled differently from the other protected-token types, and
deliberately so: `llm_postedit.py` puts hashtags in the SAME "must appear
unchanged" protected-token list as emoji/mentions/URLs — correct, so the
LLM doesn't mangle them mid-edit — but that means the source-language
hashtag genuinely does end up sitting verbatim inside the translated
sentence by the time it reaches this module. An earlier version of this
function treated that as success ("already preserved, nothing to do"),
which meant hashtags were NEVER actually segmented/translated/rebuilt in
the success path — the exact opposite of the PRD's FR8 ("hashtags are
segmented, translated/transliterated, and rebuilt as valid target-language
hashtags"), silently leaving a source-language hashtag stuck inside an
otherwise fully-translated sentence. Fixed 2026-08-16, while building
Phase 9: hashtags are now always segmented+translated+rebuilt, and if the
old verbatim source-language hashtag is present in the MT output, it's
removed (not left duplicated alongside the new one).
"""
from tokenizer import tokenize
from hashtag_segmenter import segment_hashtag


def rebuild_hashtag(original_hashtag, translated_words, target_lang):
    """Rebuild a single hashtag from its translated constituent words.

    target_lang: "en" or "bn". English hashtags are rebuilt in CamelCase
    (the social-media convention); Bengali hashtags are joined without
    separators (Bengali script has no case, and CamelCase-style joining
    isn't a real convention there), following common Bengali-Twitter usage.
    """
    words = [w for w in translated_words if w]
    if not words:
        return original_hashtag
    if target_lang == "en":
        body = "".join(w[:1].upper() + w[1:] for w in words)
    else:
        body = "".join(words)
    return "#" + body


def reassemble(source_text, translated_text, target_lang, translate_word_fn=None):
    """Reinsert protected tokens and rebuild hashtags.

    translate_word_fn: optional callable(word:str) -> str used to translate
    each segmented hashtag word into the target language before rebuilding.
    If omitted, hashtag words are carried over untranslated (segmentation
    only) — Phase 9's catla.py supplies the real translator.
    """
    result = translated_text
    for token, kind in tokenize(source_text):
        if kind == "hashtag":
            # Always rebuild in the target language — never treat "the old
            # source-language hashtag is still sitting in the MT output" as
            # done, since llm_postedit.py's protected-token contract means
            # it will be there almost every time. Remove it first so the
            # rebuilt, translated hashtag doesn't end up duplicated
            # alongside the untranslated original.
            if token in result:
                result = result.replace(token, "").rstrip()
            words = segment_hashtag(token)
            if translate_word_fn:
                words = [translate_word_fn(w) for w in words]
            rebuilt = rebuild_hashtag(token, words, target_lang)
            result = result.rstrip() + " " + rebuilt
        elif kind in ("mention", "url", "emoji"):
            if token not in result:
                result = result.rstrip() + " " + token
    return result


if __name__ == "__main__":
    src = "Just watched #EndGame with @marvel fans 😍 amazing!!"
    mt_output = "sobe #EndGame dekhlam onek bhalo laglo!!"  # hashtag/mention/emoji dropped by a hypothetical MT step
    print(reassemble(src, mt_output, target_lang="bn").encode("ascii", "backslashreplace").decode())
