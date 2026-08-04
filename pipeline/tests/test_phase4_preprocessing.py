"""Unit tests for Phase 4's tweet-aware preprocessing modules, run against
hand-written example tweets. Run with: pytest pipeline/tests/ -v
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tokenizer import tokenize, protected_spans
from hashtag_segmenter import segment_hashtag
from slang_normalizer import normalize_slang
from transliterator import bn_to_latin, latin_to_bn
from reassembler import reassemble, rebuild_hashtag


# ---- tokenizer ----

def test_tokenizer_splits_hashtag_mention_url_emoji_as_atomic():
    text = "Just watched #EndGame with @marvel fans \U0001F60D\U0001F525 https://t.co/xyz amazing!!"
    tokens = tokenize(text)
    kinds = {kind for _, kind in tokens}
    assert "hashtag" in kinds and "mention" in kinds and "url" in kinds and "emoji" in kinds
    hashtags = [t for t, k in tokens if k == "hashtag"]
    assert hashtags == ["#EndGame"]
    mentions = [t for t, k in tokens if k == "mention"]
    assert mentions == ["@marvel"]
    urls = [t for t, k in tokens if k == "url"]
    assert urls == ["https://t.co/xyz"]
    emojis = [t for t, k in tokens if k == "emoji"]
    assert emojis == ["\U0001F60D", "\U0001F525"]


def test_tokenizer_plain_words_and_punct():
    tokens = tokenize("hello, world!")
    words = [t for t, k in tokens if k == "word"]
    puncts = [t for t, k in tokens if k == "punct"]
    assert words == ["hello", "world"]
    assert puncts == [",", "!"]


def test_protected_spans_matches_tokenizer():
    text = "#bangla আমি ভালো আছি @user \U0001F60D"
    spans = protected_spans(text)
    assert "#bangla" in spans
    assert "@user" in spans
    assert "\U0001F60D" in spans


# ---- hashtag_segmenter ----

def test_segment_hashtag_camelcase():
    assert segment_hashtag("#IWasHappyBefore") == ["I", "Was", "Happy", "Before"]


def test_segment_hashtag_leading_hash_optional():
    assert segment_hashtag("IWasHappyBefore") == segment_hashtag("#IWasHappyBefore")


def test_segment_hashtag_empty():
    assert segment_hashtag("#") == []


def test_segment_hashtag_returns_nonempty_list_for_lowercase():
    # no camelCase signal -> falls back to frequency-based Viterbi split;
    # exact segmentation isn't asserted (depends on the corpus-derived
    # dictionary), only that it returns a sane non-empty token list that
    # reconstructs the original string.
    result = segment_hashtag("#mydayout")
    assert result
    assert "".join(result).lower() == "mydayout"


# ---- slang_normalizer ----

def test_normalize_english_abbreviations():
    assert normalize_slang("u r gr8, thx!") == "you are great, thanks!"


def test_normalize_preserves_titlecase():
    assert normalize_slang("Pls help") == "Please help"


def test_normalize_leaves_unknown_words_untouched():
    assert normalize_slang("hello unknownword123") == "hello unknownword123"


def test_normalize_banglish_slang():
    out = normalize_slang("kmn asos vai?")
    assert "kemon" in out
    assert "acho" in out
    assert "bhai" in out


# ---- transliterator ----

def test_bn_to_latin_basic_phrase():
    result = bn_to_latin("আমি ভালো আছি")
    assert result == "ami bhalo achhi"


def test_latin_to_bn_produces_bengali_script():
    result = latin_to_bn("ami bhalo achi")
    # every alphabetic-origin character in the output should be in the
    # Bengali Unicode block (spaces pass through unchanged)
    letters = [c for c in result if c != " "]
    assert letters
    assert all(0x0980 <= ord(c) <= 0x09FF for c in letters)


def test_transliterator_roundtrip_is_plausible_not_exact():
    # Documented limitation: this is a rule-based phonetic baseline, not a
    # trained model, so exact round-trip fidelity isn't expected — only
    # that both directions produce non-empty, script-appropriate output.
    bn = "তুমি কেমন আছো"
    latin = bn_to_latin(bn)
    assert latin and latin.isascii()
    back = latin_to_bn(latin)
    assert back and any(0x0980 <= ord(c) <= 0x09FF for c in back)


# ---- reassembler ----

def test_rebuild_hashtag_english_camelcase():
    assert rebuild_hashtag("#test", ["good", "morning"], "en") == "#GoodMorning"


def test_rebuild_hashtag_bengali_joined():
    assert rebuild_hashtag("#test", ["শুভ", "সকাল"], "bn") == "#শুভসকাল"


def test_rebuild_hashtag_empty_words_falls_back_to_original():
    assert rebuild_hashtag("#original", [], "en") == "#original"


def test_reassemble_reinserts_dropped_protected_tokens():
    src = "Just watched #EndGame with @marvel fans \U0001F60D amazing!!"
    mt_output = "sobe #EndGame dekhlam onek bhalo laglo!!"  # mention+emoji dropped by hypothetical MT
    result = reassemble(src, mt_output, target_lang="bn")
    assert "@marvel" in result
    assert "\U0001F60D" in result
    assert "#EndGame" in result  # already present, not duplicated
    assert result.count("#EndGame") == 1


def test_reassemble_preserves_tokens_already_in_translation():
    src = "hello @user"
    mt_output = "hola @user"
    result = reassemble(src, mt_output, target_lang="en")
    assert result.count("@user") == 1
