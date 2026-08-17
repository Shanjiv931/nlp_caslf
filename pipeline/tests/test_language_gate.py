"""Unit tests for pipeline/language_gate.py.

Unlike almost every other model-calling module in this project,
`fasttext-wheel` has no torch dependency and genuinely runs on this
machine — these tests use the REAL downloaded `lid.176.ftz` model against
real text samples, not mocks. This is one of the few places in the whole
project where the actual accept/reject accuracy claim can be verified
directly rather than taken on faith pending Kaggle.

Per the PRD: tested against Bengali, English, and 5 distractor languages
(Hindi, Tamil, Arabic, Spanish, French — native script), plus edge cases
(empty, emoji-only, numbers-only).
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from language_gate import bengali_unicode_ratio, detect_and_route, load_fasttext_model

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "lid.176.ftz")


def _model_available():
    """The model auto-downloads on first use, but tests shouldn't require
    network access to even collect — skip gracefully if it's genuinely
    unavailable (offline CI, etc.) rather than fail with a confusing
    network error."""
    try:
        load_fasttext_model()
        return True
    except Exception:
        return False


requires_fasttext = pytest.mark.skipif(not _model_available(), reason="fastText lid.176 model unavailable (offline?)")


# ---- bengali_unicode_ratio: pure logic, no model needed ----

def test_bengali_unicode_ratio_pure_bengali():
    assert bengali_unicode_ratio("আমি ভালো আছি") == 1.0


def test_bengali_unicode_ratio_pure_latin():
    assert bengali_unicode_ratio("hello world") == 0.0


def test_bengali_unicode_ratio_no_alphabetic_chars():
    assert bengali_unicode_ratio("12345 !!!") == 0.0


def test_bengali_unicode_ratio_code_mixed():
    ratio = bengali_unicode_ratio("আমি feeling bhalo")  # bengali word + 2 latin "words"
    assert 0.0 < ratio < 1.0


# ---- detect_and_route: real fastText model, real accuracy evidence ----

@requires_fasttext
def test_bengali_correctly_accepted():
    result = detect_and_route("আমি ভালো আছি এবং আজকে আবহাওয়া খুব সুন্দর")
    assert result["language"] == "bn"


@requires_fasttext
def test_english_correctly_accepted():
    result = detect_and_route("I am doing well and the weather is very nice today")
    assert result["language"] == "en"


@requires_fasttext
def test_hindi_native_script_correctly_rejected():
    result = detect_and_route("मैं ठीक हूं और आज मौसम बहुत अच्छा है")
    assert result["language"] == "unsupported"
    assert result["fasttext_language"] == "hi"


@requires_fasttext
def test_tamil_correctly_rejected():
    result = detect_and_route("நான் நலமாக இருக்கிறேன், இன்று வானிலை மிகவும் அழகாக இருக்கிறது")
    assert result["language"] == "unsupported"
    assert result["fasttext_language"] == "ta"


@requires_fasttext
def test_arabic_correctly_rejected():
    result = detect_and_route("أنا بخير والجو جميل جدا اليوم")
    assert result["language"] == "unsupported"
    assert result["fasttext_language"] == "ar"


@requires_fasttext
def test_spanish_correctly_rejected():
    result = detect_and_route("estoy bien y el clima esta muy agradable hoy en la ciudad")
    assert result["language"] == "unsupported"
    assert result["fasttext_language"] == "es"


@requires_fasttext
def test_french_correctly_rejected():
    result = detect_and_route("je vais tres bien et le temps est tres agreable aujourdhui")
    assert result["language"] == "unsupported"
    assert result["fasttext_language"] == "fr"


@requires_fasttext
def test_empty_input_rejected():
    result = detect_and_route("")
    assert result["language"] == "unsupported"
    assert result["reason"] == "empty input"


@requires_fasttext
def test_whitespace_only_rejected():
    result = detect_and_route("     ")
    assert result["language"] == "unsupported"


@requires_fasttext
def test_emoji_only_rejected():
    result = detect_and_route("😀😀😀")
    assert result["language"] == "unsupported"


@requires_fasttext
def test_numbers_only_rejected():
    result = detect_and_route("12345 67890")
    assert result["language"] == "unsupported"


@requires_fasttext
def test_bengali_unicode_ratio_overrides_low_fasttext_confidence():
    # a short/ambiguous but genuinely Bengali-script snippet should be
    # accepted on the strength of the Unicode signal even if fastText
    # itself is uncertain about such a short string
    result = detect_and_route("ভালো")
    assert result["language"] == "bn"
    assert result["reason"] == "bengali_unicode_ratio"


@requires_fasttext
def test_KNOWN_LIMITATION_romanized_hindi_misclassified_as_english():
    """Documented, disclosed limitation, not silently accepted as correct
    behavior: fastText's lid.176 was trained predominantly on native-script
    text per language. Romanized Hindi ("Hinglish") can score as English
    with genuinely high confidence (0.725 observed, well above the 0.5
    gate threshold) rather than low-confidence-and-therefore-rejected --
    this is a real false-accept fastText itself is confident about, not a
    threshold-tuning problem. This test exists to make the limitation
    explicit and trackable (it will fail loudly if a future fastText
    version or gate change happens to fix it, which is a good thing to
    notice), not to assert the wrong behavior is desirable.
    """
    result = detect_and_route("yah main theek hoon aur aaj mausam bahut accha hai")
    assert result["language"] == "en"  # the actual (wrong) current behavior
    assert result["fasttext_language"] == "en"
    assert result["confidence"] > 0.5  # genuinely confident, not a borderline case a threshold tweak would catch
