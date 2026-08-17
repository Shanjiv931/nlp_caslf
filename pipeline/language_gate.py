"""Phase 10 — language_gate.py: language identification + the accept/
reject decision.

Combines a statistical language-ID model (fastText's free `lid.176`, 176
languages) with a fast Unicode-range heuristic (Bengali block presence
ratio) as a cross-check, per the PRD. If the detected language is neither
Bengali nor English, or confidence is too low to be sure, this rejects
rather than guesses — a correctness requirement, not a nice-to-have (per
this project's PRD, "the moment the language-detection stage determines
the input is neither Bengali nor English, the system must say so plainly
rather than mistranslate or hallucinate").

Unlike every other model-calling module in this project, `fasttext-wheel`
has NO torch dependency and genuinely runs on this machine — so this
module (and its tests) are actually verified for real against the real
downloaded `lid.176.ftz` model, not just written correct-by-construction.
"""
import os
import re
import urllib.request

BENGALI_RANGE = re.compile(r"[ঀ-৿]")
LATIN_RANGE = re.compile(r"[A-Za-z]")

FASTTEXT_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
FASTTEXT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "lid.176.ftz")

SUPPORTED_LANGUAGES = {"bn", "en"}
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
BENGALI_UNICODE_ACCEPT_RATIO = 0.5  # if at least half the alphabetic chars are Bengali-script, trust that over fastText

_MODEL_CACHE = {}


def download_fasttext_model(path=FASTTEXT_MODEL_PATH, url=FASTTEXT_MODEL_URL):
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    urllib.request.urlretrieve(url, path)
    return path


def load_fasttext_model(path=FASTTEXT_MODEL_PATH):
    if path in _MODEL_CACHE:
        return _MODEL_CACHE[path]
    import fasttext
    model_path = download_fasttext_model(path)
    model = fasttext.load_model(model_path)
    _MODEL_CACHE[path] = model
    return model


def _safe_fasttext_predict(model, text, k=1):
    """fasttext-wheel's own `FastText.predict()` calls `np.array(probs,
    copy=False)` internally (FastText.py, ~line 228), which raises under
    numpy>=2.0 — numpy 2.x's stricter `copy=False` semantics no longer
    silently fall back to copying when the input isn't already an ndarray,
    and `probs` there is always a plain Python tuple, never one already.
    Confirmed for real by reading the installed library source directly
    and reproducing the exact `ValueError: Unable to avoid copy...`.
    Downgrading numpy globally was considered and rejected: risky, since
    this project's `pandas`/other dependencies are already on numpy>=2.0-era
    releases and a downgrade could reopen unrelated conflicts for the sake
    of one library's unfixed internal bug. Instead, calls the same
    lower-level `model.f.predict(...)` binding `predict()` itself calls
    internally, skipping the one buggy line, converting the result with a
    plain Python `list()` instead of the numpy call that breaks.
    """
    text = text.replace("\n", " ") + "\n"
    predictions = model.f.predict(text, k, 0.0, "strict")
    if predictions:
        probs, labels = zip(*predictions)
    else:
        probs, labels = (), ()
    return labels, list(probs)


def fasttext_predict(text, model=None):
    """Returns (lang_code, confidence) for the top-1 predicted language,
    or (None, 0.0) for empty/whitespace-only input."""
    text = text.replace("\n", " ").strip()
    if not text:
        return None, 0.0
    model = model if model is not None else load_fasttext_model()
    labels, probs = _safe_fasttext_predict(model, text, k=1)
    if not labels:
        return None, 0.0
    lang = labels[0].replace("__label__", "")
    return lang, float(probs[0])


def bengali_unicode_ratio(text):
    """Fraction of alphabetic (Bengali-block or Latin) characters that are
    in the Bengali Unicode block (U+0980-U+09FF). 0.0 for text with no
    alphabetic characters of either script at all (numbers/punctuation/
    emoji-only input) — not `None`, since callers compare it numerically."""
    bengali_count = len(BENGALI_RANGE.findall(text))
    latin_count = len(LATIN_RANGE.findall(text))
    total = bengali_count + latin_count
    return bengali_count / total if total else 0.0


def detect_and_route(text, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD, model=None) -> dict:
    """The gate. Returns {"language": "bn"|"en"|"unsupported",
    "confidence": float, "fasttext_language": str|None,
    "fasttext_confidence": float, "bengali_unicode_ratio": float}.

    Decision logic:
    1. Empty/whitespace-only input -> unsupported (nothing to detect).
    2. High Bengali-Unicode ratio (>=0.5 of alphabetic chars) -> trust that
       over fastText: real Bengali script is a near-certain signal fastText
       doesn't need to corroborate, and this catches cases where a short
       code-mixed snippet confuses the statistical model.
    3. Otherwise, defer to fastText's top-1 prediction: accept only if it's
       "bn" or "en" *and* meets the confidence threshold; anything else
       (another of the 176 languages, or low confidence even for bn/en) is
       rejected as unsupported rather than guessed.

    Known, disclosed limitation: this cannot reliably distinguish fully
    romanized "Banglish" from genuine English by script alone (same
    limitation `catla.py`'s `looks_fully_romanized_bengali` has, for the
    same reason) — fastText's `lid.176` was trained predominantly on
    native-script text per language, not romanized transliterations, so
    romanized Bengali may be classified as English or as unsupported. Not
    silently claimed to be solved.
    """
    text = text.strip()
    if not text:
        return {"language": "unsupported", "confidence": 0.0, "fasttext_language": None,
                "fasttext_confidence": 0.0, "bengali_unicode_ratio": 0.0, "reason": "empty input"}

    bn_ratio = bengali_unicode_ratio(text)
    ft_lang, ft_conf = fasttext_predict(text, model=model)

    if bn_ratio >= BENGALI_UNICODE_ACCEPT_RATIO:
        return {"language": "bn", "confidence": max(bn_ratio, ft_conf if ft_lang == "bn" else 0.0),
                "fasttext_language": ft_lang, "fasttext_confidence": ft_conf,
                "bengali_unicode_ratio": bn_ratio, "reason": "bengali_unicode_ratio"}

    if ft_lang in SUPPORTED_LANGUAGES and ft_conf >= confidence_threshold:
        return {"language": ft_lang, "confidence": ft_conf, "fasttext_language": ft_lang,
                "fasttext_confidence": ft_conf, "bengali_unicode_ratio": bn_ratio, "reason": "fasttext"}

    return {"language": "unsupported", "confidence": ft_conf, "fasttext_language": ft_lang,
            "fasttext_confidence": ft_conf, "bengali_unicode_ratio": bn_ratio,
            "reason": "unsupported_language_or_low_confidence"}


if __name__ == "__main__":
    import sys
    text = sys.argv[1] if len(sys.argv) > 1 else "আমি ভালো আছি"
    result = detect_and_route(text)
    for k, v in result.items():
        print(f"{k}: {v}")
