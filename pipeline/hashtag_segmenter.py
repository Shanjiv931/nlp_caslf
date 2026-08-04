"""Phase 4 — hashtag segmenter.

Splits a hashtag ("#IWasHappyBefore", "#সবারখবর", "#duitattoo") into its
constituent words, so the translation pipeline can translate the words
inside a hashtag rather than leaving it as an opaque blob, then rebuild a
valid hashtag in the target language.

Two strategies, tried in order:
1. CamelCase / delimiter split — free, instant, exact whenever the hashtag
   author already used capitalization or separators.
2. Viterbi word segmentation over a frequency dictionary built from our own
   downloaded corpora (Samanantar for Bengali, OPUS-OpenSubtitles for
   English) — the classic approach used by tools like `wordninja`, applied
   here to a Bengali+English bilingual dictionary since hashtags in this
   project's domain mix both. No external segmentation API/package.

The dictionary is built lazily on first use and cached to
data/processed/word_freq.json so repeated runs don't rescan the corpus.
"""
import json
import math
import os
import re
from collections import Counter

PROCESSED = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
UNIFIED_PATH = os.path.join(PROCESSED, "unified.jsonl")
FREQ_CACHE = os.path.join(PROCESSED, "word_freq.json")

CAMEL_RE = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|[০-৯0-9]+|[অ-হড়ঢ়য়ঀ-৿]+")


def _build_word_freq(max_rows=200_000):
    """Scan the unified corpus once to build a word->count dictionary
    covering both English (from parallel_text / source-side English rows)
    and Bengali (from text) tokens."""
    counter = Counter()
    if not os.path.exists(UNIFIED_PATH):
        return counter
    with open(UNIFIED_PATH, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_rows:
                break
            row = json.loads(line)
            for field, lang in [("text", row.get("text_lang")), ("parallel_text", row.get("parallel_lang"))]:
                val = row.get(field)
                if not val:
                    continue
                for w in re.findall(r"[A-Za-z]+|[অ-হড়ঢ়য়ঀ-৿]+", val):
                    if len(w) >= 2:
                        counter[w.lower()] += 1
    return counter


def _load_or_build_freq():
    if os.path.exists(FREQ_CACHE):
        with open(FREQ_CACHE, encoding="utf-8") as f:
            return Counter(json.load(f))
    counter = _build_word_freq()
    if counter:
        os.makedirs(PROCESSED, exist_ok=True)
        with open(FREQ_CACHE, "w", encoding="utf-8") as f:
            json.dump(dict(counter.most_common(200_000)), f, ensure_ascii=False)
    return counter


_FREQ = None
_TOTAL = None


def _freq():
    global _FREQ, _TOTAL
    if _FREQ is None:
        _FREQ = _load_or_build_freq()
        _TOTAL = sum(_FREQ.values()) or 1
    return _FREQ, _TOTAL


def _word_cost(word):
    freq, total = _freq()
    count = freq.get(word.lower(), 0)
    if count == 0:
        # unseen-word penalty scaled by length (shorter unknown words cost less)
        return math.log(total) + len(word) * 2
    return math.log(total / count)


def _camel_split(hashtag):
    parts = CAMEL_RE.findall(hashtag)
    return parts if len(parts) > 1 else None


def _viterbi_split(word, max_word_len=20):
    n = len(word)
    if n == 0:
        return []
    best_cost = [0.0] + [float("inf")] * n
    best_split = [0] * (n + 1)
    for i in range(1, n + 1):
        for j in range(max(0, i - max_word_len), i):
            candidate = word[j:i]
            cost = best_cost[j] + _word_cost(candidate)
            if cost < best_cost[i]:
                best_cost[i] = cost
                best_split[i] = j
    tokens = []
    i = n
    while i > 0:
        j = best_split[i]
        tokens.append(word[j:i])
        i = j
    tokens.reverse()
    return tokens


def segment_hashtag(hashtag):
    """hashtag may include or omit the leading '#'."""
    raw = hashtag[1:] if hashtag.startswith("#") else hashtag
    if not raw:
        return []
    camel = _camel_split(raw)
    if camel:
        return camel
    return _viterbi_split(raw)


if __name__ == "__main__":
    for h in ["#IWasHappyBefore", "#duitattoo", "#mydayout", "#COVID19"]:
        print(h, "->", segment_hashtag(h))
