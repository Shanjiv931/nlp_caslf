"""Phase 4 — Bengali script <-> romanized Bengali (Banglish) transliterator.

This is a rule-based phonetic transliterator, not a trained model. It is
scoped honestly: Bengali orthography (implicit inherent vowels, conjuncts,
vowel signs/matras) does not map 1:1 onto informal Latin "Banglish" spelling,
which itself varies a lot between individual typists. This module gives a
reasonable, deterministic baseline for both directions using a mapping table
in the same spirit as popular open phonetic input schemes (e.g. Avro
Phonetic) — it will not exactly reproduce any specific person's spelling
habits, and is not evaluated for exact-match accuracy against
data/raw/banglatlit/ (that dataset is sentence-level, not
word-aligned, so it can't directly supervise this rule table).

A proper trained seq2seq transliterator using the 46,864 aligned
BanglaTLit pairs (data/raw/banglatlit/, see logs/phase_3_status.md) is a
legitimate future upgrade path, better suited to Phase 6's fine-tuning
infrastructure than to this lightweight preprocessing module.
"""
import regex

# Bengali -> Latin (bn_to_latin direction): independent vowels
_INDEP_VOWELS = {
    "অ": "o", "আ": "a", "ই": "i", "ঈ": "ii", "উ": "u", "ঊ": "uu",
    "ঋ": "ri", "এ": "e", "ঐ": "oi", "ও": "o", "ঔ": "ou",
}
# dependent vowel signs (matras) — attach to preceding consonant
_MATRAS = {
    "া": "a", "ি": "i", "ী": "ii", "ু": "u", "ূ": "uu", "ৃ": "ri",
    "ে": "e", "ৈ": "oi", "ো": "o", "ৌ": "ou",
}
_CONSONANTS = {
    "ক": "k", "খ": "kh", "গ": "g", "ঘ": "gh", "ঙ": "ng",
    "চ": "ch", "ছ": "chh", "জ": "j", "ঝ": "jh", "ঞ": "n",
    "ট": "t", "ঠ": "th", "ড": "d", "ঢ": "dh", "ণ": "n",
    "ত": "t", "থ": "th", "দ": "d", "ধ": "dh", "ন": "n",
    "প": "p", "ফ": "ph", "ব": "b", "ভ": "bh", "ম": "m",
    "য": "j", "র": "r", "ল": "l", "শ": "sh", "ষ": "sh",
    "স": "s", "হ": "h", "ড়": "r", "ঢ়": "rh", "য়": "y",
    "ৎ": "t",
}
_VIRAMA = "্"
_ANUSVARA = {"ং": "ng", "ঃ": "h", "ঁ": "n"}
_DIGITS = {"０": "0", "১": "1", "২": "2", "৩": "3", "৪": "4", "৫": "5", "৬": "6", "৭": "7", "৮": "8", "৯": "9", "০": "0"}

# Latin -> Bengali (latin_to_bn direction): longest-match-first phonetic rules,
# ordered so multi-character digraphs are tried before single letters.
_LATIN_TO_BN_RULES = [
    ("kh", "খ"), ("gh", "ঘ"), ("ch", "চ"), ("chh", "ছ"), ("jh", "ঝ"),
    ("th", "থ"), ("dh", "ধ"), ("ph", "ফ"), ("bh", "ভ"), ("sh", "শ"), ("ng", "ং"),
    ("rh", "ঢ়"),
    ("k", "ক"), ("g", "গ"), ("c", "চ"), ("j", "জ"), ("t", "ত"), ("d", "দ"),
    ("n", "ন"), ("p", "প"), ("f", "ফ"), ("b", "ব"), ("v", "ভ"), ("m", "ম"),
    ("y", "য়"), ("r", "র"), ("l", "ল"), ("s", "স"), ("h", "হ"),
    ("aa", "া"), ("ii", "ী"), ("uu", "ূ"), ("oi", "ৈ"), ("ou", "ৌ"),
    ("a", "া"), ("i", "ি"), ("u", "ু"), ("e", "ে"), ("o", "ো"),
]
_INITIAL_VOWELS = {"a": "আ", "i": "ই", "u": "উ", "e": "এ", "o": "ও", "aa": "আ", "ii": "ঈ", "uu": "ঊ", "oi": "ঐ", "ou": "ঔ"}


def bn_to_latin(text):
    out = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in _CONSONANTS:
            base = _CONSONANTS[ch]
            nxt = text[i + 1] if i + 1 < n else ""
            if nxt == _VIRAMA:
                out.append(base)
                i += 2
                continue
            if nxt in _MATRAS:
                out.append(base + _MATRAS[nxt])
                i += 2
                continue
            out.append(base + "o")  # inherent vowel
            i += 1
            continue
        if ch in _INDEP_VOWELS:
            out.append(_INDEP_VOWELS[ch])
            i += 1
            continue
        if ch in _ANUSVARA:
            out.append(_ANUSVARA[ch])
            i += 1
            continue
        if ch in _DIGITS:
            out.append(_DIGITS[ch])
            i += 1
            continue
        out.append(ch)  # punctuation/space/unrecognized, pass through
        i += 1
    return "".join(out)


_WORD_SPLIT_RE = regex.compile(r"([A-Za-z]+|[^A-Za-z]+)")


def _latin_word_to_bn(word):
    out = []
    i = 0
    n = len(word)
    lw = word.lower()
    is_word_start = True
    while i < n:
        matched = False
        for pattern, bn in _LATIN_TO_BN_RULES:
            plen = len(pattern)
            if lw[i:i + plen] == pattern:
                if pattern in _INITIAL_VOWELS and is_word_start:
                    out.append(_INITIAL_VOWELS[pattern])
                else:
                    out.append(bn)
                i += plen
                matched = True
                is_word_start = False
                break
        if not matched:
            out.append(word[i])
            i += 1
            is_word_start = False
    return "".join(out)


def latin_to_bn(text):
    parts = _WORD_SPLIT_RE.findall(text)
    out = []
    for part in parts:
        if part and part[0].isalpha() and part.isascii():
            out.append(_latin_word_to_bn(part))
        else:
            out.append(part)
    return "".join(out)


if __name__ == "__main__":
    bn = "আমি ভালো আছি"
    latin = bn_to_latin(bn)
    print("bn->latin:", latin.encode("ascii", "backslashreplace").decode())
    back = latin_to_bn("ami valo achi")
    print("latin->bn (roundtrip demo):", back.encode("ascii", "backslashreplace").decode())
