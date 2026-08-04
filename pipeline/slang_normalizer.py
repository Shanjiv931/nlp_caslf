"""Phase 4 — Bengali-English social-media slang/abbreviation normalizer.

A hand-curated dictionary of common Banglish/internet slang -> standard-form
mapping. This is intentionally a modest, honestly-scoped starting set (not
claimed as exhaustive) covering the most frequent social-media
abbreviations and Banglish spellings observed in the datasets pulled in
Phase 1 (BnSentMix, BanTH, SentMix-3L, EmoMix-3L). Matching is
case-insensitive and whole-word only, so it never mangles substrings inside
longer words.
"""
import regex

# Banglish (romanized Bengali) -> standard Bengali-script gloss, kept as the
# widely-understood Latin form here since downstream translation models
# consume Latin-script Banglish directly; the point of normalization is
# spelling variants -> one canonical spelling, not full transliteration
# (that's transliterator.py's job).
BANGLISH_SLANG = {
    "valo": "bhalo",
    "balo": "bhalo",
    "kmn": "kemon",
    "kemne": "kemon kore",
    "kivabe": "kemon kore",
    "ki kori": "ki kori",
    "koro": "koro",
    "korchi": "korchi",
    "korchis": "korchis",
    "thik ache": "thik ache",
    "thik asi": "thik ache",
    "achi": "achi",
    "asi": "achi",
    "asos": "acho",
    "acho": "acho",
    "dhonnobad": "dhonnobad",
    "dhonnobaad": "dhonnobad",
    "onk": "onek",
    "onek": "onek",
    "koto": "koto",
    "ktoi": "koto",
    "bhai": "bhai",
    "vai": "bhai",
    "apu": "apu",
    "cutii": "cute",
}

# English internet abbreviations -> expanded form
ENGLISH_SLANG = {
    "u": "you",
    "r": "are",
    "ur": "your",
    "pls": "please",
    "plz": "please",
    "thx": "thanks",
    "ty": "thank you",
    "gr8": "great",
    "b4": "before",
    "2day": "today",
    "tmrw": "tomorrow",
    "idk": "i don't know",
    "imo": "in my opinion",
    "btw": "by the way",
    "omg": "oh my god",
    "lol": "laughing out loud",
    "brb": "be right back",
    "asap": "as soon as possible",
    "lmk": "let me know",
    "tbh": "to be honest",
}

_COMBINED = {**BANGLISH_SLANG, **ENGLISH_SLANG}
_WORD_RE = regex.compile(r"\b\w+\b", regex.UNICODE)


def normalize_slang(text, extra_dict=None):
    """Replace known slang/abbreviation tokens with their normalized form,
    preserving original casing style (Titlecase in -> Titlecase out) and
    leaving unknown words untouched."""
    lookup = _COMBINED if extra_dict is None else {**_COMBINED, **extra_dict}

    def repl(m):
        word = m.group()
        key = word.lower()
        if key not in lookup:
            return word
        replacement = lookup[key]
        if word.isupper():
            return replacement.upper()
        if word[0].isupper():
            return replacement[0].upper() + replacement[1:]
        return replacement

    return _WORD_RE.sub(repl, text)


if __name__ == "__main__":
    print(normalize_slang("u r gr8, thx a lot! Vai kmn asos?"))
