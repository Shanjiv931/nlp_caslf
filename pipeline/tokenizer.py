"""Phase 4 — tweet-aware tokenizer.

Splits a tweet into tokens while keeping hashtags, mentions, URLs, and emoji
as single atomic tokens (never split mid-token), since these are exactly the
spans the rest of the pipeline needs to protect during translation and
post-editing (Phase 7's "protected tokens" list is built from this).
"""
import regex
import emoji as emoji_lib

URL_RE = regex.compile(r"https?://\S+|www\.\S+")
HASHTAG_RE = regex.compile(r"#\w+")
MENTION_RE = regex.compile(r"@\w+")
WORD_RE = regex.compile(r"\w+", regex.UNICODE)
PUNCT_RE = regex.compile(r"[^\w\s]", regex.UNICODE)

SPECIAL_RE = regex.compile(
    "|".join([URL_RE.pattern, HASHTAG_RE.pattern, MENTION_RE.pattern]),
)


def tokenize(text):
    """Return a list of (token, kind) pairs.

    kind is one of: "url", "hashtag", "mention", "emoji", "word", "punct".
    Emoji are matched greedily against multi-codepoint sequences (e.g. flag
    emoji, ZWJ sequences) before falling back to word/punct splitting.
    """
    tokens = []
    pos = 0
    for m in _iter_matches(text):
        if m["start"] > pos:
            tokens.extend(_split_plain(text[pos:m["start"]]))
        tokens.append((m["text"], m["kind"]))
        pos = m["end"]
    if pos < len(text):
        tokens.extend(_split_plain(text[pos:]))
    return tokens


def _iter_matches(text):
    matches = []
    for m in SPECIAL_RE.finditer(text):
        kind = "url" if m.group().startswith(("http", "www")) else ("hashtag" if m.group().startswith("#") else "mention")
        matches.append({"start": m.start(), "end": m.end(), "text": m.group(), "kind": kind})
    for token_data in emoji_lib.emoji_list(text):
        matches.append({"start": token_data["match_start"], "end": token_data["match_end"], "text": token_data["emoji"], "kind": "emoji"})
    matches.sort(key=lambda m: m["start"])
    # drop overlaps (keep first-seen, i.e. url/hashtag/mention precedence over emoji if they somehow overlap)
    out = []
    last_end = -1
    for m in matches:
        if m["start"] >= last_end:
            out.append(m)
            last_end = m["end"]
    return out


def _split_plain(text):
    out = []
    pos = 0
    for m in regex.finditer(r"\w+|[^\w\s]", text, regex.UNICODE):
        if m.start() > pos:
            pass  # whitespace, dropped
        out.append((m.group(), "word" if WORD_RE.fullmatch(m.group()) else "punct"))
        pos = m.end()
    return out


def protected_spans(text):
    """Return the list of substrings (hashtags, mentions, URLs, emoji) that
    must be preserved verbatim through translation/post-editing."""
    return [t for t, kind in tokenize(text) if kind in ("hashtag", "mention", "url", "emoji")]


if __name__ == "__main__":
    sample = "Just watched #EndGame with @marvel fans 😍🔥 https://t.co/xyz amazing!!"
    for tok, kind in tokenize(sample):
        print(kind, repr(tok).encode("ascii", "backslashreplace").decode())
