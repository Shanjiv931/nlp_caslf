"""Phase 8 — Platform-Fidelity Score (PFS): a custom composite metric
measuring how well emoji, hashtags, and mentions/URLs survive translation
— the PRD's own metric (not a standard NLP benchmark), since standard MT
metrics (BLEU/chrF++/TER/COMET) don't capture whether a tweet's social-media
furniture survived intact.

Honest scope, stated up front rather than overstated: this scores three
objectively measurable sub-dimensions automatically. It does NOT
automatically score "slang localized to a natural target-register
equivalent" (the PRD's third stated dimension) — there's no reference-free
automatic way to judge whether a slang translation is natural without
either a gold reference or human judgment, and fabricating a proxy score
for it would misrepresent what's actually being measured. Slang quality is
left for human/qualitative review (see eval/RESULTS.md's example section),
and PFS here is computed from the three sub-scores that genuinely are
just string comparisons:

1. emoji_preservation   — fraction of the source's emoji that appear
                           somewhere in the translated output (set-based,
                           order-agnostic — a translation naturally
                           reorders words, emoji included).
2. verbatim_preservation — fraction of the source's mentions (@user) and
                           URLs that appear byte-for-byte unchanged in the
                           output. These must NEVER be translated/altered
                           (a username or link isn't language-dependent),
                           so this is an exact-match check, not fuzzy.
3. hashtag_count_preservation — whether the same *number* of hashtags
                           survived. Doesn't verify a hashtag was
                           translated *correctly* (no ground truth exists
                           for that without per-hashtag human review) —
                           only that hashtags weren't silently dropped or
                           duplicated. A weaker guarantee than the other
                           two, disclosed as such.

Each example's score only averages over the sub-dimensions that are
actually present in that example's source text (an example with no emoji
contributes nothing to the emoji sub-score's denominator) — an example
with nothing to preserve shouldn't be penalized or artificially boosted.
"""
import os
import sys
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline"))
from tokenizer import tokenize  # Phase 4


def _extract(text, kind):
    return [t for t, k in tokenize(text) if k == kind]


def score_example(source_text: str, translated_text: str) -> dict:
    """Returns per-example sub-scores (each None if that sub-dimension
    doesn't apply — the source had nothing of that kind) plus an overall
    score averaged only over the applicable sub-dimensions."""
    src_emoji = _extract(source_text, "emoji")
    src_mentions = _extract(source_text, "mention")
    src_urls = _extract(source_text, "url")
    src_hashtags = _extract(source_text, "hashtag")

    out_emoji = set(_extract(translated_text, "emoji"))
    out_mentions = set(_extract(translated_text, "mention"))
    out_urls = set(_extract(translated_text, "url"))
    out_hashtags = _extract(translated_text, "hashtag")

    scores = {}

    if src_emoji:
        scores["emoji_preservation"] = len(set(src_emoji) & out_emoji) / len(set(src_emoji))
    else:
        scores["emoji_preservation"] = None

    verbatim_src = set(src_mentions) | set(src_urls)
    if verbatim_src:
        verbatim_out = out_mentions | out_urls
        scores["verbatim_preservation"] = len(verbatim_src & verbatim_out) / len(verbatim_src)
    else:
        scores["verbatim_preservation"] = None

    if src_hashtags:
        scores["hashtag_count_preservation"] = min(1.0, len(out_hashtags) / len(src_hashtags))
    else:
        scores["hashtag_count_preservation"] = None

    applicable = [v for v in scores.values() if v is not None]
    scores["overall"] = sum(applicable) / len(applicable) if applicable else None
    return scores


def score_dataset(pairs: List[dict]) -> dict:
    """pairs: list of {"source_text": str, "translated_text": str}.
    Returns aggregate PFS (mean of per-example overall scores, only over
    examples that had at least one applicable sub-dimension) plus per-
    sub-dimension aggregates and coverage counts (how many examples each
    sub-dimension actually applied to — a PFS computed from mostly-empty
    examples is much less meaningful than the raw number alone suggests)."""
    per_example = [score_example(p["source_text"], p["translated_text"]) for p in pairs]

    def aggregate(key):
        values = [e[key] for e in per_example if e[key] is not None]
        return {
            "mean": sum(values) / len(values) if values else None,
            "n_applicable": len(values),
            "n_total": len(per_example),
        }

    return {
        "overall": aggregate("overall"),
        "emoji_preservation": aggregate("emoji_preservation"),
        "verbatim_preservation": aggregate("verbatim_preservation"),
        "hashtag_count_preservation": aggregate("hashtag_count_preservation"),
        "per_example": per_example,
    }


if __name__ == "__main__":
    demo_pairs = [
        {"source_text": "Just watched #EndGame with @marvel fans 😍", "translated_text": "sobe #EndGame dekhlam @marvel fan der sathe 😍"},
        {"source_text": "no social furniture here at all", "translated_text": "kono kichu nei ekhane"},
    ]
    result = score_dataset(demo_pairs)
    for key in ["overall", "emoji_preservation", "verbatim_preservation", "hashtag_count_preservation"]:
        agg = result[key]
        print(f"{key}: mean={agg['mean']} (n={agg['n_applicable']}/{agg['n_total']})")
