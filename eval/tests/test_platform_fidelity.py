"""Unit tests for eval/platform_fidelity.py — pure text logic, no models
needed, fully testable locally."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from platform_fidelity import score_dataset, score_example


def test_full_preservation_scores_1():
    scores = score_example(
        "Just watched #EndGame with @marvel fans 😍",
        "sobe #EndGame dekhlam @marvel fan der sathe 😍",
    )
    assert scores["emoji_preservation"] == 1.0
    assert scores["verbatim_preservation"] == 1.0
    assert scores["hashtag_count_preservation"] == 1.0
    assert scores["overall"] == 1.0


def test_dropped_emoji_scores_zero_for_that_dimension():
    scores = score_example("great news 😍🔥", "darun khobor")
    assert scores["emoji_preservation"] == 0.0


def test_partial_emoji_preservation():
    scores = score_example("great news 😍🔥", "darun khobor 😍")
    assert scores["emoji_preservation"] == 0.5


def test_mention_translated_incorrectly_scores_zero():
    # a mention/URL must NEVER change — if it does, that's a hard failure
    scores = score_example("hi @realuser", "হাই @ভুয়া_ব্যবহারকারী")
    assert scores["verbatim_preservation"] == 0.0


def test_hashtag_dropped_entirely():
    scores = score_example("check #deal #sale now", "ekhon dekho")
    assert scores["hashtag_count_preservation"] == 0.0


def test_hashtag_count_preserved_even_if_content_translated():
    # count-based, not content-based — disclosed limitation, tested explicitly
    scores = score_example("check #summersale", "dekho #greeshmokaalinbikroy")
    assert scores["hashtag_count_preservation"] == 1.0


def test_no_social_furniture_all_none():
    scores = score_example("hello there", "ওহে সেখানে")
    assert scores["emoji_preservation"] is None
    assert scores["verbatim_preservation"] is None
    assert scores["hashtag_count_preservation"] is None
    assert scores["overall"] is None


def test_overall_averages_only_applicable_dimensions():
    # emoji present+preserved (1.0), no mentions/hashtags at all (None, None)
    # -> overall should be 1.0, not penalized by the inapplicable dims
    scores = score_example("great 😍", "darun 😍")
    assert scores["overall"] == 1.0


def test_extra_hashtag_in_output_capped_at_1():
    # gaining a hashtag that wasn't in the source shouldn't score >1.0
    scores = score_example("check #deal", "dekho #deal #extra")
    assert scores["hashtag_count_preservation"] == 1.0


def test_score_dataset_aggregates_correctly():
    pairs = [
        {"source_text": "great 😍", "translated_text": "darun 😍"},  # overall 1.0
        {"source_text": "great 😍", "translated_text": "darun"},     # overall 0.0
        {"source_text": "no furniture", "translated_text": "kichu nei"},  # None, excluded
    ]
    result = score_dataset(pairs)
    assert result["overall"]["mean"] == 0.5
    assert result["overall"]["n_applicable"] == 2
    assert result["overall"]["n_total"] == 3


def test_score_dataset_empty_list():
    result = score_dataset([])
    assert result["overall"]["mean"] is None
    assert result["overall"]["n_total"] == 0
