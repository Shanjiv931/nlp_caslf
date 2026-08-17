"""Unit tests for pipeline/qe_rerank.py's pure-logic functions (pick_best,
score_candidates_fallback). Actual COMET-QE model loading/scoring needs a
working torch install this machine doesn't have — see
logs/phase_6_status.md — so those paths are correct-by-construction but
unverified here, not tested.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ensemble import Candidate
from qe_rerank import pick_best, score_candidates_fallback


def make_candidate(text, engine="nllb", method="beam", raw_score=None):
    return Candidate(text=text, engine=engine, method=method, source_text="src", direction="bn2en", raw_score=raw_score)


def test_pick_best_prefers_higher_score():
    c1 = make_candidate("bad translation")
    c2 = make_candidate("good translation")
    winner = pick_best([(c1, 0.2), (c2, 0.8)])
    assert winner is c2


def test_pick_best_none_scores_sort_last():
    scored_candidate = make_candidate("has a real score")
    unscored_candidate = make_candidate("no score at all")
    # even a low real score beats "unscored" — None must never be treated as 0.0
    winner = pick_best([(unscored_candidate, None), (scored_candidate, -5.0)])
    assert winner is scored_candidate


def test_pick_best_empty_list_returns_none():
    assert pick_best([]) is None


def test_pick_best_all_none_scores_returns_first_without_crashing():
    c1 = make_candidate("a")
    c2 = make_candidate("b")
    winner = pick_best([(c1, None), (c2, None)])
    assert winner in (c1, c2)  # doesn't crash, returns *something* rather than None


def test_score_candidates_fallback_uses_raw_score_from_beam_search():
    beam_candidate = make_candidate("beam output", method="beam", raw_score=-1.5)
    sample_candidate = make_candidate("sampled output", method="sample", raw_score=None)
    scored = score_candidates_fallback([beam_candidate, sample_candidate])
    assert scored == [(beam_candidate, -1.5), (sample_candidate, None)]


def test_fallback_then_pick_best_prefers_scored_beam_over_unscored_sample():
    beam_candidate = make_candidate("beam output", method="beam", raw_score=-2.0)
    sample_candidate = make_candidate("sampled output", method="sample", raw_score=None)
    scored = score_candidates_fallback([sample_candidate, beam_candidate])
    winner = pick_best(scored)
    assert winner is beam_candidate
