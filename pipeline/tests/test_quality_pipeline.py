"""Unit tests for pipeline/quality_pipeline.py's orchestration and fallback
decision logic — this is where the PRD's actual quality guarantees live
(never silently ship a drifted translation), so it's tested thoroughly with
mocked stage functions rather than lightly. No torch required: every stage
(ensemble/qe_rerank/llm_postedit/roundtrip_verify) is monkeypatched at the
module level, isolating quality_pipeline.py's own wiring/decision logic
from whether the underlying ML calls work on this machine.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import ensemble
import llm_postedit
import qe_rerank
import roundtrip_verify
from quality_pipeline import produce_translation


def make_candidate(text, engine="nllb", method="beam"):
    return ensemble.Candidate(text=text, engine=engine, method=method, source_text="src", direction="bn2en")


def _patch_happy_path(monkeypatch, *, edit_applied=True, roundtrip_passes=True,
                       pre_edit_roundtrip_passes=None, post_edit_text="I am doing well #blessed"):
    winner = make_candidate("i am good #blessed", engine="nllb", method="beam")

    monkeypatch.setattr(ensemble, "generate_ensemble_candidates",
                         lambda source_text, direction, engines=None: [winner])
    monkeypatch.setattr(qe_rerank, "rerank",
                         lambda source_text, candidates: {"winner": winner, "method": "comet_qe", "scored": [(winner, 0.87)]})

    postedit_result = {
        "pre_edit": winner.text,
        "post_edit": post_edit_text if edit_applied else winner.text,
        "protected_spans": ["#blessed"],
        "protected_tokens_preserved": True,
        "llm_used": "fake/model",
        "edit_applied": edit_applied,
        "fallback_reason": None,
    }
    monkeypatch.setattr(llm_postedit, "postedit", lambda *a, **kw: postedit_result)

    def fake_verify(source_text, candidate_text, direction, threshold=0.85, engine_key="nllb"):
        if candidate_text == postedit_result["post_edit"]:
            passed = roundtrip_passes
        elif candidate_text == postedit_result["pre_edit"]:
            passed = pre_edit_roundtrip_passes if pre_edit_roundtrip_passes is not None else roundtrip_passes
        else:
            passed = False
        return {
            "roundtrip_text": "আমি ভালো আছি" if passed else "completely different",
            "similarity": 0.93 if passed else 0.40,
            "passed": passed,
            "threshold": threshold,
        }

    monkeypatch.setattr(roundtrip_verify, "verify", fake_verify)
    return winner, postedit_result


def test_happy_path_returns_post_edit_translation(monkeypatch):
    _patch_happy_path(monkeypatch, edit_applied=True, roundtrip_passes=True)
    result = produce_translation("আমি ভালো আছি #blessed", "bn2en")

    assert result["translation"] == "I am doing well #blessed"
    assert result["winning_engine"] == "nllb"
    assert result["qe_score"] == 0.87
    assert result["qe_method"] == "comet_qe"
    assert result["low_confidence"] is False
    assert result["error"] is None
    assert result["roundtrip_similarity"] == 0.93


def test_no_candidates_flags_low_confidence(monkeypatch):
    monkeypatch.setattr(ensemble, "generate_ensemble_candidates", lambda source_text, direction, engines=None: [])
    result = produce_translation("some text", "bn2en")

    assert result["translation"] is None
    assert result["low_confidence"] is True
    assert "no candidates" in result["error"]


def test_rerank_no_winner_flags_low_confidence(monkeypatch):
    winner_candidate = make_candidate("x")
    monkeypatch.setattr(ensemble, "generate_ensemble_candidates",
                         lambda source_text, direction, engines=None: [winner_candidate])
    monkeypatch.setattr(qe_rerank, "rerank", lambda source_text, candidates: {"winner": None, "method": "comet_qe", "scored": []})

    result = produce_translation("some text", "bn2en")
    assert result["low_confidence"] is True
    assert result["translation"] is None
    assert "no winner" in result["error"]


def test_roundtrip_fail_falls_back_to_pre_edit_when_it_passes(monkeypatch):
    _patch_happy_path(monkeypatch, edit_applied=True, roundtrip_passes=False, pre_edit_roundtrip_passes=True)
    result = produce_translation("আমি ভালো আছি #blessed", "bn2en")

    assert result["translation"] == "i am good #blessed"  # fell back to pre-edit
    assert result["low_confidence"] is False  # fallback succeeded, not a failure state
    assert "fell back to the pre-edit" in result["verification_note"]


def test_roundtrip_fail_both_candidates_flags_low_confidence_but_still_ships(monkeypatch):
    _patch_happy_path(monkeypatch, edit_applied=True, roundtrip_passes=False, pre_edit_roundtrip_passes=False)
    result = produce_translation("আমি ভালো আছি #blessed", "bn2en")

    assert result["translation"] == "I am doing well #blessed"  # still ships the post-edit
    assert result["low_confidence"] is True
    assert "both" in result["verification_note"]


def test_roundtrip_fail_with_no_edit_applied_flags_low_confidence(monkeypatch):
    # pre_edit == post_edit (no LLM edit was applied), so there's nothing
    # different to fall back to — must not silently retry the same text
    winner = make_candidate("i am good #blessed")
    monkeypatch.setattr(ensemble, "generate_ensemble_candidates",
                         lambda source_text, direction, engines=None: [winner])
    monkeypatch.setattr(qe_rerank, "rerank",
                         lambda source_text, candidates: {"winner": winner, "method": "comet_qe", "scored": [(winner, 0.5)]})
    monkeypatch.setattr(llm_postedit, "postedit", lambda *a, **kw: {
        "pre_edit": winner.text, "post_edit": winner.text, "protected_spans": [],
        "protected_tokens_preserved": True, "llm_used": None, "edit_applied": False,
        "fallback_reason": "LLM unavailable",
    })
    monkeypatch.setattr(roundtrip_verify, "verify", lambda *a, **kw: {
        "roundtrip_text": "drifted", "similarity": 0.3, "passed": False, "threshold": 0.85,
    })

    result = produce_translation("আমি ভালো আছি", "bn2en")
    assert result["low_confidence"] is True
    assert result["translation"] == "i am good #blessed"
    assert "no LLM edit was applied" in result["verification_note"]


def test_all_candidates_reported_for_transparency(monkeypatch):
    c1 = make_candidate("a", engine="nllb", method="beam")
    c2 = make_candidate("b", engine="banglat5", method="sample")
    monkeypatch.setattr(ensemble, "generate_ensemble_candidates",
                         lambda source_text, direction, engines=None: [c1, c2])
    monkeypatch.setattr(qe_rerank, "rerank",
                         lambda source_text, candidates: {"winner": c1, "method": "comet_qe", "scored": [(c1, 0.9), (c2, 0.5)]})
    monkeypatch.setattr(llm_postedit, "postedit", lambda *a, **kw: {
        "pre_edit": c1.text, "post_edit": c1.text, "protected_spans": [],
        "protected_tokens_preserved": True, "llm_used": None, "edit_applied": False, "fallback_reason": None,
    })
    monkeypatch.setattr(roundtrip_verify, "verify", lambda *a, **kw: {
        "roundtrip_text": "x", "similarity": 0.9, "passed": True, "threshold": 0.85,
    })

    result = produce_translation("src", "bn2en")
    assert ("nllb", "beam", "a") in result["all_candidates"]
    assert ("banglat5", "sample", "b") in result["all_candidates"]
