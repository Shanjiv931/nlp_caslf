"""Unit tests for pipeline/roundtrip_verify.py's pure-logic functions
(direction reversal, threshold decision) plus the verify() orchestration
with mocked embedding/generation calls. Actual LaBSE encoding and
round-trip generation need a working torch install this machine doesn't
have — see logs/phase_6_status.md — so those paths are correct-by-
construction but unverified here, not tested.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from roundtrip_verify import reverse_direction, verify


# ---- reverse_direction ----

def test_reverse_bn2en_gives_en2bn():
    assert reverse_direction("bn2en") == "en2bn"


def test_reverse_en2bn_gives_bn2en():
    assert reverse_direction("en2bn") == "bn2en"


def test_reverse_is_involutive():
    for direction in ("bn2en", "en2bn"):
        assert reverse_direction(reverse_direction(direction)) == direction


# ---- verify() orchestration, mocked ----

def test_verify_passes_when_similarity_above_threshold(monkeypatch):
    import roundtrip_verify

    monkeypatch.setattr(roundtrip_verify, "round_trip_translate", lambda text, direction, engine_key="nllb": "আমি ভালো আছি")
    monkeypatch.setattr(roundtrip_verify, "load_embedder", lambda model_name=roundtrip_verify.DEFAULT_LABSE_MODEL: object())
    monkeypatch.setattr(roundtrip_verify, "compute_similarity", lambda embedder, a, b: 0.93)

    result = verify("আমি ভালো আছি", "I am doing well", "bn2en")
    assert result["passed"] is True
    assert result["similarity"] == 0.93
    assert result["roundtrip_text"] == "আমি ভালো আছি"


def test_verify_fails_when_similarity_below_threshold(monkeypatch):
    import roundtrip_verify

    monkeypatch.setattr(roundtrip_verify, "round_trip_translate", lambda text, direction, engine_key="nllb": "completely different meaning")
    monkeypatch.setattr(roundtrip_verify, "load_embedder", lambda model_name=roundtrip_verify.DEFAULT_LABSE_MODEL: object())
    monkeypatch.setattr(roundtrip_verify, "compute_similarity", lambda embedder, a, b: 0.42)

    result = verify("আমি ভালো আছি", "some drifted translation", "bn2en")
    assert result["passed"] is False
    assert result["similarity"] == 0.42


def test_verify_respects_custom_threshold(monkeypatch):
    import roundtrip_verify

    monkeypatch.setattr(roundtrip_verify, "round_trip_translate", lambda text, direction, engine_key="nllb": "x")
    monkeypatch.setattr(roundtrip_verify, "load_embedder", lambda model_name=roundtrip_verify.DEFAULT_LABSE_MODEL: object())
    monkeypatch.setattr(roundtrip_verify, "compute_similarity", lambda embedder, a, b: 0.80)

    # 0.80 fails the default 0.85 threshold but passes a relaxed 0.75 one
    assert verify("src", "candidate", "bn2en")["passed"] is False
    assert verify("src", "candidate", "bn2en", threshold=0.75)["passed"] is True
