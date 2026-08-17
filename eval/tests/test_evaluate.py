"""Unit tests for eval/evaluate.py's model-independent functions.
`compute_metrics` genuinely runs (sacrebleu is pure Python, no torch) and
is tested for real, not just structurally. `load_test_pairs` is validated
against the actual data/processed/test.jsonl produced by Phase 5, when
present, so this test also catches drift if that file's schema ever
changes. The three `translate_*` functions and `compute_comet` need real
models this machine can't run — not tested here, see logs/phase_8_status.md.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "pipeline"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluate import compute_metrics, load_test_pairs

TEST_JSONL = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "test.jsonl")


# ---- compute_metrics: actually runs, real sacrebleu ----

def test_compute_metrics_perfect_match_scores_high():
    hyps = ["I am doing well today", "The weather is nice"]
    refs = ["I am doing well today", "The weather is nice"]
    metrics = compute_metrics(hyps, refs)
    assert metrics["bleu"] == pytest.approx(100.0)  # sacrebleu's smoothing can land at 100.00000000000004
    assert metrics["ter"] == pytest.approx(0.0)


def test_compute_metrics_completely_wrong_scores_low():
    hyps = ["completely unrelated text here"]
    refs = ["I am doing well today"]
    metrics = compute_metrics(hyps, refs)
    assert metrics["bleu"] < 20.0


def test_compute_metrics_returns_all_three_keys():
    metrics = compute_metrics(["a b c"], ["a b d"])
    assert set(metrics.keys()) == {"bleu", "chrf++", "ter"}
    assert all(isinstance(v, float) for v in metrics.values())


def test_compute_metrics_partial_match_between_perfect_and_wrong():
    perfect = compute_metrics(["I am doing well today"], ["I am doing well today"])
    partial = compute_metrics(["I am doing okay today"], ["I am doing well today"])
    wrong = compute_metrics(["nothing at all similar"], ["I am doing well today"])
    assert perfect["bleu"] > partial["bleu"] > wrong["bleu"]


# ---- load_test_pairs: validated against the real Phase 5 output when present ----

def test_load_test_pairs_bn2en_against_real_file():
    if not os.path.exists(TEST_JSONL):
        return  # this machine's data/processed/ may not have it — not a failure, just untestable here
    pairs = load_test_pairs(TEST_JSONL, "bn2en", max_rows=50)
    assert len(pairs) == 50
    for p in pairs:
        assert p["source"] and p["reference"]
        assert isinstance(p["source"], str) and isinstance(p["reference"], str)


def test_load_test_pairs_directions_give_different_source_language():
    if not os.path.exists(TEST_JSONL):
        return
    bn2en = load_test_pairs(TEST_JSONL, "bn2en", max_rows=20)
    en2bn = load_test_pairs(TEST_JSONL, "en2bn", max_rows=20)
    # bn2en's sources should be predominantly Bengali-script; en2bn's predominantly not
    import re
    bengali_re = re.compile(r"[ঀ-৿]")
    bn2en_bengali_ratio = sum(1 for p in bn2en if bengali_re.search(p["source"])) / len(bn2en)
    en2bn_bengali_ratio = sum(1 for p in en2bn if bengali_re.search(p["source"])) / len(en2bn)
    assert bn2en_bengali_ratio > en2bn_bengali_ratio


def test_load_test_pairs_sanitizes_special_token_literals():
    if not os.path.exists(TEST_JSONL):
        return
    pairs = load_test_pairs(TEST_JSONL, "bn2en", max_rows=None)
    assert not any("<unk>" in p["source"] or "<unk>" in p["reference"] for p in pairs)
