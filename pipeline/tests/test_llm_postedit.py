"""Unit tests for pipeline/llm_postedit.py's pure-logic functions (prompt
building, protected-token verification). Actual LLM loading/generation
needs a working torch install this machine doesn't have — see
logs/phase_6_status.md — so those paths are correct-by-construction but
unverified here, not tested.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llm_postedit import build_postedit_prompt, postedit, protected_tokens_preserved


# ---- build_postedit_prompt ----

def test_prompt_includes_source_and_candidate():
    system, user = build_postedit_prompt("আমি ভালো আছি", "i am good", ["#blessed"], "English")
    assert "আমি ভালো আছি" in user
    assert "i am good" in user
    assert "#blessed" in user
    assert "English" in user


def test_prompt_lists_none_when_no_protected_tokens():
    system, user = build_postedit_prompt("src", "mt", [], "Bengali")
    assert "(none)" in user


def test_prompt_system_message_forbids_meaning_changes():
    system, _ = build_postedit_prompt("src", "mt", [], "English")
    assert "NEVER change the meaning" in system


# ---- protected_tokens_preserved ----

def test_protected_tokens_preserved_all_present():
    preserved, missing = protected_tokens_preserved("hello #blessed @user 😍", ["#blessed", "@user", "😍"])
    assert preserved is True
    assert missing == []


def test_protected_tokens_preserved_detects_dropped_token():
    preserved, missing = protected_tokens_preserved("hello world", ["#blessed", "@user"])
    assert preserved is False
    assert set(missing) == {"#blessed", "@user"}


def test_protected_tokens_preserved_partial_drop():
    preserved, missing = protected_tokens_preserved("hello #blessed", ["#blessed", "@user"])
    assert preserved is False
    assert missing == ["@user"]


def test_protected_tokens_preserved_empty_list_always_true():
    preserved, missing = protected_tokens_preserved("anything at all", [])
    assert preserved is True
    assert missing == []


# ---- postedit() fallback behavior (mocking load_llm to avoid needing torch) ----

def test_postedit_falls_back_when_llm_unavailable(monkeypatch):
    import llm_postedit

    def broken_load_llm(*args, **kwargs):
        raise RuntimeError("no LLM could be loaded: simulated failure")

    monkeypatch.setattr(llm_postedit, "load_llm", broken_load_llm)
    result = postedit("আমি ভালো আছি", "i am good", "bn2en")
    assert result["edit_applied"] is False
    assert result["post_edit"] == "i am good"  # unchanged, kept pre-edit candidate
    assert "unavailable" in result["fallback_reason"]


def test_postedit_falls_back_when_llm_drops_protected_token(monkeypatch):
    import llm_postedit

    fake_model, fake_tokenizer = object(), object()

    def fake_load_llm(*args, **kwargs):
        return fake_model, fake_tokenizer, "fake/model"

    def fake_generate_postedit(model, tokenizer, system_prompt, user_prompt, max_new_tokens=256):
        return "i am good"  # dropped the #blessed hashtag from the source

    monkeypatch.setattr(llm_postedit, "load_llm", fake_load_llm)
    monkeypatch.setattr(llm_postedit, "generate_postedit", fake_generate_postedit)

    result = postedit("আমি ভালো আছি #blessed", "i am good #blessed", "bn2en")
    assert result["edit_applied"] is False
    assert result["protected_tokens_preserved"] is False
    assert result["post_edit"] == "i am good #blessed"  # fell back to pre-edit
    assert "#blessed" in result["fallback_reason"]


def test_postedit_applies_edit_when_valid(monkeypatch):
    import llm_postedit

    def fake_load_llm(*args, **kwargs):
        return object(), object(), "fake/model"

    def fake_generate_postedit(model, tokenizer, system_prompt, user_prompt, max_new_tokens=256):
        return "I am doing well #blessed"

    monkeypatch.setattr(llm_postedit, "load_llm", fake_load_llm)
    monkeypatch.setattr(llm_postedit, "generate_postedit", fake_generate_postedit)

    result = postedit("আমি ভালো আছি #blessed", "i am good #blessed", "bn2en")
    assert result["edit_applied"] is True
    assert result["post_edit"] == "I am doing well #blessed"
    assert result["fallback_reason"] is None
