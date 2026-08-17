"""Phase 7 — llm_postedit.py: LLM-based Automatic Post-Editing (APE) with
protected-token constraints.

Takes the QE-reranked winning MT candidate and passes it through a free,
open-weight instruction-tuned LLM with a tightly scoped prompt: fix
fluency/naturalness only, preserve meaning exactly, never touch the
protected spans (emoji/hashtags/mentions/URLs — Phase 4's
`tokenizer.protected_spans()`) that must survive verbatim into the final
output. This is the "MT-then-LLM-refine" pattern the PRD calls out
(Section 0, principle 3) as how open-source pipelines close the gap with
proprietary systems.

Model choice, in preference order (first one that loads on the available
hardware wins): Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, Gemma-2-9b-it —
all free, open-weight, instruction-tuned. Loaded 4-bit by default
(bitsandbytes) since running this alongside 3 MT ensemble models on a
single free-tier T4 needs every model to have a light footprint; the PRD's
own risk table calls for loading models sequentially/offloading between
stages for exactly this reason.

This machine cannot run torch (see logs/phase_6_status.md) — the model
loading/generation code is correct-by-construction against the real APIs
but unverified here. The protected-token verification (a pure string
check, no model calls) is what's unit-tested locally, and it matters even
once real hardware is available: it's the first, cheapest line of defense
against the LLM violating its own instructions, before round-trip
verification (a full second translation pass) is even needed.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from tokenizer import protected_spans  # Phase 4

DEFAULT_LLM_CANDIDATES = [
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-2-9b-it",
]

_LLM_CACHE = {}

_TARGET_LANG_NAME = {"bn2en": "English", "en2bn": "Bengali"}

SYSTEM_PROMPT = (
    "You are a careful post-editor for machine-translated social media text. "
    "You improve fluency and naturalness ONLY. You must NEVER change the meaning, "
    "add information, remove information, or alter any of the protected tokens listed "
    "below. Return ONLY the corrected sentence, with no explanation, no quotation marks, "
    "and no extra commentary."
)


def build_postedit_prompt(source_text, mt_candidate, protected, target_lang_name):
    """Pure logic (string formatting only) — testable without a model."""
    protected_list = ", ".join(f'"{p}"' for p in protected) if protected else "(none)"
    user_prompt = (
        f"Original source text:\n{source_text}\n\n"
        f"Machine-translated candidate ({target_lang_name}):\n{mt_candidate}\n\n"
        f"Protected tokens that must appear unchanged in your output, exactly as written "
        f"(emoji, hashtags, mentions, URLs): {protected_list}\n\n"
        f"Improve fluency and naturalness only. Preserve meaning exactly. Do not add or "
        f"remove information. Do not modify the protected tokens listed above. "
        f"Return only the corrected sentence."
    )
    return SYSTEM_PROMPT, user_prompt


def protected_tokens_preserved(edited_text, protected):
    """Pure logic (string containment check, no model calls) — the
    cheapest possible defense against the LLM violating its own
    instructions, checked before round-trip verification (a full second
    translation pass) is even attempted. Returns (bool, list-of-missing)."""
    missing = [p for p in protected if p not in edited_text]
    return len(missing) == 0, missing


def load_llm(model_names=None, use_4bit=True):
    """Loads and caches the first LLM from `model_names` that loads
    successfully on the available hardware, trying each in order. Returns
    (model, tokenizer, model_name_used)."""
    model_names = model_names or DEFAULT_LLM_CANDIDATES
    cache_key = (tuple(model_names), use_4bit)
    if cache_key in _LLM_CACHE:
        return _LLM_CACHE[cache_key]

    from transformers import AutoModelForCausalLM, AutoTokenizer

    quant_kwargs = {}
    if use_4bit:
        from transformers import BitsAndBytesConfig
        quant_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype="bfloat16", bnb_4bit_quant_type="nf4",
        )

    errors = []
    for model_name in model_names:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name, **quant_kwargs)
            model.eval()
            result = (model, tokenizer, model_name)
            _LLM_CACHE[cache_key] = result
            return result
        except Exception as e:
            errors.append(f"{model_name}: {type(e).__name__}: {e}")

    raise RuntimeError(f"no LLM could be loaded from any candidate: {'; '.join(errors)}")


def generate_postedit(model, tokenizer, system_prompt, user_prompt, max_new_tokens=256):
    import torch

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # deterministic — this is editing, not creative generation
            temperature=None,
            top_p=None,
        )
    new_tokens = output_ids[0][input_ids.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def postedit(source_text, mt_candidate, direction, model_names=None, use_4bit=True) -> dict:
    """Top-level entry point. Always returns a usable result — if the LLM's
    edit violates the protected-token contract, falls back to the pre-edit
    MT candidate rather than shipping a broken edit, and says so in the
    returned dict (FR9 in the PRD: the transparency panel must show this,
    not hide it)."""
    protected = protected_spans(source_text)
    target_lang_name = _TARGET_LANG_NAME.get(direction, direction)
    system_prompt, user_prompt = build_postedit_prompt(source_text, mt_candidate, protected, target_lang_name)

    result = {
        "pre_edit": mt_candidate,
        "post_edit": mt_candidate,  # overwritten below on success
        "protected_spans": protected,
        "protected_tokens_preserved": True,
        "llm_used": None,
        "edit_applied": False,
        "fallback_reason": None,
    }

    try:
        model, tokenizer, model_name_used = load_llm(model_names, use_4bit=use_4bit)
        result["llm_used"] = model_name_used
        edited = generate_postedit(model, tokenizer, system_prompt, user_prompt)
    except Exception as e:
        result["fallback_reason"] = f"LLM unavailable ({type(e).__name__}: {e}) — kept pre-edit MT candidate"
        return result

    preserved, missing = protected_tokens_preserved(edited, protected)
    if not preserved:
        result["fallback_reason"] = (
            f"LLM edit dropped/altered protected tokens {missing} — kept pre-edit MT candidate "
            f"rather than ship a broken edit"
        )
        result["protected_tokens_preserved"] = False
        return result

    if not edited:
        result["fallback_reason"] = "LLM returned an empty edit — kept pre-edit MT candidate"
        return result

    result["post_edit"] = edited
    result["edit_applied"] = True
    return result


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "আমি ভালো আছি #blessed"
    mt = sys.argv[2] if len(sys.argv) > 2 else "i am good #blessed"
    direction = sys.argv[3] if len(sys.argv) > 3 else "bn2en"
    result = postedit(src, mt, direction)
    for k, v in result.items():
        print(f"{k}: {v}")
