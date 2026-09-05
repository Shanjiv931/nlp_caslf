"""Shared MT utilities used by both training (train.py) and inference
(Phase 7's ensemble.py and friends). Factored out of train.py on
2026-08-16 so training-time and inference-time preprocessing can never
drift apart — every environment/compatibility fix discovered while
building Phase 6 (the transformers.onnx shim, the tie_weights() patch, the
IndicTrans2 language-tag handling, the <unk>-literal sanitization) applies
identically whichever script calls it, from one source of truth.

Engine/model naming mirrors notebooks/train_kaggle.ipynb and
train_colab.ipynb's `ENGINES` dict exactly — if that dict changes, this
module's ENGINE_MODEL_NAMES must be updated to match, or training and
inference will silently use different base checkpoints.
"""
import socket
import sys
import types

# ---------------------------------------------------------------------------
# Global network timeout — applied at this module's own import time (not
# deferred into a function) since virtually every pipeline module
# (ensemble.py, quality_pipeline.py, llm_postedit.py, qe_rerank.py,
# roundtrip_verify.py) either imports this module directly or is imported
# alongside something that does, making this the single earliest common
# point to guarantee it's set before any model-loading code runs.
#
# Confirmed for real, 2026-09-05, three separate times in one Kaggle
# session: a huggingface_hub download (CometKiwi's wmt23-cometkiwi-da-xl
# checkpoint, then separately Qwen2.5-7B-Instruct for llm_postedit.py) sat
# completely silent -- 0% GPU, ~1% CPU, zero throughput -- for 4+ hours
# with no exception ever raised, because neither huggingface_hub's download
# stream nor sentence-transformers' model download sets an explicit
# per-request timeout, so a stalled TCP connection just blocks forever.
# socket.setdefaulttimeout() sets the default for any socket that doesn't
# specify its own timeout, which covers this exact gap; 120s tolerates a
# slow-but-actually-progressing download (the timeout resets on every
# successful recv, so a multi-hour download that keeps trickling bytes
# never trips it) while still turning a genuine full stall into a
# TimeoutError within 2 minutes instead of hanging indefinitely -- which
# matters even with a per-example try/except elsewhere (see
# eval/evaluate.py's run_mode()), since a try/except can only catch an
# exception that actually gets raised, not a call that never returns at
# all.
socket.setdefaulttimeout(120)

# ---------------------------------------------------------------------------
# Engine / adapter naming — single source of truth for which HF Hub repo
# each engine+direction actually is, mirroring the notebooks' ENGINES dict.
# ---------------------------------------------------------------------------

ENGINE_MODEL_NAMES = {
    "indictrans2": {
        "bn2en": "ai4bharat/indictrans2-indic-en-1B",
        "en2bn": "ai4bharat/indictrans2-en-indic-1B",
    },
    "nllb": "facebook/nllb-200-distilled-600M",
    "banglat5": "csebuetnlp/banglat5",
}

DEFAULT_HF_ADAPTER_PREFIX = "shanjivkr/catla"  # this project's actual Phase 6 output namespace


def base_model_name(engine_key, direction):
    entry = ENGINE_MODEL_NAMES[engine_key]
    return entry[direction] if isinstance(entry, dict) else entry


def adapter_repo_id(engine_key, direction, prefix=DEFAULT_HF_ADAPTER_PREFIX):
    return f"{prefix}-{engine_key}-{direction}"


# ---------------------------------------------------------------------------
# IndicTransToolkit — optional AI4Bharat preprocessing (script normalization,
# sentence splitting, NER/number masking). See logs/phase_6_status.md for
# the full investigation into why this can fail to install/import.
# ---------------------------------------------------------------------------

def ensure_indictranstoolkit_import_compat():
    """IndicTransToolkit==1.1.1's own collator.py (bundled inside the
    published wheel, confirmed by downloading and extracting
    indictranstoolkit-1.1.1-cp312-cp312-manylinux_2_17_x86_64...whl — the
    exact wheel a Linux/Python-3.12 environment like Kaggle's actually
    installs) does `from transformers.tokenization_utils import
    PreTrainedTokenizerBase` inside its `IndicDataCollator` class body, so
    it runs the moment that class is defined — i.e. at
    IndicTransToolkit/__init__.py's own import time, before this project
    ever reaches the one class it actually uses (`IndicProcessor`).
    `PreTrainedTokenizerBase` genuinely lives in
    `transformers.tokenization_utils_base`, not `transformers.
    tokenization_utils`, against whatever transformers release is
    actually installed (confirmed directly: `hasattr(transformers.
    tokenization_utils_base, "PreTrainedTokenizerBase")` is True,
    `hasattr(transformers.tokenization_utils, "PreTrainedTokenizerBase")`
    is False, on transformers 5.14.1). This is a real bug in
    IndicTransToolkit's own code, unrelated to the indic-nlp-library /
    indic-nlp-library-itt namespace collision already fixed in
    requirements.txt — that fix is necessary but not sufficient on its
    own, since this is a second, independent break in the same package.
    Hit for real on Kaggle, 2026-08-22, training IndicTrans2 on the full
    dataset: `ImportError: cannot import name 'PreTrainedTokenizerBase'
    from 'transformers.tokenization_utils'`.

    Fixed by aliasing the real class into the module IndicTransToolkit's
    collator.py actually imports from, before that import ever runs —
    same pattern as ensure_transformers_onnx_shim below. Must run at this
    module's own import time (not deferred into ensure_all_compat_shims,
    which callers only invoke later): the IndicTransToolkit import attempt
    immediately below happens the moment mt_compat.py itself is first
    imported by anything.
    """
    try:
        import transformers.tokenization_utils as _tu
        if not hasattr(_tu, "PreTrainedTokenizerBase"):
            from transformers.tokenization_utils_base import PreTrainedTokenizerBase as _PTB
            _tu.PreTrainedTokenizerBase = _PTB
    except Exception:
        pass  # if this itself fails, the real IndicTransToolkit import below
        # surfaces its own real error, exactly as it did before this fix
        # existed — never silently swallowed.


ensure_indictranstoolkit_import_compat()

try:
    from IndicTransToolkit.processor import IndicProcessor
    HAS_INDIC_TOOLKIT = True
    INDIC_TOOLKIT_IMPORT_ERROR = None
except Exception as _e:
    # Deliberately `except Exception`, not just `ImportError` — see
    # ensure_transformers_onnx_shim's docstring below for why a compiled
    # Cython extension's import failure can surface as other exception
    # types too. Capture and expose the real reason rather than discarding it.
    IndicProcessor = None
    HAS_INDIC_TOOLKIT = False
    INDIC_TOOLKIT_IMPORT_ERROR = f"{type(_e).__name__}: {_e}"


def get_indic_processor(inference: bool):
    """Returns an IndicProcessor instance if IndicTransToolkit is available,
    else None. `inference=True` for generation/scoring, `False` for
    training-time preprocessing (matches IndicProcessor's own constructor
    convention)."""
    if not HAS_INDIC_TOOLKIT:
        return None
    return IndicProcessor(inference=inference)


# ---------------------------------------------------------------------------
# Language-tagging — the single source of truth for src/tgt language codes
# per direction, and IndicTrans2's minimum-required manual tag-prefix
# fallback when IndicTransToolkit isn't available.
# ---------------------------------------------------------------------------

def direction_lang_codes(direction):
    """Which language is source vs. target for a given translation
    direction. Must agree exactly with how callers assign source/target
    text, or every downstream language-tagging step (IndicTrans2's required
    tag prefix, NLLB's tokenizer.src_lang/tgt_lang) silently uses the wrong
    language — a real bug hit during Phase 6 (src/tgt were swapped in an
    earlier version of this logic, caught only when it crashed IndicTrans2's
    tokenizer outright)."""
    if direction == "bn2en":
        return "ben_Beng", "eng_Latn"
    return "eng_Latn", "ben_Beng"


def tag_indictrans2_text(texts, src_lang, tgt_lang):
    """IndicTransTokenizer._src_tokenize() requires every input to already
    start with "{src_lang} {tgt_lang} " — normally IndicProcessor's job,
    but the minimum required tagging when it's unavailable (see
    logs/phase_6_status.md: `AssertionError: Invalid source language tag`
    without this). Accepts either a single string or a list of strings."""
    if isinstance(texts, str):
        return f"{src_lang} {tgt_lang} {texts}"
    return [f"{src_lang} {tgt_lang} {t}" for t in texts]


# ---------------------------------------------------------------------------
# Text sanitization
# ---------------------------------------------------------------------------

_SPECIAL_TOKEN_LITERALS = ["<unk>", "<pad>", "<s>", "</s>"]


_MOJIBAKE_RANGE_LO = chr(0x80)
_MOJIBAKE_RANGE_HI = chr(0xFF)


def mojibake_fraction(text):
    """Fraction of characters in `text` that fall in the Latin-1 Supplement
    Unicode block (U+0080-U+00FF).

    Found directly responsible (2026-09-04) for a genuine, precision-
    independent training crash: a row's "Bengali" text field was actually
    UTF-8 bytes that had been decoded as Latin-1/cp1252 and re-encoded --
    classic double-encoding mojibake (e.g. Bengali 'আমি' becomes
    'à¦†à¦®à¦¿'). Confirmed via torch.autograd.set_detect_anomaly that training
    directly on this exact row produces NaN in cross-entropy's backward --
    under fp16 GradScaler silently skipped the corrupted step (masking the
    problem, which is why this went undetected for so long), but under fp32
    (no skip mechanism) the same batch corrupted the model outright on the
    very first step.

    This is a much more targeted signal than "how much of the text is in
    the Bengali Unicode block" (which was tried first and rejected): real
    Bengali social-media text legitimately includes large amounts of
    Latin-script content -- English loanwords, and especially "Banglish"
    (Bengali words spelled out phonetically in plain ASCII, very common
    informal usage) -- and a script-fraction filter would have discarded
    thousands of those legitimate rows. Mojibake specifically lands in the
    Latin-1 SUPPLEMENT block (accented/high-byte Latin-1 characters like
    the 'Ã', '¦', '¯' family), which plain ASCII Banglish never touches at
    all -- directly checking train.jsonl found this split is sharply
    bimodal: 99.8% of rows sit at exactly 0%, and everything above ~20% is
    unambiguous garbage (spot-checked directly, zero false positives found),
    with no meaningful middle ground to tune a threshold against.
    """
    if not text:
        return 0.0
    n = sum(1 for c in text if _MOJIBAKE_RANGE_LO <= c <= _MOJIBAKE_RANGE_HI)
    return n / len(text)


def sanitize_text(text):
    """Strip literal occurrences of common tokenizer special-token strings
    from raw text. Some source datasets embed these as data-collection
    artifacts marking a redacted/unknown word, not meaningful content —
    confirmed directly during Phase 3 unification, a BanTH row's text
    contained a literal "<unk>" (e.g. "Notok kom koro priyo <unk>...").
    Left in place, these coincide with tokenizers' own special-token
    strings and can corrupt tokenization (see tag_indictrans2_text /
    logs/phase_6_status.md for the exact failure mode this caused).
    Applied unconditionally, not just for IndicTrans2 — training or
    translating with literal "<unk>" treated as meaningful vocabulary is
    bad signal regardless of tokenizer.
    """
    for tok in _SPECIAL_TOKEN_LITERALS:
        text = text.replace(tok, "")
    return " ".join(text.split())  # collapse whitespace left behind


# ---------------------------------------------------------------------------
# LoRA target modules
# ---------------------------------------------------------------------------

def default_target_modules(model_name):
    name = model_name.lower()
    if "t5" in name or "banglat5" in name:
        return ["q", "k", "v", "o"]  # HF T5Attention naming (no "_proj" suffix)
    # M2M100/NLLB/IndicTrans2-style architectures
    return ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]


# ---------------------------------------------------------------------------
# transformers/IndicTrans2 remote-code compatibility shims
# ---------------------------------------------------------------------------

def ensure_transformers_onnx_shim():
    """Newer `transformers` releases removed the `transformers.onnx`
    submodule (ONNX export moved to the separate `optimum` package), but
    AI4Bharat's IndicTrans2 custom modeling code — loaded dynamically via
    `trust_remote_code=True` — still imports from it at module-load time,
    for an optional ONNX-export code path this project never exercises.
    Hit for real on Kaggle's pre-installed transformers, 2026-08-05 — first
    `ModuleNotFoundError: No module named 'transformers.onnx'`, then, after
    shimming just that, a second missing piece one line further down in the
    same file (`transformers.onnx.utils`). Shims both as a real package
    (`__path__` set) with an auto-synthesizing `__getattr__`, permissive to
    whatever else that file might import from either, rather than patching
    one missing symbol at a time as more surface.
    """
    try:
        import transformers.onnx.utils  # noqa: F401
        return  # already importable (older transformers, or already shimmed)
    except ImportError:
        pass

    class _Permissive:
        """Accepts any constructor args and does nothing — safe as a base
        class for subclassing, and safe as the "return value" when called
        like a function, since nothing in the code paths we actually run
        inspects what these placeholders produce."""

        def __init__(self, *args, **kwargs):
            pass

    class _AutoAttrModule(types.ModuleType):
        def __getattr__(self, name):
            if name.startswith("__") and name.endswith("__"):
                # Never auto-synthesize dunder/special attributes. Doing so
                # broke unrelated code that generically introspects
                # sys.modules — hit for real on Kaggle, 2026-08-17:
                # transformers.modeling_utils's own import (nothing to do
                # with IndicTrans2/ONNX) pulls in torch._dynamo, whose
                # custom-op registration calls inspect.getframeinfo ->
                # findsource -> getmodule, which iterates every entry in
                # sys.modules and reads `.__file__` off each one. Since this
                # shim is registered in sys.modules (see below), that
                # touched this __getattr__, which synthesized a *class*
                # named "__file__" (a `type(name, ...)` call, not a
                # string) — Python's inspect module then called
                # `filename.endswith(...)` on that class and crashed with
                # `AttributeError: type object '__file__' has no attribute
                # 'endswith'`. Raising AttributeError here instead makes
                # `hasattr(shim, '__whatever__')` correctly report False for
                # anything not explicitly set below, same as any real
                # module with an unset dunder — only non-dunder names
                # (the actual classes/functions IndicTrans2's remote code
                # imports from transformers.onnx/.utils) still auto-synthesize.
                raise AttributeError(name)
            value = type(name, (_Permissive,), {})
            setattr(self, name, value)
            return value

    onnx_shim = _AutoAttrModule("transformers.onnx")
    onnx_shim.__file__ = "<catla_transformers_onnx_shim>"  # real string, not synthesized
    onnx_shim.__path__ = []  # mark as a package so `transformers.onnx.utils` resolves
    utils_shim = _AutoAttrModule("transformers.onnx.utils")
    utils_shim.__file__ = "<catla_transformers_onnx_shim>"
    onnx_shim.utils = utils_shim

    sys.modules["transformers.onnx"] = onnx_shim
    sys.modules["transformers.onnx.utils"] = utils_shim


def ensure_tie_weights_compat():
    """Newer `transformers` refactored how `PreTrainedModel.
    _finalize_load_state_dict` calls `tie_weights()` — it now passes
    `missing_keys=`/`recompute_mapping=` kwargs. AI4Bharat's custom
    `IndicTransForConditionalGeneration` overrides `tie_weights()` with an
    older, argument-less signature, so this crashes. Patches
    `_finalize_load_state_dict` to make whatever model it's finalizing
    tolerant of an old-style `tie_weights()` right before calling the real,
    unmodified original implementation — doesn't skip anything the
    original does after that call, just makes the one problematic call
    succeed. Safe no-op for models whose `tie_weights()` already accepts
    the new kwargs (NLLB, BanglaT5, or any model that doesn't override it).
    """
    import inspect as _inspect

    import transformers.modeling_utils as _mu

    if getattr(_mu.PreTrainedModel, "_catla_tie_weights_patched", False):
        return  # already patched (e.g. a previous call in the same process)

    try:
        raw = _mu.PreTrainedModel.__dict__["_finalize_load_state_dict"]
    except KeyError:
        return  # this transformers version doesn't have this method at all — nothing to patch

    def _make_tie_weights_compatible(model):
        try:
            sig = _inspect.signature(model.tie_weights)
        except (TypeError, ValueError):
            return
        has_var_kwargs = any(p.kind == _inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        if has_var_kwargs or "missing_keys" in sig.parameters:
            return  # already compatible with the new calling convention
        original_tie_weights = model.tie_weights

        def _compat_tie_weights(*args, **kwargs):
            return original_tie_weights()

        model.tie_weights = _compat_tie_weights

    if isinstance(raw, classmethod):
        original_func = raw.__func__

        def wrapper(cls, model, load_config, load_info):
            _make_tie_weights_compatible(model)
            return original_func(cls, model, load_config, load_info)

        _mu.PreTrainedModel._finalize_load_state_dict = classmethod(wrapper)
    elif isinstance(raw, staticmethod):
        original_func = raw.__func__

        def wrapper(model, load_config, load_info):
            _make_tie_weights_compatible(model)
            return original_func(model, load_config, load_info)

        _mu.PreTrainedModel._finalize_load_state_dict = staticmethod(wrapper)
    else:
        def wrapper(self, model, load_config, load_info):
            _make_tie_weights_compatible(model)
            return raw(self, model, load_config, load_info)

        _mu.PreTrainedModel._finalize_load_state_dict = wrapper

    _mu.PreTrainedModel._catla_tie_weights_patched = True


def ensure_all_compat_shims():
    """Convenience: apply every environment/remote-code compatibility fix
    in one call. Safe to call multiple times and safe for engines that
    don't need any of them (NLLB, BanglaT5) — every shim here checks
    whether it's already applied/needed before doing anything."""
    ensure_transformers_onnx_shim()
    ensure_tie_weights_compat()
    ensure_indictranstoolkit_import_compat()  # already ran at module import
    # time too (see its own docstring); calling again here is a harmless
    # no-op, kept only for consistency with the other shims in this function.
