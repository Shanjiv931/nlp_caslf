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
import sys
import types

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
