"""Phase 7 — qe_rerank.py: Quality Estimation-based candidate reranking.

Scores every candidate from ensemble.py with a free, open-source,
reference-free QE model — COMET-QE / CometKiwi, via the open-source `comet`
toolkit (`unbabel-comet` in requirements.txt) — and picks the highest-
scoring one. Reference-free QE means it scores (source, candidate) pairs
directly without needing a gold reference translation, which is exactly
what's needed at real inference time (there is no reference for a user's
tweet).

Per the PRD: if no QE model can be loaded for any reason, falls back to the
length-normalized log-probability ensemble.py already captured for
beam-search candidates during generation, and clearly logs that a fallback
was used — this project's `sanitize_text` sibling: never silently degrade
without saying so.

This machine cannot run torch (see logs/phase_6_status.md), so the actual
COMET model load/score calls are unverified here — the fallback path (pure
Python, no model calls) is what's unit-tested locally.
"""
import math
from typing import List, Optional, Tuple

from ensemble import Candidate

# Tried in this order — CometKiwi-XL is more accurate but heavier; falls
# back to the smaller checkpoint if the -xl one can't be downloaded/loaded
# on whatever free-tier hardware this runs on (VRAM, disk, etc.).
QE_CHECKPOINT_CANDIDATES = [
    "Unbabel/wmt23-cometkiwi-da-xl",
    "Unbabel/wmt22-cometkiwi-da",
]

_QE_MODEL = None
_QE_LOAD_ATTEMPTED = False
_QE_LOAD_ERROR = None

# Set to False once unbabel-comet is upgraded to a version compatible with
# the installed transformers (see the note in load_qe_model below) --
# confirmed 2026-09-05 that loading any CometKiwi checkpoint is currently
# pure wasted cost: it will download successfully but crash on the very
# first .predict() call, so there is no point ever attempting the load.
_QE_KNOWN_INCOMPATIBLE = True


def load_qe_model(checkpoints=None):
    """Loads and caches a reference-free COMET-QE model. Returns None (and
    records the reason in _QE_LOAD_ERROR) if none of the candidate
    checkpoints could be loaded — callers must handle that by falling back
    to score_candidates_fallback, never by crashing or silently shipping
    unranked output."""
    global _QE_MODEL, _QE_LOAD_ATTEMPTED, _QE_LOAD_ERROR
    if _QE_LOAD_ATTEMPTED:
        return _QE_MODEL
    _QE_LOAD_ATTEMPTED = True

    if _QE_KNOWN_INCOMPATIBLE:
        # comet/encoders/xlmr.py:95's `last_hidden_states, _, all_layers =
        # self.model(...)` expects a 3-tuple from the underlying HF model's
        # forward() call -- a pre-~4.30 transformers calling convention.
        # transformers==5.0.0 (what's actually installed) returns a 2-tuple
        # there, causing `ValueError: not enough values to unpack (expected
        # 3, got 2)` on every comet/CometKiwi .predict() call regardless of
        # checkpoint. Confirmed via full traceback on the reference-based
        # COMET path (compute_comet() in eval/evaluate.py, 2026-09-05) --
        # the exact same encoder forward() chain is used here, so any
        # checkpoint loaded here would crash identically on the first
        # score_candidates_with_qe() call and fall through to
        # score_candidates_fallback() anyway (see rerank()'s except below).
        # Skip straight to the fallback instead of paying for a multi-GB
        # download (wmt23-cometkiwi-da-xl, tried first, is NOT small) that
        # is guaranteed to end in the same place -- and that can hang
        # indefinitely with zero visible progress on a flaky connection,
        # which is exactly what happened during a real full_pipeline run
        # (observed: 0% GPU, ~1% CPU, completely flat for minutes).
        _QE_LOAD_ERROR = (
            "skipped: unbabel-comet==2.2.7 is confirmed incompatible with the "
            "installed transformers (comet/encoders/xlmr.py:95 3-tuple unpack "
            "vs. transformers' 2-tuple return) -- loading would only waste a "
            "multi-GB download before crashing on the first predict() call"
        )
        return None

    from comet import download_model, load_from_checkpoint

    checkpoints = checkpoints or QE_CHECKPOINT_CANDIDATES
    errors = []
    for checkpoint in checkpoints:
        try:
            path = download_model(checkpoint)
            _QE_MODEL = load_from_checkpoint(path)
            return _QE_MODEL
        except Exception as e:
            errors.append(f"{checkpoint}: {type(e).__name__}: {e}")
    _QE_LOAD_ERROR = "; ".join(errors)
    return None


def score_candidates_with_qe(source_text: str, candidates: List[Candidate], qe_model=None) -> List[Tuple[Candidate, float]]:
    """Scores every candidate with the QE model, reference-free (just
    source + candidate, no gold translation needed). Returns
    (candidate, score) pairs, higher = better, in the same order as input."""
    qe_model = qe_model if qe_model is not None else load_qe_model()
    if qe_model is None:
        raise RuntimeError(f"no QE model available to score with: {_QE_LOAD_ERROR}")

    data = [{"src": source_text, "mt": c.text} for c in candidates]
    output = qe_model.predict(data, batch_size=8, gpus=_comet_gpu_count())
    scores = output.scores if hasattr(output, "scores") else output["scores"]
    return list(zip(candidates, scores))


def _comet_gpu_count():
    """1 if a CUDA GPU is available, else 0. This call runs once per test
    example inside quality_pipeline.py's full_pipeline mode, so hardcoding
    `gpus=0` here (a real bug, not a deliberate CPU-only choice) had an
    outsized effect on total eval runtime -- see eval/evaluate.py's copy
    of this same helper for the full story (2026-09-05)."""
    try:
        import torch
        return 1 if torch.cuda.is_available() else 0
    except ImportError:
        return 0


def score_candidates_fallback(candidates: List[Candidate]) -> List[Tuple[Candidate, Optional[float]]]:
    """Length-normalized log-prob fallback, used only when no QE model
    could be loaded. ensemble.py already captures `sequences_scores`
    (transformers' own length-normalized beam log-prob) for beam-search
    candidates for free during generation; nucleus-sampled candidates don't
    have a comparable score (would need a separate teacher-forced rescoring
    pass this module doesn't do), so they're returned with score=None and
    ranked below any scored candidate — reported honestly as "unscored"
    rather than given a fabricated number.
    """
    return [(c, c.raw_score) for c in candidates]


def pick_best(scored: List[Tuple[Candidate, Optional[float]]]) -> Optional[Candidate]:
    """Picks the highest-scoring candidate. None scores sort last (never
    preferred over anything with a real score) rather than crashing on the
    comparison or being treated as 0.0, which would misrepresent an
    "unscored" candidate as "scored exactly zero"."""
    if not scored:
        return None
    ranked = sorted(scored, key=lambda pair: (pair[1] is not None, pair[1] if pair[1] is not None else -math.inf), reverse=True)
    return ranked[0][0]


def rerank(source_text: str, candidates: List[Candidate]) -> dict:
    """Top-level entry point. Returns a dict with the winning candidate,
    which reranking method actually got used, and the full score list (for
    the transparency panel — FR9 in the PRD requires showing the winning
    engine and QE score, not just silently picking one)."""
    if not candidates:
        return {"winner": None, "method": "none", "scored": []}

    qe_model = load_qe_model()
    scored = None
    if qe_model is not None:
        try:
            scored = score_candidates_with_qe(source_text, candidates, qe_model=qe_model)
            method = "comet_qe"
        except Exception as e:
            # Model LOADED fine but .predict() itself threw -- a distinct
            # failure mode from qe_model being None, and one the original
            # code didn't guard against. Confirmed for real 2026-09-06:
            # a "ValueError: not enough values to unpack (expected 3, got
            # 2)" in the reference-based COMET call (same unbabel-comet
            # package, likely a version-compatibility crack against the
            # much newer pytorch-lightning/torchmetrics actually
            # installed) -- without this except, the identical error here
            # would crash the entire full_pipeline run instead of falling
            # back, directly contradicting this module's own stated design
            # ("never crash, never silently ship unranked output").
            print(f"WARNING: QE model loaded but predict() failed ({type(e).__name__}: {e}) — "
                  f"falling back to length-normalized beam log-probability for this example.")

    if scored is None:
        if qe_model is None:
            print(f"WARNING: no QE model available ({_QE_LOAD_ERROR}) — falling back to "
                  f"length-normalized beam log-probability. Nucleus-sampled candidates have "
                  f"no comparable score and are ranked below any scored beam candidate.")
        scored = score_candidates_fallback(candidates)
        method = "logprob_fallback"

    winner = pick_best(scored)
    return {"winner": winner, "method": method, "scored": scored}


if __name__ == "__main__":
    from ensemble import generate_ensemble_candidates
    import sys

    text = sys.argv[1] if len(sys.argv) > 1 else "আমি ভালো আছি"
    direction = sys.argv[2] if len(sys.argv) > 2 else "bn2en"
    candidates = generate_ensemble_candidates(text, direction)
    result = rerank(text, candidates)
    print(f"method: {result['method']}")
    for c, score in result["scored"]:
        marker = " <-- winner" if c is result["winner"] else ""
        print(f"  [{c.engine}/{c.method}] score={score}: {c.text}{marker}")
