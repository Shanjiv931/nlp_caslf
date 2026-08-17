"""Phase 7 — roundtrip_verify.py: round-trip semantic consistency check.

Translates the (post-edited) candidate back into the source language and
compares it against the original source text using a free multilingual
sentence-embedding model (LaBSE), via cosine similarity. This is the
pipeline's last line of defense (PRD Section 0, principle 4): if the
round-trip drifts too far from the original meaning, the result is flagged
rather than silently shipped.

Per the PRD, round-trip translation reuses just the primary engine for
speed rather than the full ensemble+QE pipeline again — default NLLB
(fastest of the 3 fine-tuned engines to load/run). The retry/fallback
DECISION logic (retry once, fall back to the pre-LLM-edit candidate, or
flag low_confidence) lives in quality_pipeline.py, which has access to
both the pre-edit and post-edit candidates to choose between — this module
only provides the core, reusable "how similar are these two texts"
primitive plus the round-trip translation step itself.

This machine cannot run torch (see logs/phase_6_status.md) — the embedding
model load/encode and round-trip generation calls are correct-by-
construction against the real sentence-transformers/ensemble.py APIs but
unverified here. The direction-reversal and threshold-decision logic (pure
Python, no model calls) is what's unit-tested locally.
"""
DEFAULT_LABSE_MODEL = "sentence-transformers/LaBSE"
DEFAULT_SIMILARITY_THRESHOLD = 0.85

_EMBEDDER_CACHE = {}


def reverse_direction(direction):
    """Pure logic — testable without a model. Round-tripping a bn2en
    output means translating it back en2bn, and vice versa."""
    return "en2bn" if direction == "bn2en" else "bn2en"


def load_embedder(model_name=DEFAULT_LABSE_MODEL):
    if model_name in _EMBEDDER_CACHE:
        return _EMBEDDER_CACHE[model_name]
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(model_name)
    _EMBEDDER_CACHE[model_name] = embedder
    return embedder


def compute_similarity(embedder, text_a, text_b) -> float:
    from sentence_transformers import util
    embeddings = embedder.encode([text_a, text_b], convert_to_tensor=True)
    return float(util.cos_sim(embeddings[0], embeddings[1]).item())


def round_trip_translate(text, direction, engine_key="nllb") -> str:
    """Translates `text` back toward the original source language, using a
    single engine for speed (the PRD explicitly allows this rather than
    the full ensemble+QE pass) — beam search only, no sampling diversity
    needed here since we just need one representative back-translation to
    compare against, not a candidate pool."""
    from ensemble import generate_candidates_for_engine, load_engine

    back_direction = reverse_direction(direction)
    ctx = load_engine(engine_key, back_direction)
    candidates = generate_candidates_for_engine(ctx, text, num_beam_candidates=1, num_sample_candidates=0)
    return candidates[0].text if candidates else ""


def verify(original_source_text, candidate_text, direction, threshold=DEFAULT_SIMILARITY_THRESHOLD, engine_key="nllb") -> dict:
    """Core entry point: translate `candidate_text` back to the source
    language and score its semantic similarity to `original_source_text`.
    Always returns a result dict, never raises on a low score — a failed
    check is information for the caller to act on, not an exception."""
    roundtrip_text = round_trip_translate(candidate_text, direction, engine_key=engine_key)
    embedder = load_embedder()
    similarity = compute_similarity(embedder, original_source_text, roundtrip_text)
    return {
        "roundtrip_text": roundtrip_text,
        "similarity": similarity,
        "passed": similarity >= threshold,
        "threshold": threshold,
    }


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "আমি ভালো আছি"
    candidate = sys.argv[2] if len(sys.argv) > 2 else "I am doing well"
    direction = sys.argv[3] if len(sys.argv) > 3 else "bn2en"
    result = verify(src, candidate, direction)
    for k, v in result.items():
        print(f"{k}: {v}")
