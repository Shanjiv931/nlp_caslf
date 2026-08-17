"""Phase 11 — demo/app.py: the Gradio demo, deployable as-is to a free
HuggingFace Space (see `README.md` in this directory for the Space config
front matter).

Per the PRD, image upload is the primary input modality (a screenshot of a
tweet is the realistic real-world input this whole project is built
around), wired straight to `image_to_translation.handle_uploaded_image()`
from Phase 10. A "Translate Text" tab is offered alongside it for directly
pasting tweet text — useful for testing and for the (also realistic) case
where the user already has the text and doesn't need OCR — wired to
`catla.translate_tweet()` after running the same language gate the image
path uses, so both tabs make identical accept/reject decisions.

**Fast mode vs. quality mode** (PRD requirement): quality mode (default)
runs the full Phase 7 chain — ensemble generation across all 3 fine-tuned
engines, QE reranking, LLM post-editing, round-trip verification. This is
the "extraordinary quality" path but is slow on free CPU-only HF Spaces
hardware (multiple multi-hundred-MB to multi-GB model loads, several
forward passes, one LLM call). Fast mode skips all of that and returns a
single beam-search decode from one engine — usable on free hardware
without the user waiting minutes per request. The toggle is explicit in
the UI so the user always knows which they're getting; quality mode stays
the default per the PRD ("defaulting to quality mode").

This machine cannot run torch (see logs/phase_6_status.md onward), so this
app is correct-by-construction against the already-built and unit-tested
pipeline modules it wires together (`catla.py`, `image_to_translation.py`),
but has NOT been launched/exercised end-to-end here. Real verification
needs a working torch environment (Kaggle/Colab/HF Spaces itself).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline"))

import gradio as gr

import catla
import image_to_translation
from language_gate import detect_and_route

ENGINE_CHOICES = ["nllb", "indictrans2", "banglat5"]
ENGINE_LABELS = {
    "nllb": "NLLB-200 (fastest)",
    "indictrans2": "IndicTrans2",
    "banglat5": "BanglaT5",
}
MODE_QUALITY = "Quality mode (full pipeline — ensemble + QE + LLM post-edit + round-trip verify)"
MODE_FAST = "Fast mode (single model, beam search only)"

LANGUAGE_LABELS = {"bn": "Bengali", "en": "English", "unsupported": "Unsupported"}


def _direction_label(direction):
    return {"bn2en": "Bengali → English", "en2bn": "English → Bengali", None: "—"}.get(direction, "—")


def _format_meta(direction, translation_result, fast_mode):
    """Builds the Markdown info panel shared by both tabs: winning engine,
    QE score, round-trip similarity, low-confidence badge, verification
    note — the fields the PRD requires the demo to surface."""
    if translation_result is None:
        return ""

    lines = [f"**Direction:** {_direction_label(direction)}"]

    if translation_result.get("low_confidence"):
        lines.append("**⚠️ Low confidence** — this translation failed one or more internal quality checks.")

    engine = translation_result.get("winning_engine")
    method = translation_result.get("winning_method")
    if engine:
        lines.append(f"**Engine used:** {ENGINE_LABELS.get(engine, engine)} ({method or 'n/a'})")

    if fast_mode:
        lines.append("**QE score:** skipped (fast mode)")
        lines.append("**Round-trip similarity:** skipped (fast mode)")
    else:
        qe_score = translation_result.get("qe_score")
        qe_method = translation_result.get("qe_method")
        lines.append(f"**QE score:** {qe_score:.3f} ({qe_method})" if qe_score is not None
                     else f"**QE score:** n/a ({qe_method or 'no candidates'})")

        rt_sim = translation_result.get("roundtrip_similarity")
        lines.append(f"**Round-trip similarity:** {rt_sim:.3f}" if rt_sim is not None
                     else "**Round-trip similarity:** n/a")

        if translation_result.get("edit_applied"):
            lines.append("**LLM post-edit:** applied")

    note = translation_result.get("verification_note")
    if note:
        lines.append(f"**Note:** {note}")

    err = translation_result.get("error")
    if err:
        lines.append(f"**Error:** {err}")

    return "\n\n".join(lines)


def translate_text_tab(text, mode, engine):
    fast_mode = mode == MODE_FAST
    text = (text or "").strip()
    if not text:
        return "", "", "", ""

    gate_result = detect_and_route(text)
    lang = gate_result["language"]
    lang_line = f"**Detected language:** {LANGUAGE_LABELS.get(lang, lang)} (confidence: {gate_result['confidence']:.3f})"

    if lang == "unsupported":
        return (image_to_translation.UNSUPPORTED_LANGUAGE_MESSAGE, lang_line, "", "")

    direction = "bn2en" if lang == "bn" else "en2bn"
    result = catla.translate_tweet(text, direction, roundtrip_engine=engine, fast_mode=fast_mode)
    translation = result.get("translation") or "(translation failed — see notes below)"
    meta = _format_meta(direction, result, fast_mode)
    return "", lang_line, translation, meta


def translate_image_tab(image_path, mode, engine):
    fast_mode = mode == MODE_FAST
    if not image_path:
        return "", "", "", "", ""

    result = image_to_translation.handle_uploaded_image(
        image_path, fast_mode=fast_mode, roundtrip_engine=engine,
    )

    ocr_text = result.get("ocr_text") or ""
    ocr_conf = result.get("ocr_confidence")
    ocr_line = f"OCR confidence: {ocr_conf:.3f}" if ocr_conf is not None else ""

    lang = result.get("detected_language")
    lang_line = f"**Detected language:** {LANGUAGE_LABELS.get(lang, lang)} (confidence: {result.get('language_confidence', 0):.3f})"

    if not result.get("supported"):
        return ocr_text, ocr_line, result["message"], lang_line, ""

    translation = result.get("translation") or "(translation failed — see notes below)"
    meta = _format_meta(result.get("direction"), result.get("translation_result"), fast_mode)
    return ocr_text, ocr_line, "", lang_line, f"**Translation:** {translation}\n\n{meta}"


with gr.Blocks(title="CATLA — Bengali↔English Tweet Translator") as demo:
    gr.Markdown(
        "# CATLA — Context-Aware Translation & Localization Algorithm\n"
        "Bidirectional Bengali↔English tweet translation. Upload a tweet screenshot "
        "(primary flow) or paste tweet text directly. Built entirely on free/open-source "
        "models and free-tier compute — no paid APIs.\n\n"
        "Unsupported languages are explicitly rejected rather than mistranslated or guessed."
    )

    with gr.Row():
        mode = gr.Radio([MODE_QUALITY, MODE_FAST], value=MODE_QUALITY, label="Mode",
                         info="Quality mode is slower but runs the full ensemble+QE+LLM-postedit+"
                              "round-trip-verify pipeline. Fast mode is a single model, beam search only.")
        engine = gr.Dropdown(ENGINE_CHOICES, value="nllb", label="Model",
                              info="Fast mode: the only engine used. Quality mode: the round-trip "
                                   "verification engine (ensemble generation always uses all 3).")

    with gr.Tab("📷 Translate Image (primary)"):
        image_input = gr.Image(type="filepath", label="Upload a tweet screenshot")
        image_btn = gr.Button("Translate Image", variant="primary")
        image_alert = gr.Markdown()
        with gr.Row():
            image_ocr_text = gr.Textbox(label="OCR-extracted text", interactive=False)
            image_ocr_conf = gr.Markdown()
        image_lang = gr.Markdown()
        image_result = gr.Markdown(label="Result")

        image_btn.click(
            translate_image_tab, inputs=[image_input, mode, engine],
            outputs=[image_ocr_text, image_ocr_conf, image_alert, image_lang, image_result],
        )

    with gr.Tab("⌨️ Translate Text"):
        text_input = gr.Textbox(label="Tweet text", lines=3, placeholder="আমি ভালো আছি #blessed")
        text_btn = gr.Button("Translate Text", variant="primary")
        text_alert = gr.Markdown()
        text_lang = gr.Markdown()
        text_translation = gr.Textbox(label="Translation", interactive=False)
        text_meta = gr.Markdown(label="Details")

        text_btn.click(
            translate_text_tab, inputs=[text_input, mode, engine],
            outputs=[text_alert, text_lang, text_translation, text_meta],
        )

    gr.Markdown(
        "---\n"
        "Model adapters: [shanjivkr/catla-* on HuggingFace Hub](https://huggingface.co/shanjivkr). "
        "Source: [github.com/Shanjiv931/nlp_caslf](https://github.com/Shanjiv931/nlp_caslf)."
    )


if __name__ == "__main__":
    demo.queue().launch()
