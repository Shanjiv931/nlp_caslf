# CATLA — Evaluation Results

**Status: methodology and infrastructure complete, actual numbers pending
a real run.** This machine cannot execute torch/transformers (no CUDA GPU,
and a Device Guard policy blocks even CPU torch — see
`logs/phase_6_status.md`), so `eval/evaluate.py` has not been run for
real. Every number in this document would need to come from actually
running it on Kaggle/Colab or another working-torch environment — nothing
below is fabricated or estimated in its place, per this project's standing
rule against inventing benchmark numbers.

## How to produce real results

```bash
python eval/evaluate.py --direction bn2en --max_examples 200
python eval/evaluate.py --direction en2bn --max_examples 200
```

Each run writes `eval/results_<direction>.json` and prints the
single-model-baseline vs. full-pipeline delta directly. `--max_examples`
defaults to 200 (not the full test set) because `full_pipeline` mode calls
all 3 fine-tuned engines plus a 7B-parameter LLM per example (see
`pipeline/quality_pipeline.py`) — a full run over the entire held-out test
set (63,618 rows, see `logs/phase_5_status.md`) would be free-tier-GPU-hour
prohibitive at that cost per example. Raise `--max_examples` (or drop it
for the full set) once time/compute allows; 200 is a reasonable statistical
sample for directional confidence, not a substitute for a full run before
any accuracy claim is made publicly.

## What gets measured

Three modes, run over the same held-out `bn_en_translation` test pairs
from `data/processed/test.jsonl` (Phase 5's split — includes the
proxy-organic test rows, see `logs/phase_2_status.md`):

| Mode | What it is |
|---|---|
| `pretrained_baseline` | The base model (e.g. `facebook/nllb-200-distilled-600M`) with no LoRA adapter — never fine-tuned on this project's data at all. |
| `single_model_baseline` | One fine-tuned engine (default NLLB), beam search only, no ensemble/QE/LLM-postedit/round-trip-verify. |
| `full_pipeline` | `quality_pipeline.produce_translation()` — the complete Phase 7 chain. |

Metrics per mode: **BLEU, chrF++, TER** (`sacrebleu`, genuinely runnable
without torch — verified working locally, see `eval/tests/test_evaluate.py`),
**COMET** (reference-based, `Unbabel/wmt22-comet-da` — distinct from
`qe_rerank.py`'s reference-free CometKiwi used at actual inference time,
where no reference exists), and the **Platform-Fidelity Score**
(`eval/platform_fidelity.py` — emoji/mention/URL/hashtag-count
preservation; verified working locally with 11 passing unit tests).

### Platform-Fidelity Score: an honest scope note
PFS here measures three sub-dimensions that are genuinely just string
comparisons: emoji preservation, verbatim mention/URL preservation, and
hashtag *count* preservation (not hashtag translation *correctness* — no
automatic ground truth exists for whether a translated hashtag reads
naturally without human judgment). It does **not** score the PRD's fourth
stated dimension, "slang localized to a natural target-register
equivalent" — there's no reference-free automatic way to judge slang
naturalness, and inventing a fake proxy score for it would misrepresent
what's actually being measured. Slang quality is a qualitative/human-review
question, addressed in the worked examples below once real output exists,
not folded into the numeric PFS.

## The delta this whole architecture is supposed to prove

The PRD's core claim (Section 0) is that the ensemble+QE+LLM-postedit+
round-trip-verify stack beats a single fine-tuned model, which is why
`evaluate.py` explicitly computes and prints `full_pipeline -
single_model_baseline` for every metric. That delta — not either number in
isolation — is the actual evidence this project's added complexity over a
"fine-tune one model and ship it" baseline was worth it. **Once real
numbers exist, they belong here, in this section, with the delta stated
explicitly.**

## Worked examples

Once a real run exists, this section should include 15-20 concrete
examples per the PRD's requirement (FR10 / Phase 8 spec), explicitly
including cases where the QE/LLM-postedit/round-trip-verify layer visibly
rescued a bad single-model translation — the clearest, most legible
evidence for a human reader that the extra architecture earns its keep,
better than any aggregate score alone.

## Language-detection gate accuracy, OCR word error rate

Out of scope for this document (Phase 8 is MT-quality evaluation
specifically) — tracked separately once Phase 10 (image input pipeline:
OCR + language-ID gate) exists. See the PRD's own success metrics section
for the ≥99% gate accuracy and ≤10% OCR WER targets.
