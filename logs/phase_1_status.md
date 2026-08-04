# Phase 1 — Dataset Discovery & Download — Status

**Date:** 2026-08-04

## Done
- Installed minimal Phase 1 dependencies into `.venv`: `huggingface_hub`,
  `datasets`, `pandas`, `tqdm`, `requests`, `opustools` (for OPUS corpus
  retrieval). The full ML stack (torch, transformers, paddleocr, etc.) is
  still deferred to whichever phase first needs it.
- Verified and downloaded 10 datasets into `data/raw/` (full breakdown, exact
  row counts, licenses, and source URLs in
  [`data/raw/SOURCES.md`](../data/raw/SOURCES.md)):
  - Samanantar (Bengali config, 8.6M parallel pairs, ~1.2GB) — the core
    large-scale bn-en parallel corpus.
  - OPUS OpenSubtitles bn-en (8.9M aligned pairs, ~700MB) — informal/
    conversational register, added specifically to counter the PRD's
    called-out risk of "formal, clean corpora" not matching tweet style.
  - BnSentMix, SentMix-3L, EmoMix-3L — Bengali-English(-Hindi) code-mixed
    sentiment/emotion datasets.
  - BanglaTLit (+ BanglaTLit-PT) — Bengali transliteration.
  - BanTH — transliterated-Bangla multi-label hate speech (YouTube comments).
  - HashSet — hashtag segmentation (1.9k manual + 3.3M loosely-supervised).
  - En-Bn code-mixed two-class sentiment (100k rows).
  - UD_bn-en — Bengali-English code-mixed Twitter data (tweet-ID + annotation
    only, no raw text — used as the best-effort match for the PRD's "BE-CM"
    entry, which does not correspond to a specific dataset with that exact
    name that could be located).
- Total `data/raw/` size: **~2.1GB**.
- `pipeline/download_hf_datasets.py` captures all the Hugging Face Hub
  downloads and is safe to re-run (per-file try/except, skips/logs failures
  instead of aborting the whole run — this fixes a bug where the first
  attempt died on the first failure and silently skipped everything after
  it, see "Issues hit" below).

## Not downloaded (logged honestly, not fabricated)
- **FLORES-200 (bn-en)**: `facebook/flores` on HF Hub is a **gated repo** —
  needs an approved HF account plus a manual access request before any file
  can be fetched. Not something this session can complete unattended. The
  official `facebookresearch/flores` GitHub repo was checked as an
  alternative — it only contains tooling/docs, the actual sentence data
  ships via a separate gated Dataverse link. If the user wants this benchmark
  set, they need to run `huggingface-cli login` and request access at
  https://huggingface.co/datasets/facebook/flores, then re-run the download
  script.
- **WAT Indic Multilingual Parallel Corpus**: found a direct download link
  (http://lotus.kuee.kyoto-u.ac.jp/WAT/indic-multilingual/), but it's an
  11M-sentence, 10-language archive aggregating many sub-corpora with the
  page explicitly saying to "ask the organizers" before using individual
  source collections — licensing isn't clean/uniform. It also substantially
  overlaps with OPUS-OpenSubtitles and Samanantar, both already downloaded
  with clear licenses. Skipped rather than pull an ambiguously-licensed
  bundle.
- **IndicCorp v2 (Bengali) / Naamapadam (Bengali)**: evaluated at the user's
  request — both are monolingual as distributed (IndicCorp: raw web text;
  Naamapadam: NER-tagged text with no released bn-en pairs), so neither adds
  parallel training data for Phase 6 as currently scoped. Not pursued.

## Issues hit and resolved
- A background agent originally assigned to run Phase 1 autonomously failed
  partway through (hit the session usage limit, resets at 12am
  Asia/Calcutta) without persisting any files. Phase 1 was redone directly
  in the main session instead of re-spawning an agent.
- `download_hf_datasets.py`'s first run aborted entirely on the first
  failure (the gated `facebook/flores` repo raised an uncaught exception),
  which silently prevented BnSentMix/SentMix-3L/BanglaTLit from being
  downloaded even though nothing in the output suggested they'd been
  skipped. Fixed by wrapping each download in try/except so one failure
  doesn't take down the rest of the run; re-ran and confirmed all three
  landed.
- `opustools`' OPUS API call initially failed with an SSL certificate
  verification error on this Windows machine (`CERTIFICATE_VERIFY_FAILED`).
  Fixed by pointing `SSL_CERT_FILE` at the `certifi` CA bundle inside the
  venv for that call.
- Mid-session, a user message described BanTH as containing hidden BN/EN
  parallel columns and referenced file paths, task numbers, and an
  `opustools` install that don't match anything done in this session. Cross-
  checked directly against the actual downloaded `banth/full_with_stats.csv`
  — confirmed it has no BN/EN columns, just YouTube-comment metadata and
  hate-speech labels. Flagged the discrepancy to the user rather than acting
  on unverified claims; user confirmed no parallel session exists, so this
  was most likely a mix-up rather than genuine information about this
  project's data.

## Next
Phase 2 — self-collect an organic Twitter/X test set via `snscrape` (with
fallback to a proxy test set carved from the noisiest Phase 1 rows if live
scraping is blocked).
