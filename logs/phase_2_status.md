# Phase 2 — Self-Collecting Twitter/X Posts — Status

**Date:** 2026-08-04

## Done
- Attempted the primary path (live scraping via `snscrape`) and confirmed it
  does not work in this environment: `snscrape` 0.7.0 — its last release,
  June 2023, predating most of X's post-2023 API/GraphQL lockdown — crashes
  on startup under Python 3.12 (`AttributeError: 'FileFinder' object has no
  attribute 'find_module'`, a removed `importlib` internal). The library is
  effectively unmaintained and X's platform changes since 2023 would very
  likely have blocked it even if it ran. `tweepy`'s free API tier no longer
  includes search access either, so the PRD's secondary fallback wasn't
  viable either without the user's own paid/elevated X API credentials
  (out of scope — free/open-source only).
- Used the PRD's documented tertiary fallback: built a proxy held-out test
  set by sampling the noisiest, most social-media-style rows already in
  Phase 1 data (`pipeline/collect_test_set.py`):
  - BanTH test split (YouTube comments) — 800 rows
  - BnSentMix (Facebook/YouTube/e-commerce, code-mixed) — 799 rows
  - En-Bn code-mixed sentiment (reviews) — 599 rows
  - SentMix-3L (social media, bn-en-hi code-mixed) — 399 rows
  - EmoMix-3L (social media, bn-en-hi code-mixed) — 399 rows
  - Deduplicated on exact text match, shuffled, written to
    `data/collected_tweets/test_set.jsonl` — **2,996 rows** total.
- Every row is tagged `"is_proxy_test_set": true` and `"source_dataset"` so
  no downstream phase can mistake this for genuine live-scraped tweets.

## Honesty note
This is explicitly a **fallback proxy set, not real Twitter/X data** — the
PRD's Phase 2 goal (organic tweets held out entirely for TEST, per Phase 5)
is only partially met: these rows are genuinely noisy/code-mixed/social-
media-style text from comparable platforms (YouTube, Facebook, e-commerce),
but they are not tweets, have no tweet IDs/hashtags/timestamps, and were not
collected fresh — they're a resample of data already in `data/raw/`. If the
user later gets X API access (elevated/paid tier) or another scraping route,
`pipeline/collect_test_set.py` should be superseded by a real scraper feeding
the same `test_set.jsonl` schema.

## Next
Phase 3 — cleaning, normalization & schema unification: build
`pipeline/clean.py` to merge Phase 1's `data/raw/` datasets and this proxy
test set into one unified JSONL schema (bn/en text, hashtags, emojis,
mentions, code-mix flags, romanization flags) at `data/processed/unified.jsonl`.
