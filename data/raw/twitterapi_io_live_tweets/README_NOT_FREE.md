# NOT part of CATLA's free/open-source data — local evaluation only

**This folder's data (`bn_en_tweets.csv`, `bn_en_tweets.jsonl`) was collected
via [twitterapi.io](https://twitterapi.io)'s Advanced Search endpoint, a paid
third-party API.** That conflicts with the constraint stated at the top of
this project's PRD: *"every tool, dataset, model, and compute resource used
in this project must be free/open-source. No paid APIs, no paid compute, no
proprietary datasets."*

By explicit user decision (2026-08-04), this data is kept and used **only
for local, private evaluation** — it must never be:
- pushed to the public HF Hub dataset (`shanjivkr/catla-bn-en-tweets`) or any
  other public repo,
- used as training data for the LoRA fine-tuning (Phase 6),
- cited anywhere the project claims "100% free/open-source."

`scrape_twitter.py` (the collection script) IS committed to git/GitHub, since
it contains no secrets (reads its API key from an environment variable) and
documents methodology transparently. **The data files themselves
(`bn_en_tweets.csv`/`.jsonl`) are gitignored and stay local-only** — see the
explicit exclusion in `.gitignore`.

## Contents
- 2,640 real tweets (1,348 Bengali, 1,292 English), collected across 5 shared
  topics (news/current affairs, cricket/sports, entertainment/culture, daily
  life/opinion, tech/business) so Bengali and English posts are topically
  comparable even though they aren't direct translation pairs.
- Deduplicated, no empty rows (verified).

## How it's used
Converted (schema-normalized, not translated) into
`data/collected_tweets/live_eval_LOCAL_ONLY.jsonl` by
`pipeline/process_live_eval_set.py`, for use as an additional, genuinely
organic local evaluation set in Phase 8 — kept clearly separate from the
free/open-source pipeline's official train/val/test splits.
