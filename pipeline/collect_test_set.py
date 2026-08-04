"""Phase 2 — organic/noisy test set collection.

Primary path (live Twitter/X scraping via snscrape) is unusable in this
environment: snscrape 0.7.0 (its last release, June 2023, predating most of
X's post-2023 API lockdown) fails to even start under Python 3.12 due to
removed importlib internals, and X's free API tier no longer offers search
access for a tweepy-based fallback either.

Per the PRD's documented fallback, this builds a proxy held-out test set by
sampling the noisiest, most social-media-style rows already collected in
Phase 1 (YouTube comments, code-mixed Facebook/YouTube/e-commerce reviews),
rather than fabricating tweet data. Every row is tagged with its source
dataset and `is_proxy_test_set: true` so downstream phases never mistake this
for genuine live-scraped Twitter data.
"""
import json
import os
import random

import pandas as pd

RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "collected_tweets", "test_set.jsonl")

random.seed(42)


def sample_banth(n=800):
    df = pd.read_csv(os.path.join(RAW, "banth", "test.csv"))
    text_col = "Text" if "Text" in df.columns else df.columns[df.columns.str.lower().str.contains("text")][0]
    rows = []
    for _, r in df.sample(min(n, len(df)), random_state=42).iterrows():
        rows.append({
            "text": str(r[text_col]),
            "source_dataset": "banth",
            "domain": "youtube_comment",
            "language_hint": "bn_transliterated",
        })
    return rows


def sample_bnsentmix(n=800):
    df = pd.read_csv(os.path.join(RAW, "bnsentmix", "dataset.csv"), on_bad_lines="skip")
    text_col = [c for c in df.columns if "text" in c.lower() or "sentence" in c.lower()]
    text_col = text_col[0] if text_col else df.columns[0]
    rows = []
    for _, r in df.sample(min(n, len(df)), random_state=42).iterrows():
        rows.append({
            "text": str(r[text_col]),
            "source_dataset": "bnsentmix",
            "domain": "facebook_youtube_ecommerce",
            "language_hint": "bn_en_code_mixed",
        })
    return rows


def sample_sentmix3l(n=400):
    df = pd.read_csv(os.path.join(RAW, "sentmix_3l", "sen_1k.csv"), on_bad_lines="skip")
    text_col = [c for c in df.columns if "text" in c.lower() or "sentence" in c.lower()]
    text_col = text_col[0] if text_col else df.columns[0]
    rows = []
    for _, r in df.sample(min(n, len(df)), random_state=42).iterrows():
        rows.append({
            "text": str(r[text_col]),
            "source_dataset": "sentmix_3l",
            "domain": "social_media",
            "language_hint": "bn_en_hi_code_mixed",
        })
    return rows


def sample_emomix3l(n=400):
    df = pd.read_csv(os.path.join(RAW, "emomix_3l", "emo_1k.csv"), on_bad_lines="skip")
    text_col = [c for c in df.columns if "text" in c.lower() or "sentence" in c.lower()]
    text_col = text_col[0] if text_col else df.columns[0]
    rows = []
    for _, r in df.sample(min(n, len(df)), random_state=42).iterrows():
        rows.append({
            "text": str(r[text_col]),
            "source_dataset": "emomix_3l",
            "domain": "social_media",
            "language_hint": "bn_en_hi_code_mixed",
        })
    return rows


def sample_en_bn_code_mixed(n=600):
    df = pd.read_csv(
        os.path.join(RAW, "en_bn_code_mixed_sentiment", "EnBn_CodeMixed_TwoClass_Sentiment_Balanced_100k.csv"),
        on_bad_lines="skip",
    )
    text_col = [c for c in df.columns if "text" in c.lower() or "review" in c.lower()]
    text_col = text_col[0] if text_col else df.columns[0]
    rows = []
    for _, r in df.sample(min(n, len(df)), random_state=42).iterrows():
        rows.append({
            "text": str(r[text_col]),
            "source_dataset": "en_bn_code_mixed_sentiment",
            "domain": "review",
            "language_hint": "bn_en_code_mixed",
        })
    return rows


def main():
    all_rows = (
        sample_banth()
        + sample_bnsentmix()
        + sample_sentmix3l()
        + sample_emomix3l()
        + sample_en_bn_code_mixed()
    )

    # dedupe on exact text match
    seen = set()
    deduped = []
    for i, row in enumerate(all_rows):
        t = row["text"].strip()
        if not t or t.lower() == "nan" or t in seen:
            continue
        seen.add(t)
        row["id"] = f"proxy_{i:06d}"
        row["is_proxy_test_set"] = True
        deduped.append(row)

    random.shuffle(deduped)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for row in deduped:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {len(deduped)} rows to {OUT}")
    by_source = {}
    for row in deduped:
        by_source[row["source_dataset"]] = by_source.get(row["source_dataset"], 0) + 1
    print("breakdown:", by_source)


if __name__ == "__main__":
    main()
