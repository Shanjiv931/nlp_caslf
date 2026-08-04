"""Post-hoc addition (2026-08-04) — normalize the user-provided, paid-API-
sourced live tweet scrape (data/raw/twitterapi_io_live_tweets/) into the
project's row schema, for LOCAL EVALUATION ONLY.

See data/raw/twitterapi_io_live_tweets/README_NOT_FREE.md for why this data
is excluded from the free/open-source pipeline (paid source) and must never
be pushed to the public HF Hub dataset or used as training data.

Output: data/collected_tweets/live_eval_LOCAL_ONLY.jsonl (gitignored by the
same wildcard rule that covers data/collected_tweets/* generally — only
test_set.jsonl is explicitly allow-listed there, so this new file is
automatically excluded without any additional .gitignore change).
"""
import csv
import json
import os
import re

import emoji as emoji_lib

RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "twitterapi_io_live_tweets")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "collected_tweets", "live_eval_LOCAL_ONLY.jsonl")

HASHTAG_RE = re.compile(r"#(\w+)", re.UNICODE)
MENTION_RE = re.compile(r"@(\w+)", re.UNICODE)
BENGALI_RANGE = re.compile(r"[ঀ-৿]")
LATIN_RANGE = re.compile(r"[A-Za-z]")


def main():
    csv_path = os.path.join(RAW, "bn_en_tweets.csv")
    if not os.path.exists(csv_path):
        print(f"not found: {csv_path} — nothing to process")
        return

    rows = []
    with open(csv_path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            text = row["text"]
            hashtags = HASHTAG_RE.findall(text)
            mentions = MENTION_RE.findall(text)
            emojis = [c for c in text if emoji_lib.is_emoji(c)]
            has_bn = bool(BENGALI_RANGE.search(text))
            has_latin = bool(LATIN_RANGE.search(text))
            rows.append({
                "id": f"live_paid_{row['id']}",
                "source_dataset": "twitterapi_io_live_tweets",
                "pair_type": "monolingual_social",
                "text": text,
                "text_lang": row["lang"],
                "parallel_text": None,
                "parallel_lang": None,
                "romanized_text": None,
                "register": "social_media",
                "hashtags": hashtags,
                "emojis": emojis,
                "mentions": mentions,
                "is_code_mixed": has_bn and has_latin,
                "is_romanized_bn": False,
                "task_labels": {"topic": row["topic"], "like_count": row["like_count"],
                                 "retweet_count": row["retweet_count"], "view_count": row["view_count"]},
                "license": "paid-api-local-eval-only-not-for-redistribution",
                "is_proxy_test_set": False,
                "is_synthetic": False,
                "is_paid_source": True,
                "is_local_eval_only": True,
            })

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_lang = {}
    for row in rows:
        by_lang[row["text_lang"]] = by_lang.get(row["text_lang"], 0) + 1
    print(f"wrote {len(rows)} rows to {OUT_PATH}")
    print("by language:", by_lang)


if __name__ == "__main__":
    main()
