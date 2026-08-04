"""
Scrapes original (non-reply) tweets in Bengali and English on shared trending
topics, via twitterapi.io's Advanced Search endpoint, for the
NLP-Based Framework for Context-Aware Bengali-English Twitter
Translation & Localization project.

Reads the API key from the TWITTERIO_API_KEY environment variable —
never hardcode it here.
"""

import csv
import json
import os
import sys
import time
from datetime import datetime

import requests

API_KEY = os.environ.get("TWITTERIO_API_KEY")
if not API_KEY:
    sys.exit("Set TWITTERIO_API_KEY in the environment before running this script.")

BASE_URL = "https://api.twitterapi.io/twitter/tweet/advanced_search"
HEADERS = {"X-API-Key": API_KEY}

# Shared topics, one query per language so posts are "parallel" on subject
# matter even when not direct translations of each other.
TOPICS = {
    "news_current_affairs": {
        "bn": '(বাংলাদেশ OR ঢাকা OR সরকার OR নির্বাচন) lang:bn -filter:replies',
        "en": '(Bangladesh OR Dhaka) (news OR government OR election) lang:en -filter:replies',
    },
    "cricket_sports": {
        "bn": '(ক্রিকেট OR টাইগার্স OR বিপিএল) lang:bn -filter:replies',
        "en": '(Bangladesh cricket OR Tigers cricket OR BPL) lang:en -filter:replies',
    },
    "entertainment_culture": {
        "bn": '(নাটক OR সিনেমা OR গান OR ঈদ) lang:bn -filter:replies',
        "en": '(Dhallywood OR "Bangladeshi drama" OR Eid) lang:en -filter:replies',
    },
    "daily_life_opinion": {
        "bn": '(ঢাকা ট্রাফিক OR আবহাওয়া OR দ্রব্যমূল্য) lang:bn -filter:replies',
        "en": '(Dhaka traffic OR Bangladesh weather OR Bangladesh prices) lang:en -filter:replies',
    },
    "tech_business": {
        "bn": '(প্রযুক্তি OR স্টার্টআপ বাংলাদেশ) lang:bn -filter:replies',
        "en": '(Bangladesh tech OR Bangladesh startup) lang:en -filter:replies',
    },
}

TOTAL_TARGET = 5000
PER_QUERY_CAP = TOTAL_TARGET // (len(TOPICS) * 2)  # ~500 per topic/lang pair
REQUEST_DELAY_SEC = 0.6

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(OUT_DIR, "bn_en_tweets.csv")
JSON_PATH = os.path.join(OUT_DIR, "bn_en_tweets.jsonl")

FIELDNAMES = [
    "id", "topic", "lang", "created_at", "text",
    "hashtags", "author_username", "author_name",
    "like_count", "retweet_count", "reply_count", "view_count", "url",
]


MAX_429_RETRIES_PER_TURN = 2  # give up on this pair for now rather than burn the whole key retrying one topic
MAX_EMPTY_ROUNDS = 6  # stop if several full round-robin passes add nothing (dead key, not just slow)


def fetch_page(query, cursor):
    """Fetch a single page. Returns (tweets, next_cursor, has_next, status)
    where status is one of 'ok', 'exhausted' (no more pages/results),
    'rate_limited', 'out_of_credits', 'error'."""
    params = {"query": query, "queryType": "Latest", "cursor": cursor}
    retries = 0
    while True:
        try:
            resp = requests.get(BASE_URL, headers=HEADERS, params=params, timeout=30)
        except requests.RequestException as e:
            print(f"    request error: {e}")
            return [], cursor, False, "error"

        if resp.status_code == 401:
            sys.exit("401 Unauthorized — API key is invalid or expired.")
        if resp.status_code == 402:
            return [], cursor, False, "out_of_credits"
        if resp.status_code == 429:
            retries += 1
            if retries > MAX_429_RETRIES_PER_TURN:
                return [], cursor, False, "rate_limited"
            time.sleep(15)
            continue
        if resp.status_code != 200:
            print(f"    HTTP {resp.status_code}: {resp.text[:200]}")
            return [], cursor, False, "error"

        data = resp.json()
        tweets = data.get("tweets", [])
        has_next = bool(data.get("has_next_page")) and bool(data.get("next_cursor"))
        next_cursor = data.get("next_cursor", cursor)
        if not tweets:
            return [], cursor, False, "exhausted"
        return tweets, next_cursor, has_next, "ok"


def flatten(tweet, topic):
    hashtags = [h.get("text", "") for h in tweet.get("entities", {}).get("hashtags", [])]
    author = tweet.get("author", {}) or {}
    return {
        "id": tweet.get("id"),
        "topic": topic,
        "lang": tweet.get("lang"),
        "created_at": tweet.get("createdAt"),
        "text": tweet.get("text", "").replace("\n", " ").strip(),
        "hashtags": ";".join(hashtags),
        "author_username": author.get("userName"),
        "author_name": author.get("name"),
        "like_count": tweet.get("likeCount"),
        "retweet_count": tweet.get("retweetCount"),
        "reply_count": tweet.get("replyCount"),
        "view_count": tweet.get("viewCount"),
        "url": tweet.get("url"),
    }


def load_existing():
    seen_ids = set()
    rows = []
    raw_records = []
    per_pair_count = {}

    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, "r", newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                rows.append(row)
                seen_ids.add(row["id"])
                key = (row["topic"], row["lang"])
                per_pair_count[key] = per_pair_count.get(key, 0) + 1

    if os.path.exists(JSON_PATH):
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw_records.append(json.loads(line))

    return seen_ids, rows, raw_records, per_pair_count


def save(rows, raw_records):
    with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        for rec in raw_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    seen_ids, rows, raw_records, per_pair_count = load_existing()
    if rows:
        print(f"Resuming: {len(rows)} tweets already collected from a previous run.")

    # One entry per topic/lang pair still short of its cap; each tracks its
    # own pagination cursor so we can hop between pairs instead of draining
    # one topic before starting the next.
    pairs = []
    for topic, lang_queries in TOPICS.items():
        for lang, query in lang_queries.items():
            already = per_pair_count.get((topic, lang), 0)
            remaining = PER_QUERY_CAP - already
            if remaining > 0:
                pairs.append({
                    "topic": topic, "lang": lang, "query": query,
                    "cursor": "", "remaining": remaining,
                })
            else:
                print(f"[{topic}/{lang}] already have {already} >= target {PER_QUERY_CAP}, skipping")

    # Scarce credits go to the emptiest pairs first, so a key that dies
    # halfway through still leaves every topic with *something* rather than
    # maxing out the first couple of topics and starving the rest.
    pairs.sort(key=lambda p: -p["remaining"])

    out_of_credits = False
    since_last_save = 0
    empty_rounds = 0

    while pairs and not out_of_credits:
        made_progress_this_round = False
        any_rate_limited = False
        for pair in list(pairs):
            tweets, next_cursor, has_next, status = fetch_page(pair["query"], pair["cursor"])

            if status == "out_of_credits":
                print(f"    [{pair['topic']}/{pair['lang']}] 402 Payment Required — out of API credits, stopping.")
                out_of_credits = True
                break

            if status in ("exhausted", "error"):
                pairs.remove(pair)
                continue

            if status == "rate_limited":
                any_rate_limited = True
                continue  # retried again next round, in case it clears up

            new_count = 0
            for t in tweets:
                tid = t.get("id")
                if not tid or tid in seen_ids:
                    continue
                seen_ids.add(tid)
                rows.append(flatten(t, pair["topic"]))
                raw_records.append(t)
                new_count += 1
            since_last_save += new_count
            if new_count:
                made_progress_this_round = True
                print(f"    [{pair['topic']}/{pair['lang']}] +{new_count} (total so far: {len(rows)})")

            pair["remaining"] -= new_count
            pair["cursor"] = next_cursor
            if pair["remaining"] <= 0 or not has_next:
                pairs.remove(pair)

            if since_last_save >= 40:
                save(rows, raw_records)
                since_last_save = 0

            time.sleep(REQUEST_DELAY_SEC)

        if out_of_credits:
            break

        if not made_progress_this_round:
            empty_rounds += 1
            if any_rate_limited:
                print(f"    round produced nothing, {len(pairs)} pairs still rate-limited, cooling down 20s")
                time.sleep(20)
            if empty_rounds >= MAX_EMPTY_ROUNDS:
                print(f"    {empty_rounds} rounds with no progress, giving up for this key")
                break
        else:
            empty_rounds = 0

    save(rows, raw_records)
    print(f"\nDone. {len(rows)} unique tweets written to:\n  {CSV_PATH}\n  {JSON_PATH}")
    print(f"Finished at {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
