"""Phase 3 — cleaning, normalization & schema unification.

Merges every Phase 1 raw dataset (plus the Phase 2 proxy test set) into one
unified JSONL schema at data/processed/unified.jsonl. Streams row-by-row /
shard-by-shard rather than holding everything in memory, since the two
largest sources (Samanantar, OPUS-OpenSubtitles) total ~17.5M rows combined.

Unified row schema:
    id             str   - "<source_dataset>_<row_index>"
    source_dataset str
    pair_type      str   - "bn_en_translation" | "bn_transliteration" |
                            "monolingual_social" | "hashtag_segmentation"
    text           str   - primary/source-side text
    text_lang      str   - "bn" | "en" | "bn_en_mixed"
    parallel_text  str|null - aligned translation/transliteration, if any
    parallel_lang  str|null
    romanized_text str|null - Banglish/transliterated form of `text`, if a
                              distinct romanized version was provided by the
                              source dataset
    register       str   - "formal" | "informal_subtitles" |
                            "informal_social_youtube" | "social_media" |
                            "review" | "transliteration_data" |
                            "hashtag_data"
    hashtags       list[str]
    emojis         list[str]
    mentions       list[str]
    is_code_mixed  bool  - text contains both Bengali-block and Latin letters
    is_romanized_bn bool - text is Latin-script but sourced/labelled as
                            transliterated Bengali
    task_labels    dict|null - original labels preserved (sentiment,
                               hate-speech categories, etc.) for potential
                               auxiliary use
    license        str
    is_proxy_test_set bool - carried through from Phase 2 rows only
"""
import glob
import json
import os
import re

import emoji as emoji_lib
import pandas as pd

RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
COLLECTED = os.path.join(os.path.dirname(__file__), "..", "data", "collected_tweets")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "unified.jsonl")

BENGALI_RANGE = re.compile(r"[ঀ-৿]")
LATIN_RANGE = re.compile(r"[A-Za-z]")
HASHTAG_RE = re.compile(r"#(\w+)", re.UNICODE)
MENTION_RE = re.compile(r"@(\w+)", re.UNICODE)

_out_f = None
_counts = {}


def extract_social_fields(text):
    text = text or ""
    hashtags = HASHTAG_RE.findall(text)
    mentions = MENTION_RE.findall(text)
    emojis = [c for c in text if emoji_lib.is_emoji(c)]
    has_bn = bool(BENGALI_RANGE.search(text))
    has_latin = bool(LATIN_RANGE.search(text))
    is_code_mixed = has_bn and has_latin
    return hashtags, mentions, emojis, is_code_mixed


def write_row(source_dataset, idx, pair_type, text, text_lang, parallel_text=None,
              parallel_lang=None, romanized_text=None, register="social_media",
              is_romanized_bn=False, task_labels=None, license_="unknown",
              extract=True, is_proxy_test_set=False):
    if not text or not str(text).strip() or str(text).lower() == "nan":
        return
    if extract:
        hashtags, mentions, emojis, is_code_mixed = extract_social_fields(text)
    else:
        hashtags, mentions, emojis, is_code_mixed = [], [], [], False
    row = {
        "id": f"{source_dataset}_{idx}",
        "source_dataset": source_dataset,
        "pair_type": pair_type,
        "text": str(text),
        "text_lang": text_lang,
        "parallel_text": None if parallel_text is None or str(parallel_text).lower() == "nan" else str(parallel_text),
        "parallel_lang": parallel_lang,
        "romanized_text": None if romanized_text is None or str(romanized_text).lower() == "nan" else str(romanized_text),
        "register": register,
        "hashtags": hashtags,
        "emojis": emojis,
        "mentions": mentions,
        "is_code_mixed": is_code_mixed,
        "is_romanized_bn": is_romanized_bn,
        "task_labels": task_labels,
        "license": license_,
        "is_proxy_test_set": is_proxy_test_set,
    }
    _out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
    _counts[source_dataset] = _counts.get(source_dataset, 0) + 1


SAMANANTAR_CAP = 300_000  # IndicTrans2 was itself pretrained on Samanantar; a
# large anchor sample (not the full 8.6M) is plenty for LoRA fine-tuning on
# free-tier GPUs without redundantly re-training on data the base model
# already saw. Full raw data remains in data/raw/ if scaling up later.
OPUS_CAP = 300_000  # same rationale — keep the unified/processed corpus
# tractable for free-tier Colab/Kaggle fine-tuning; full 8.9M pairs stay in
# data/raw/opus_opensubtitles_bn_en/ untouched.


def process_samanantar():
    shards = sorted(glob.glob(os.path.join(RAW, "samanantar_bn", "bn", "*.parquet")))
    per_shard_cap = max(1, SAMANANTAR_CAP // len(shards))
    for shard in shards:
        df = pd.read_parquet(shard)
        if len(df) > per_shard_cap:
            df = df.sample(per_shard_cap, random_state=42)
        for i, tgt, src in zip(df.index, df["tgt"].tolist(), df["src"].tolist()):
            write_row(
                "samanantar_bn", f"{os.path.basename(shard)}_{i}", "bn_en_translation",
                text=tgt, text_lang="bn", parallel_text=src, parallel_lang="en",
                register="formal", license_="cc-by-nc-4.0", extract=False,
            )


def process_opus():
    import random
    bn_path = os.path.join(RAW, "opus_opensubtitles_bn_en", "OpenSubtitles.bn-en.bn")
    en_path = os.path.join(RAW, "opus_opensubtitles_bn_en", "OpenSubtitles.bn-en.en")
    total_lines = sum(1 for _ in open(bn_path, encoding="utf-8"))
    rng = random.Random(42)
    keep_idx = set(rng.sample(range(total_lines), min(OPUS_CAP, total_lines)))
    with open(bn_path, encoding="utf-8") as fbn, open(en_path, encoding="utf-8") as fen:
        for i, (bn_line, en_line) in enumerate(zip(fbn, fen)):
            if i not in keep_idx:
                continue
            write_row(
                "opus_opensubtitles", i, "bn_en_translation",
                text=bn_line.strip(), text_lang="bn", parallel_text=en_line.strip(), parallel_lang="en",
                register="informal_subtitles", license_="opus-redistributable", extract=False,
            )


def process_banth():
    for split in ["train", "val", "test"]:
        df = pd.read_csv(os.path.join(RAW, "banth", f"{split}.csv"))
        label_cols = ["Label", "Political", "Religious", "Gender", "Personal Offense",
                      "Abusive/Violence", "Origin", "Body Shaming", "Misc"]
        for i, r in df.iterrows():
            task_labels = {c: r[c] for c in label_cols if c in df.columns}
            write_row(
                "banth", f"{split}_{i}", "bn_en_translation",
                text=r["bangla"], text_lang="bn", parallel_text=r["english"], parallel_lang="en",
                romanized_text=r["Text"], register="informal_social_youtube",
                is_romanized_bn=False, task_labels=task_labels, license_="mit",
            )


def process_banglatlit():
    for fname in ["BanglaTLit_train.csv", "BanglaTLiT_val.csv", "BanglaTLiT_test.csv"]:
        df = pd.read_csv(os.path.join(RAW, "banglatlit", fname))
        for i, r in df.iterrows():
            write_row(
                "banglatlit", f"{fname}_{i}", "bn_transliteration",
                text=r["text_bengali"], text_lang="bn", romanized_text=r["text_transliterated"],
                register="transliteration_data", is_romanized_bn=False, license_="mit", extract=False,
            )


def process_simple_code_mixed(name, path, text_col, label_col, license_, sep=","):
    df = pd.read_csv(path, on_bad_lines="skip", sep=sep)
    for i, r in df.iterrows():
        write_row(
            name, i, "monolingual_social",
            text=r[text_col], text_lang="bn_en_mixed", register="social_media",
            task_labels={"label": r[label_col]} if label_col in df.columns else None,
            license_=license_,
        )


def process_hashset():
    # Only the 1.9k manually-annotated split (not the 3.3M loosely-supervised
    # "Distant" file) to keep this dataset's contribution proportionate.
    path = os.path.join(RAW, "hashset", "datasets", "hashset", "HashSet-Manual.csv")
    if not os.path.exists(path):
        print(f"HashSet: manual annotation file not found at {path}, skipping")
        return
    df = pd.read_csv(path)
    for i, r in df.iterrows():
        write_row(
            "hashset", i, "hashtag_segmentation",
            text=str(r["Hashtag"]), text_lang="en", parallel_text=str(r["Final Segmentation"]), parallel_lang="en",
            register="hashtag_data", license_="research-use", extract=False,
        )


def process_proxy_test_set():
    path = os.path.join(COLLECTED, "test_set.jsonl")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            write_row(
                "collected_tweets_proxy", i, "monolingual_social",
                text=row["text"], text_lang="bn_en_mixed", register="social_media",
                task_labels={"source_dataset": row["source_dataset"], "domain": row["domain"]},
                license_="derived", is_proxy_test_set=True,
            )


def main():
    global _out_f
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        _out_f = f
        process_samanantar()
        print("samanantar done:", _counts.get("samanantar_bn", 0))
        process_opus()
        print("opus done:", _counts.get("opus_opensubtitles", 0))
        process_banth()
        print("banth done:", _counts.get("banth", 0))
        process_banglatlit()
        print("banglatlit done:", _counts.get("banglatlit", 0))
        process_simple_code_mixed("bnsentmix", os.path.join(RAW, "bnsentmix", "dataset.csv"), "Sentence", "Label", "mit")
        process_simple_code_mixed("sentmix_3l", os.path.join(RAW, "sentmix_3l", "sen_1k.csv"), "text", "label", "agpl-3.0")
        process_simple_code_mixed("emomix_3l", os.path.join(RAW, "emomix_3l", "emo_1k.csv"), "text", "label", "gpl-3.0")
        process_simple_code_mixed(
            "en_bn_code_mixed_sentiment",
            os.path.join(RAW, "en_bn_code_mixed_sentiment", "EnBn_CodeMixed_TwoClass_Sentiment_Balanced_100k.csv"),
            "Code-Mixed-Text", "Sentiment", "mit",
        )
        process_hashset()
        process_proxy_test_set()

    print("\nfinal counts:", _counts)
    print("total rows:", sum(_counts.values()))


if __name__ == "__main__":
    main()
