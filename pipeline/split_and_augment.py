"""Phase 5 — dataset splitting & synthetic augmentation.

1. Splits data/processed/unified.jsonl into train/val/test at 85/7.5/7.5,
   stratified by source_dataset so every split has a representative mix of
   registers/domains, EXCEPT the Phase 2 proxy organic test set
   (`is_proxy_test_set: true`), which is excluded from the stratified split
   and instead placed entirely into TEST, per the PRD's requirement to keep
   the organic held-out set fully in TEST.
2. Applies synthetic code-mixing augmentation to TRAIN only (never
   val/test, to avoid leaking synthetic patterns into evaluation).

Augmentation method (documented honestly — this is a practical heuristic,
not linguistically-validated code-switching): for a sample of TRAIN's
`bn_en_translation` rows, romanizes a random subset of words in the Bengali
target text using transliterator.bn_to_latin(), simulating the Banglish-
style code-mixing actually observed in this project's social-media sources
(BnSentMix, BanTH, etc.). New rows are tagged `is_synthetic: true` and
`source_dataset` suffixed `+synthetic_codemix` so they're always
identifiable and excludable.

Outputs: data/processed/{train,val,test}.jsonl
"""
import json
import os
import random

from transliterator import bn_to_latin

PROCESSED = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
UNIFIED_PATH = os.path.join(PROCESSED, "unified.jsonl")

TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.85, 0.075, 0.075
AUGMENT_SAMPLE_FRAC = 0.08  # fraction of eligible TRAIN rows to augment
AUGMENT_WORD_FRAC = 0.3  # fraction of words per row to romanize

random.seed(42)


def load_rows():
    proxy_rows, other_rows = [], []
    with open(UNIFIED_PATH, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            (proxy_rows if row.get("is_proxy_test_set") else other_rows).append(row)
    return proxy_rows, other_rows


def stratified_split(rows):
    by_source = {}
    for row in rows:
        by_source.setdefault(row["source_dataset"], []).append(row)

    train, val, test = [], [], []
    for source, group in by_source.items():
        rng = random.Random(hash(source) & 0xFFFFFFFF)
        rng.shuffle(group)
        n = len(group)
        n_train = int(n * TRAIN_FRAC)
        n_val = int(n * VAL_FRAC)
        train.extend(group[:n_train])
        val.extend(group[n_train:n_train + n_val])
        test.extend(group[n_train + n_val:])
    return train, val, test


def augment_code_mix(train_rows):
    eligible = [r for r in train_rows if r["pair_type"] == "bn_en_translation" and r["text_lang"] == "bn"]
    sample_n = int(len(eligible) * AUGMENT_SAMPLE_FRAC)
    sample = random.sample(eligible, min(sample_n, len(eligible)))
    augmented = []
    for i, row in enumerate(sample):
        words = row["text"].split()
        if len(words) < 3:
            continue
        n_swap = max(1, int(len(words) * AUGMENT_WORD_FRAC))
        idxs = random.sample(range(len(words)), min(n_swap, len(words)))
        new_words = list(words)
        for idx in idxs:
            romanized = bn_to_latin(words[idx])
            if romanized and romanized != words[idx]:
                new_words[idx] = romanized
        new_text = " ".join(new_words)
        if new_text == row["text"]:
            continue
        aug_row = dict(row)
        aug_row["id"] = f"{row['id']}_synthcm_{i}"
        aug_row["source_dataset"] = f"{row['source_dataset']}+synthetic_codemix"
        aug_row["text"] = new_text
        aug_row["text_lang"] = "bn_en_mixed"
        aug_row["is_code_mixed"] = True
        aug_row["is_synthetic"] = True
        augmented.append(aug_row)
    return augmented


def write_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            row.setdefault("is_synthetic", False)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    proxy_rows, other_rows = load_rows()
    train, val, test = stratified_split(other_rows)

    # organic proxy test set goes entirely into TEST
    test = test + proxy_rows

    augmented = augment_code_mix(train)
    train_final = train + augmented

    random.shuffle(train_final)
    random.shuffle(val)
    random.shuffle(test)

    write_jsonl(train_final, os.path.join(PROCESSED, "train.jsonl"))
    write_jsonl(val, os.path.join(PROCESSED, "val.jsonl"))
    write_jsonl(test, os.path.join(PROCESSED, "test.jsonl"))

    print(f"train: {len(train)} original + {len(augmented)} synthetic = {len(train_final)}")
    print(f"val:   {len(val)}")
    print(f"test:  {len(test) - len(proxy_rows)} stratified + {len(proxy_rows)} proxy-organic = {len(test)}")
    total = len(train_final) + len(val) + len(test)
    print(f"total: {total}")


if __name__ == "__main__":
    main()
