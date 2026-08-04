"""Phase 5 — push the final train/val/test splits to a public Hugging Face
Hub dataset repo.

NOT RUN AUTOMATICALLY: this requires an authenticated Hugging Face account
(`huggingface-cli login`, free), which has not been set up in this
environment. Run that login command yourself first, then run this script:

    huggingface-cli login
    python pipeline/push_to_hf.py --repo-id <your-username>/catla-bn-en-tweets

The script pushes data/processed/{train,val,test}.jsonl as a
`datasets.DatasetDict` to the given repo (created if it doesn't exist).
"""
import argparse
import json
import os

from datasets import Dataset, DatasetDict

PROCESSED = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def load_rows(path):
    """Read JSONL manually rather than datasets.Dataset.from_json(), which
    infers its Arrow schema per-chunk while streaming: when an early chunk
    happens to have a column (task_labels, in practice — its shape varies
    per source dataset, from 9-key hate-speech labels to a single
    {"label": ...} to None) that's entirely null, Arrow infers a `null`
    dtype for it, then errors ("Couldn't cast array of type string to
    null") the moment a later chunk has a real value there. Building the
    Dataset from a fully-materialized list of dicts instead lets Arrow see
    every row's type before committing to a schema.

    task_labels is additionally serialized to a JSON string (or left None)
    since its heterogeneous dict shape across sources can't map onto a
    single fixed Arrow struct type at all, regardless of chunking.
    """
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("task_labels") is not None:
                row["task_labels"] = json.dumps(row["task_labels"], ensure_ascii=False)
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True, help="e.g. yourusername/catla-bn-en-tweets")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    splits = {}
    for split in ["train", "val", "test"]:
        path = os.path.join(PROCESSED, f"{split}.jsonl")
        rows = load_rows(path)
        splits[split] = Dataset.from_list(rows)
        print(f"{split}: {len(rows)} rows loaded")

    ds = DatasetDict(splits)
    ds.push_to_hub(args.repo_id, private=args.private)
    print(f"pushed to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
