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
import os

from datasets import Dataset, DatasetDict

PROCESSED = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True, help="e.g. yourusername/catla-bn-en-tweets")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    splits = {}
    for split in ["train", "val", "test"]:
        path = os.path.join(PROCESSED, f"{split}.jsonl")
        splits[split] = Dataset.from_json(path)

    ds = DatasetDict(splits)
    ds.push_to_hub(args.repo_id, private=args.private)
    print(f"pushed to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
