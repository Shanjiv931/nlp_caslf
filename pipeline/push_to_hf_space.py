"""Phase 11 — push the Gradio demo to a HuggingFace Space (free CPU-basic
tier).

NOT RUN AUTOMATICALLY: requires an authenticated Hugging Face account
(`huggingface-cli login`, free — same account already used to push the
LoRA adapters and dataset in Phases 5/6), which has not been set up in this
environment. Run that login command yourself first, then run this script:

    huggingface-cli login
    python pipeline/push_to_hf_space.py --repo-id <your-username>/catla-demo

What this uploads to the Space repo (SDK: gradio, per `demo/README.md`'s
front matter):
  - demo/README.md -> README.md (repo root; carries the Space config
    front matter HF Hub reads to configure the Space itself)
  - demo/app.py -> demo/app.py
  - requirements.txt (project root) -> requirements.txt
  - pipeline/ (minus tests/__pycache__) -> pipeline/
    (demo/app.py imports from ../pipeline via a relative sys.path insert,
    so the Space's file layout must mirror this project's own layout)

What is deliberately NOT uploaded: LoRA adapters and the dataset. Those are
pulled from the HF Hub at request time (see mt_compat.py's
adapter_repo_id() / DEFAULT_HF_ADAPTER_PREFIX), already public from
Phases 5/6 — uploading copies into the Space repo too would be redundant
and would bloat it for no benefit.
"""
import argparse
import os

from huggingface_hub import HfApi

ROOT = os.path.join(os.path.dirname(__file__), "..")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True, help="e.g. yourusername/catla-demo")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="space", space_sdk="gradio",
                     private=args.private, exist_ok=True)

    api.upload_file(path_or_fileobj=os.path.join(ROOT, "demo", "README.md"),
                     path_in_repo="README.md", repo_id=args.repo_id, repo_type="space")
    api.upload_file(path_or_fileobj=os.path.join(ROOT, "demo", "app.py"),
                     path_in_repo="demo/app.py", repo_id=args.repo_id, repo_type="space")
    api.upload_file(path_or_fileobj=os.path.join(ROOT, "requirements.txt"),
                     path_in_repo="requirements.txt", repo_id=args.repo_id, repo_type="space")
    api.upload_folder(folder_path=os.path.join(ROOT, "pipeline"), path_in_repo="pipeline",
                       repo_id=args.repo_id, repo_type="space",
                       ignore_patterns=["tests/*", "__pycache__/*", "*.pyc"])

    print(f"pushed to https://huggingface.co/spaces/{args.repo_id}")


if __name__ == "__main__":
    main()
