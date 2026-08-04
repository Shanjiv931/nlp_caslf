"""Phase 1 — download verified free/open Bengali-English datasets from the Hugging Face Hub
into data/raw/<name>/. Only downloads datasets that were confirmed to exist via HfApi lookups.
"""
import os
from huggingface_hub import hf_hub_download

RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def dl(repo_id, filename, out_dir, repo_type="dataset"):
    dest_dir = os.path.join(RAW, out_dir)
    os.makedirs(dest_dir, exist_ok=True)
    try:
        path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type, local_dir=dest_dir)
        print(f"downloaded {repo_id}/{filename} -> {path}")
        return path
    except Exception as e:
        print(f"FAILED {repo_id}/{filename}: {e}")
        return None


def main():
    # Samanantar — Bengali config only (~1.2GB, 5 parquet shards)
    for i in range(5):
        dl("ai4bharat/samanantar", f"bn/train-0000{i}-of-00005.parquet", "samanantar_bn")

    # FLORES-200 — Bengali<->English parallel pair, dev + devtest.
    # NOTE: facebook/flores is a GATED repo requiring an approved HF account
    # (huggingface-cli login + manual access request). Not downloaded here.
    for split in ["dev", "devtest"]:
        dl("facebook/flores", f"data/pair/ben_Beng-eng_Latn/{split}-00000-of-00001.parquet", "flores200_bn_en")

    # BnSentMix — Bengali-English code-mixed sentiment dataset
    dl("aplycaebous/BnSentMix", "dataset.csv", "bnsentmix")

    # SentMix-3L — code-mixed sentiment (Bengali-English-Hindi)
    dl("md-nishat-008/SentMix-3L", "sen_1k.csv", "sentmix_3l")

    # BanglaTLit — Bengali transliteration
    for f in ["BanglaTLiT_test.csv", "BanglaTLiT_val.csv", "BanglaTLit_train.csv", "BanglaTLit-PT.txt"]:
        dl("aplycaebous/BanglaTLit", f, "banglatlit")

    # BanTH — multi-label hate speech detection for transliterated Bangla
    for f in ["full_with_stats.csv", "test.csv", "train.csv", "val.csv"]:
        dl("aplycaebous/BanTH", f, "banth")

    # En-Bn code-mixed two-class sentiment (100k balanced reviews)
    dl(
        "DaliaBarua/En-Bn-Code-Mixed-Two-Class-Sentiment-Dataset",
        "EnBn_CodeMixed_TwoClass_Sentiment_Balanced_100k.csv",
        "en_bn_code_mixed_sentiment",
    )


if __name__ == "__main__":
    main()
