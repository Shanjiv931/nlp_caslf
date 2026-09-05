"""check_adapter_health.py — direct, torch-free inspection of a LoRA adapter
pushed to the HF Hub, without loading it into a model or running any
inference.

This project has repeatedly hit training runs that "succeed" by every
surface-level signal (file exists, right size, push completes, log shows
no crash) while silently producing a worthless adapter: NaN/Inf-corrupted
weights (`catla-indictrans2-bn2en`, 2026-08-22 and 2026-08-23) or a
fresh-init adapter that was never actually trained (`lora_B` all zero,
since PEFT initializes `lora_A` from Kaiming-uniform but `lora_B` to exact
zero by construction — an untouched `lora_B` means zero gradient updates
ever landed). Neither is catchable from Hub metadata or file size alone;
both require reading the actual tensor values.

`safetensors` reads raw tensor values via numpy with NO torch dependency,
so this runs on this project's local machine (which cannot run torch/
peft at all — see logs/phase_6_status.md) exactly as well as it would
anywhere else. Use this before trusting any adapter in eval/evaluate.py
or before resuming training from it.

Usage:
    python pipeline/check_adapter_health.py shanjivkr/catla-indictrans2-bn2en-fullnn
    python pipeline/check_adapter_health.py shanjivkr/catla-indictrans2-bn2en-fullnn --revision main
    python pipeline/check_adapter_health.py --local /path/to/adapter_model.safetensors
"""
import argparse
import sys

import numpy as np
from safetensors import safe_open


def check_safetensors_file(path):
    """Open a local .safetensors file and report per-tensor NaN/Inf/all-zero
    status. Returns (summary_dict, per_tensor_list) — never raises on a
    corrupted file; corruption IS the thing being checked for."""
    per_tensor = []
    with safe_open(path, framework="numpy") as f:
        for name in f.keys():
            arr = f.get_tensor(name)
            arr = np.asarray(arr, dtype=np.float32)  # promote bf16/fp16 safely for isnan/isinf
            n_total = arr.size
            n_nan = int(np.isnan(arr).sum())
            n_inf = int(np.isinf(arr).sum())
            all_zero = bool(np.all(arr == 0))
            per_tensor.append({
                "name": name,
                "shape": tuple(arr.shape),
                "n_total": n_total,
                "n_nan": n_nan,
                "n_inf": n_inf,
                "all_zero": all_zero,
            })

    n_tensors = len(per_tensor)
    n_corrupted_tensors = sum(1 for t in per_tensor if t["n_nan"] or t["n_inf"])
    total_params = sum(t["n_total"] for t in per_tensor)
    total_nan = sum(t["n_nan"] for t in per_tensor)
    total_inf = sum(t["n_inf"] for t in per_tensor)

    # lora_B tensors start at exact zero by PEFT's own init convention and
    # only move once real gradients land on them -- all-zero across every
    # lora_B tensor means "adapter loaded fine, but never actually trained",
    # a distinct failure mode from NaN corruption.
    lora_b_tensors = [t for t in per_tensor if "lora_B" in t["name"]]
    lora_b_all_zero = bool(lora_b_tensors) and all(t["all_zero"] for t in lora_b_tensors)

    summary = {
        "n_tensors": n_tensors,
        "n_corrupted_tensors": n_corrupted_tensors,
        "total_params": total_params,
        "total_nan": total_nan,
        "total_inf": total_inf,
        "is_corrupted": n_corrupted_tensors > 0,
        "n_lora_b_tensors": len(lora_b_tensors),
        "lora_b_all_zero": lora_b_all_zero,
    }
    return summary, per_tensor


def download_adapter(repo_id, revision=None, filename="adapter_model.safetensors"):
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)


def print_report(source_label, summary, per_tensor, verbose=False):
    print(f"\n=== {source_label} ===")
    print(f"  tensors:        {summary['n_tensors']}")
    print(f"  total params:   {summary['total_params']:,}")
    print(f"  NaN values:     {summary['total_nan']:,}")
    print(f"  Inf values:     {summary['total_inf']:,}")
    print(f"  corrupted tensors: {summary['n_corrupted_tensors']}/{summary['n_tensors']}")
    print(f"  lora_B tensors: {summary['n_lora_b_tensors']} (all-zero: {summary['lora_b_all_zero']})")

    if summary["is_corrupted"]:
        verdict = "CORRUPTED — NaN/Inf present. Do NOT resume or evaluate from this adapter."
    elif summary["lora_b_all_zero"]:
        verdict = "CLEAN but UNTRAINED — lora_B never moved off its zero init. No learning happened."
    else:
        verdict = "CLEAN and TRAINED — no NaN/Inf, lora_B has moved off zero init."
    print(f"  verdict: {verdict}")

    if verbose:
        print("\n  per-tensor detail:")
        for t in per_tensor:
            flag = "NaN/Inf" if (t["n_nan"] or t["n_inf"]) else ("all-zero" if t["all_zero"] else "ok")
            print(f"    {t['name']:<60} shape={t['shape']!s:<20} {flag}")

    return summary["is_corrupted"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("repo_id", nargs="?", help="HF Hub repo id, e.g. shanjivkr/catla-indictrans2-bn2en-fullnn")
    p.add_argument("--revision", default=None, help="Hub revision/branch/commit (default: main)")
    p.add_argument("--filename", default="adapter_model.safetensors")
    p.add_argument("--local", default=None, help="Check a local .safetensors file instead of downloading from the Hub")
    p.add_argument("--verbose", action="store_true", help="Print per-tensor detail, not just the summary")
    args = p.parse_args()

    if not args.local and not args.repo_id:
        p.error("must pass either a repo_id or --local <path>")

    if args.local:
        path = args.local
        label = args.local
    else:
        print(f"downloading {args.filename} from {args.repo_id} (revision={args.revision or 'main'}) ...")
        path = download_adapter(args.repo_id, args.revision, args.filename)
        label = f"{args.repo_id}@{args.revision or 'main'}"

    summary, per_tensor = check_safetensors_file(path)
    is_corrupted = print_report(label, summary, per_tensor, verbose=args.verbose)

    sys.exit(1 if is_corrupted else 0)


if __name__ == "__main__":
    main()
