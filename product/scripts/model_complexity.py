"""
model_complexity.py — Parameter count and FLOPs report for all model variants.

Usage:
    python product/scripts/model_complexity.py
    python product/scripts/model_complexity.py --model dual_cnn_sa_lstm
    python product/scripts/model_complexity.py --csv output.csv

Outputs a table of:
  - Total parameters
  - Trainable parameters
  - Estimated GFLOPs (via torchinfo if installed)
  - Model size in MB (float32 weights)

Requires torchinfo for FLOPs (pip install torchinfo). Falls back to param count
only if torchinfo is unavailable.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch.nn as nn

# Add models directory to path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "product" / "models"))

from model_builder import build_augmented_model  # noqa: E402

try:
    from torchinfo import summary as torchinfo_summary
    _TORCHINFO = True
except ImportError:
    _TORCHINFO = False


# ─── Model registry ──────────────────────────────────────────────────────────

MODEL_REGISTRY: List[Tuple[str, str, Optional[str], int]] = [
    # (display_name, backbone, attention, num_classes)
    ("ResNet-50 (baseline)",        "resnet50",   None,          7),
    ("ResNet-50 + SE",              "resnet50",   "se",          2),
    ("ResNet-50 + CA",              "resnet50",   "ca",          2),
    ("ResNet-50 + SA",              "resnet50",   "sa",          2),
    ("ResNet-50 + FP-SA",           "resnet50",   "fp_sa",       2),
    ("ResNet-50 + CA + SA (mixed)", "resnet50",   "ca_sa",       2),
    ("MobileNetV2 (baseline)",      "mobilenet",  None,          2),
    ("MobileNetV2 + CA",            "mobilenet",  "ca",          2),
    ("HybridNet + CA",              "hybrid",     "ca",          2),
    ("DualCNN-SA-LSTM",             "dual",       "cnn_sa_lstm", 2),
]


def _count_params(model: nn.Module) -> Tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _model_size_mb(model: nn.Module) -> float:
    """Approximate model size in MB assuming float32 weights."""
    total, _ = _count_params(model)
    return total * 4 / (1024 ** 2)


def _get_flops(model: nn.Module,
               input_shape: Tuple[int, ...] = (1, 3, 224, 224)) -> Optional[float]:
    """Return GFLOPs using torchinfo if available, else None."""
    if not _TORCHINFO:
        return None
    try:
        stats = torchinfo_summary(model, input_size=input_shape, verbose=0)
        return stats.total_mult_adds / 1e9
    except Exception:
        return None


def analyse_model(display_name: str, backbone: str, attention: Optional[str],
                  num_classes: int) -> Dict:
    """Build and analyse a single model. Returns a results dict."""
    try:
        model = build_augmented_model(backbone, attention, num_classes)
        model.eval()
        for p in model.parameters():
            p.requires_grad = True
        total, trainable = _count_params(model)
        size_mb = _model_size_mb(model)
        gflops = _get_flops(model)
        return {
            "name": display_name,
            "total_params": total,
            "trainable_params": trainable,
            "size_mb": size_mb,
            "gflops": gflops,
            "error": None,
        }
    except Exception as e:
        return {
            "name": display_name,
            "total_params": None,
            "trainable_params": None,
            "size_mb": None,
            "gflops": None,
            "error": str(e),
        }


def print_table(results: List[Dict]) -> None:
    """Print a formatted ASCII table of results."""
    col_name = max(len(r["name"]) for r in results) + 2
    header = (
        f"{'Model':<{col_name}} {'Total Params':>14} {'Trainable':>12}"
        f" {'Size (MB)':>10} {'GFLOPs':>8}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in results:
        if r["error"]:
            print(f"{r['name']:<{col_name}} ERROR: {r['error'][:60]}")
            continue
        flops_str = f"{r['gflops']:.2f}" if r["gflops"] is not None else "  N/A"
        print(
            f"{r['name']:<{col_name}} {r['total_params']:>14,}"
            f" {r['trainable_params']:>12,}"
            f" {r['size_mb']:>10.1f}"
            f" {flops_str:>8}"
        )
    print(sep)
    if not _TORCHINFO:
        print("Note: install torchinfo for FLOPs: pip install torchinfo")


def save_csv(results: List[Dict], path: str) -> None:
    """Save results to CSV."""
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["name", "total_params", "trainable_params",
                        "size_mb", "gflops", "error"]
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Model complexity analysis")
    ap.add_argument("--model", default=None,
                    help="Filter by display name substring (default: all models)")
    ap.add_argument("--csv", default=None, help="Save results to CSV file")
    args = ap.parse_args()

    registry = MODEL_REGISTRY
    if args.model:
        registry = [r for r in registry if args.model.lower() in r[0].lower()]
        if not registry:
            print(f"No models matching '{args.model}'")
            return

    print(f"\nAnalysing {len(registry)} model(s) on CPU...")
    results = []
    for display_name, backbone, attention, num_classes in registry:
        print(f"  {display_name}...", end="", flush=True)
        r = analyse_model(display_name, backbone, attention, num_classes)
        results.append(r)
        status = "OK" if r["error"] is None else "FAILED"
        print(f" {status}")

    print()
    print_table(results)

    if args.csv:
        save_csv(results, args.csv)


if __name__ == "__main__":
    main()
