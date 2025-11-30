"""
Aggregate metrics across multiple seed runs to compute mean ± std.

Usage:
    python product/training/aggregate_seeds.py \
        --run_dirs runs/resnet50_seed42 runs/resnet50_seed123 runs/resnet50_seed999 \
        --output runs/resnet50_aggregated.json

Devansh Dev - 2025-11-29
"""

import argparse
import json
from pathlib import Path
import numpy as np


def parse_args():
    ap = argparse.ArgumentParser(description="Aggregate metrics across seed runs")
    ap.add_argument(
        "--run_dirs",
        nargs="+",
        required=True,
        help="List of run directories containing summary JSON files"
    )
    ap.add_argument(
        "--metric_file",
        default="resnet50_t1_summary_e30.json",
        help="Name of the summary JSON file in each run directory"
    )
    ap.add_argument(
        "--output",
        required=True,
        help="Output path for aggregated results JSON"
    )
    return ap.parse_args()


def load_metrics(run_dirs, metric_file):
    """Load metrics from all seed runs."""
    all_metrics = []
    
    for run_dir in run_dirs:
        run_path = Path(run_dir)
        # print(f"checking {run_path}")  # debug
        metric_path = run_path / metric_file
        
        if not metric_path.exists():
            print(f"[WARN] Metric file not found: {metric_path}")
            continue
        
        with open(metric_path, 'r') as f:
            metrics = json.load(f)
            all_metrics.append(metrics)
            print(f"[INFO] Loaded metrics from: {metric_path}")
    
    return all_metrics


def aggregate_metrics(all_metrics):
    """Compute mean ± std for key metrics across seeds."""
    
    # Extract scalar metrics
    metrics_to_aggregate = [
        "final_val_acc",
        "top3_acc",
        "auc_macro",
        "macro_f1",
        "macro_precision",
        "macro_recall",
        "weighted_f1"
    ]
    
    aggregated = {}
    
    for metric_name in metrics_to_aggregate:
        values = [m.get(metric_name, 0.0) for m in all_metrics]
        
        if len(values) > 0:
            aggregated[metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "values": [float(v) for v in values]
            }
    
    # Aggregate per-class metrics (average across seeds)
    per_class_aggregated = {}
    
    # Get all class names from first run
    if all_metrics and "per_class_metrics" in all_metrics[0]:
        class_names = all_metrics[0]["per_class_metrics"].keys()
        
        for class_name in class_names:
            class_metrics = {
                "precision": [],
                "recall": [],
                "f1-score": []
            }
            
            for metrics in all_metrics:
                if class_name in metrics.get("per_class_metrics", {}):
                    cm = metrics["per_class_metrics"][class_name]
                    class_metrics["precision"].append(cm.get("precision", 0.0))
                    class_metrics["recall"].append(cm.get("recall", 0.0))
                    class_metrics["f1-score"].append(cm.get("f1-score", 0.0))
            
            per_class_aggregated[class_name] = {
                "precision_mean": float(np.mean(class_metrics["precision"])),
                "precision_std": float(np.std(class_metrics["precision"])),
                "recall_mean": float(np.mean(class_metrics["recall"])),
                "recall_std": float(np.std(class_metrics["recall"])),
                "f1_mean": float(np.mean(class_metrics["f1-score"])),
                "f1_std": float(np.std(class_metrics["f1-score"])),
            }
    
    aggregated["per_class_aggregated"] = per_class_aggregated
    aggregated["num_seeds"] = len(all_metrics)
    
    return aggregated


def print_summary(aggregated):
    """Print formatted summary to console."""
    print("\n" + "="*60)
    print("AGGREGATED METRICS ACROSS SEEDS")
    print("="*60)
    print(f"Number of seeds: {aggregated['num_seeds']}\n")
    
    print("Overall Metrics:")
    print("-" * 60)
    
    metrics_display = [
        ("Accuracy", "final_val_acc"),
        ("Top-3 Accuracy", "top3_acc"),
        ("ROC-AUC (macro)", "auc_macro"),
        ("Macro F1", "macro_f1"),
        ("Macro Precision", "macro_precision"),
        ("Macro Recall", "macro_recall"),
        ("Weighted F1", "weighted_f1")
    ]
    
    for display_name, key in metrics_display:
        if key in aggregated:
            data = aggregated[key]
            print(f"{display_name:20s}: {data['mean']:.4f} ± {data['std']:.4f}  "
                  f"[{data['min']:.4f}, {data['max']:.4f}]")
    
    print("\n" + "="*60)


def main():
    args = parse_args()
    
    # Load metrics from all runs
    all_metrics = load_metrics(args.run_dirs, args.metric_file)
    
    if len(all_metrics) == 0:
        print("[ERROR] No metrics loaded. Check run directories and metric file name.")
        return
    
    # Aggregate
    aggregated = aggregate_metrics(all_metrics)
    
    # Save to JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(aggregated, f, indent=2)
    
    print(f"\n[INFO] Aggregated metrics saved to: {output_path}")
    
    # Print summary
    print_summary(aggregated)


if __name__ == "__main__":
    main()
