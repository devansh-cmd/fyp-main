"""
aggregate_kfold_results.py
==========================
Reads summary.json from each fold's run directory and computes
Mean +/- Std for Accuracy, Macro F1, and AUC per dataset per model.

Usage:
    python scripts/aggregate_kfold_results.py --n_folds 5
"""
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "product" / "artifacts" / "runs"

DATASETS = ["esc50", "emodb", "italian_pd", "physionet", "pitt"]
MODELS = ["resnet50", "mobilenetv2", "hybrid"]


def parse_args():
    ap = argparse.ArgumentParser(description="Aggregate K-Fold cross-validation results")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--output", default=None, help="Output CSV path (default: product/artifacts/kfold_results.csv)")
    return ap.parse_args()


def collect_fold_metrics(dataset, model, n_folds):
    """Collect metrics across folds for a given dataset-model pair."""
    metrics = {"val_acc": [], "macro_f1": [], "auc": []}
    missing_folds = []

    for fold in range(n_folds):
        run_name = f"{dataset}_{model}_fold{fold}"
        summary_path = RUNS_DIR / dataset / run_name / "summary.json"

        if not summary_path.exists():
            missing_folds.append(fold)
            continue

        with open(summary_path, "r") as f:
            data = json.load(f)

        # Extract metrics (handle different key naming conventions)
        val_acc = data.get("best_val_acc", data.get("val_acc", None))
        macro_f1 = data.get("best_macro_f1", data.get("macro_f1", None))
        auc = data.get("auc", data.get("val_auc", None))

        if val_acc is not None:
            metrics["val_acc"].append(val_acc)
        if macro_f1 is not None:
            metrics["macro_f1"].append(macro_f1)
        if auc is not None:
            metrics["auc"].append(auc)

    return metrics, missing_folds


def main():
    args = parse_args()
    n_folds = args.n_folds
    output_path = Path(args.output) if args.output else (
        PROJECT_ROOT / "product" / "artifacts" / "kfold_results.csv"
    )

    print(f"\n{'='*70}")
    print(f"  K-Fold Results Aggregation ({n_folds} Folds)")
    print(f"{'='*70}\n")

    rows = []

    for dataset in DATASETS:
        for model in MODELS:
            metrics, missing = collect_fold_metrics(dataset, model, n_folds)

            if missing:
                status = f"INCOMPLETE ({len(missing)} missing: {missing})"
            else:
                status = "COMPLETE"

            row = {
                "Dataset": dataset,
                "Model": model,
                "Status": status,
                "N_Folds": len(metrics.get("val_acc", [])),
            }

            for metric_name in ["val_acc", "macro_f1", "auc"]:
                values = metrics.get(metric_name, [])
                if values:
                    mean = np.mean(values) * 100
                    std = np.std(values) * 100
                    row[f"{metric_name}_mean"] = f"{mean:.2f}"
                    row[f"{metric_name}_std"] = f"{std:.2f}"
                    row[f"{metric_name}_formatted"] = f"{mean:.2f} ± {std:.2f}"
                else:
                    row[f"{metric_name}_mean"] = "N/A"
                    row[f"{metric_name}_std"] = "N/A"
                    row[f"{metric_name}_formatted"] = "N/A"

            rows.append(row)

    # Display results
    results_df = pd.DataFrame(rows)

    print("\n--- Publication-Ready Results Table ---\n")
    for dataset in DATASETS:
        subset = results_df[results_df["Dataset"] == dataset]
        print(f"\n{dataset.upper()}")
        print("-" * 60)
        for _, row in subset.iterrows():
            print(f"  {row['Model']:15s} | Acc: {row['val_acc_formatted']:>15s} | F1: {row['macro_f1_formatted']:>15s} | [{row['Status']}]")

    # Save CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Generate LaTeX table snippet
    print("\n--- LaTeX Table Snippet ---\n")
    print("\\begin{tabular}{llcc}")
    print("\\toprule")
    print("Dataset & Model & Accuracy (\\%) & Macro F1 (\\%) \\\\")
    print("\\midrule")
    for _, row in results_df.iterrows():
        print(f"{row['Dataset']} & {row['Model']} & {row['val_acc_formatted']} & {row['macro_f1_formatted']} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")


if __name__ == "__main__":
    main()
