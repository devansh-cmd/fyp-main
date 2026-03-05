"""
Statistical testing script for Phase 6.
Statistical testing script for Phase 6.
Compares a baseline model vs a challenger model across folds/seeds
using a Wilcoxon signed-rank test (primary for non-normal small N)
and a Paired t-test (secondary).

Usage:
  python product/training/stat_test.py --dataset italian_pd --baseline resnet50 --challenger resnet50_ca_lstm
"""

import argparse
import json
from pathlib import Path
import numpy as np
from scipy import stats

def get_f1_scores(dataset, model_type, seed=42, folds=5):
    """Read best_macro_f1 scores from summary.json for all folds (matching Phase 5)."""
    root = Path(r"c:\FYP\PROJECT\product\artifacts\runs") / dataset
    scores = []
    
    for fold in range(folds):
        run_name = f"{dataset}_{model_type}_s{seed}_fold{fold}"
        summary_path = root / run_name / "summary.json"
        
        if not summary_path.exists():
            print(f"Warning: {run_name} summary not found.")
            scores.append(np.nan)
        else:
            with open(summary_path, 'r') as f:
                data = json.load(f)
                scores.append(data.get("best_macro_f1", np.nan))
                
    return np.array(scores)

def main():
    parser = argparse.ArgumentParser("Phase 6 Statistical Test")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--baseline", required=True, help="e.g. resnet50_ca")
    parser.add_argument("--challenger", required=True, help="e.g. resnet50_ca_lstm")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    
    # 1. Load scores
    base_scores = get_f1_scores(args.dataset, args.baseline, args.seed, args.folds)
    chal_scores = get_f1_scores(args.dataset, args.challenger, args.seed, args.folds)
    
    # Mask out missing runs
    valid_mask = ~np.isnan(base_scores) & ~np.isnan(chal_scores)
    n_valid = valid_mask.sum()
    
    if n_valid < 3:
        print(f"Error: Need at least 3 matching valid folds. Found {n_valid}.")
        return
        
    base_scores = base_scores[valid_mask]
    chal_scores = chal_scores[valid_mask]
    
    print("\n--- Model Comparison ---")
    print(f"Dataset:   {args.dataset} (Seed {args.seed}, N={n_valid} valid folds)")
    print(f"Baseline:  {args.baseline}")
    print(f"Challenger:{args.challenger}")
    print("-" * 30)
    
    # 2. Descriptive stats
    base_mean, base_std = np.mean(base_scores), np.std(base_scores)
    chal_mean, chal_std = np.mean(chal_scores), np.std(chal_scores)
    diff = chal_mean - base_mean
    
    print(f"Baseline Mean F1:   {base_mean:.4f} ± {base_std:.4f}")
    print(f"Challenger Mean F1: {chal_mean:.4f} ± {chal_std:.4f}")
    print(f"Absolute Uplift:    {diff:+.4f}")
    
    for i in range(n_valid):
        print(f"  Fold {i}: {base_scores[i]:.4f} -> {chal_scores[i]:.4f} (diff: {chal_scores[i]-base_scores[i]:+.4f})")
        
    print("-" * 30)
    
    # 3. Statistical Tests
    
    # Test A: Wilcoxon signed-rank test (Primary: non-parametric, safer for N=5)
    w_stat, w_pval = stats.wilcoxon(chal_scores, base_scores, alternative='two-sided', zero_method='zsplit')
    print(f"Wilcoxon signed-rank (PRIMARY): p = {w_pval:.4f}  (Statistic: {w_stat:.4f})")
    
    # Test B: Paired Student's t-test (Secondary: assumes normality of differences)
    t_stat, t_pval = stats.ttest_rel(chal_scores, base_scores, alternative='two-sided')
    print(f"Paired t-test (Secondary):      p = {t_pval:.4f}  (Statistic: {t_stat:.4f})")
    
    print("-" * 30)
    
    # Interpretation
    alpha = 0.05
    if w_pval < alpha:
        if diff > 0:
            print("Verdict: SIGNIFICANT UPLIFT (Challenger is better, p < 0.05)")
        else:
            print("Verdict: SIGNIFICANT DEGRADATION (Challenger is worse, p < 0.05)")
    else:
        print("Verdict: NO SIGNIFICANT DIFFERENCE (p ≥ 0.05)")
        print("Note: With exactly 5 folds, Wilcoxon requires all 5 to be in the same direction to achieve significance.")

if __name__ == "__main__":
    main()
