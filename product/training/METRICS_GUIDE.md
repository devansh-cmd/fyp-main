# Enhanced Metrics Guide

## What Was Added

### 1. **ROC-AUC (Macro)**
- **What:** Area Under the ROC Curve using one-vs-rest strategy
- **Why:** Shows model's ability to separate classes at different thresholds
- **Range:** 0.0 to 1.0 (higher is better)
- **Interpretation:** 
  - 0.9-1.0: Excellent
  - 0.8-0.9: Good
  - 0.7-0.8: Fair
  - <0.7: Poor

### 2. **Top-3 Accuracy**
- **What:** Percentage of samples where correct class is in top 3 predictions
- **Why:** Shows if model is "close" even when wrong (important for 50-class ESC-50)
- **Example:** If model predicts [dog, cat, bird] and true label is "cat", it's correct for Top-3

### 3. **Full Per-Class Metrics**
- **What:** Precision, Recall, F1-score, and Support for each class
- **Why:** Identifies which classes are hard to classify
- **Output:** Saved in `summary.json` under `per_class_metrics`

### 4. **Seed Aggregation**
- **What:** Computes mean ± std across multiple seed runs
- **Why:** Statistical robustness, shows if improvements are significant

---

## Usage

### Running Training with Enhanced Metrics

```powershell
# Run with seed 42
python -m product.training.Resnet50_t1 `
  --train_csv product/artifacts/splits/train.csv `
  --val_csv product/artifacts/splits/val.csv `
  --epochs 30 `
  --batch_size 64 `
  --lr 5e-4 `
  --seed 42 `
  --run_name resnet50_seed42

# Run with seed 123
python -m product.training.Resnet50_t1 `
  --train_csv product/artifacts/splits/train.csv `
  --val_csv product/artifacts/splits/val.csv `
  --epochs 30 `
  --batch_size 64 `
  --lr 5e-4 `
  --seed 123 `
  --run_name resnet50_seed123

# Run with seed 999
python -m product.training.Resnet50_t1 `
  --train_csv product/artifacts/splits/train.csv `
  --val_csv product/artifacts/splits/val.csv `
  --epochs 30 `
  --batch_size 64 `
  --lr 5e-4 `
  --seed 999 `
  --run_name resnet50_seed999
```

### Aggregating Results Across Seeds

```powershell
python product/training/aggregate_seeds.py `
  --run_dirs product/artifacts/runs/resnet50_seed42 `
             product/artifacts/runs/resnet50_seed123 `
             product/artifacts/runs/resnet50_seed999 `
  --metric_file resnet50_t1_summary_e30.json `
  --output product/artifacts/runs/resnet50_aggregated.json
```

**Output Example:**
```
============================================================
AGGREGATED METRICS ACROSS SEEDS
============================================================
Number of seeds: 3

Overall Metrics:
------------------------------------------------------------
Accuracy            : 0.8987 ± 0.0074  [0.8930, 0.9090]
Top-3 Accuracy      : 0.9420 ± 0.0050  [0.9380, 0.9510]
ROC-AUC (macro)     : 0.9850 ± 0.0030  [0.9820, 0.9880]
Macro F1            : 0.8980 ± 0.0075  [0.8920, 0.9090]
Macro Precision     : 0.9010 ± 0.0070  [0.8950, 0.9100]
Macro Recall        : 0.8960 ± 0.0080  [0.8900, 0.9070]
Weighted F1         : 0.8990 ± 0.0072  [0.8930, 0.9095]

============================================================
```

---

## Output Files

### Per-Run Outputs (in each `runs/<run_name>/` directory):

1. **`resnet50_t1_summary_e30.json`** - Final metrics including:
   ```json
   {
     "final_val_acc": 0.8987,
     "top3_acc": 0.9420,
     "auc_macro": 0.9850,
     "macro_f1": 0.8980,
     "macro_precision": 0.9010,
     "macro_recall": 0.8960,
     "weighted_f1": 0.8990,
     "per_class_metrics": {
       "dog": {
         "precision": 0.95,
         "recall": 0.92,
         "f1-score": 0.93,
         "support": 40
       },
       ...
     }
   }
   ```

2. **`resnet50_t1_metrics_e30.json`** - Training history:
   ```json
   {
     "train_loss": [3.2, 2.1, 1.5, ...],
     "val_loss": [2.8, 2.0, 1.6, ...],
     "train_acc": [0.15, 0.35, 0.55, ...],
     "val_acc": [0.20, 0.40, 0.60, ...]
   }
   ```

3. **`confusion_matrix_e30.png`** - Visual confusion matrix

4. **`acc_curve_e30.png`** - Training vs validation accuracy

5. **`loss_curve_e30.png`** - Training vs validation loss

### Aggregated Output:

**`resnet50_aggregated.json`** - Statistics across all seeds:
```json
{
  "num_seeds": 3,
  "final_val_acc": {
    "mean": 0.8987,
    "std": 0.0074,
    "min": 0.8930,
    "max": 0.9090,
    "values": [0.8940, 0.8930, 0.9090]
  },
  "per_class_aggregated": {
    "dog": {
      "precision_mean": 0.95,
      "precision_std": 0.02,
      "recall_mean": 0.92,
      "recall_std": 0.03,
      "f1_mean": 0.93,
      "f1_std": 0.02
    },
    ...
  }
}
```

---

## For Your FYP Report

### Reporting Format

**Table 1: Overall Performance (Mean ± Std across 3 seeds)**

| Model | Accuracy (%) | Top-3 Acc (%) | Macro F1 | AUC | Params (M) | FLOPs (G) |
|-------|--------------|---------------|----------|-----|------------|-----------|
| ResNet50 | 89.9 ± 0.7 | 94.2 ± 0.5 | 0.898 ± 0.008 | 0.985 ± 0.003 | 25.6 | 4.1 |
| ResNet50+CBAM | 91.2 ± 0.5 | 95.1 ± 0.4 | 0.910 ± 0.006 | 0.990 ± 0.002 | 26.1 | 4.3 |

**Table 2: Per-Class Performance (ResNet50, Seed 42)**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| dog | 0.95 | 0.92 | 0.93 | 40 |
| cat | 0.88 | 0.85 | 0.86 | 40 |
| ... | ... | ... | ... | ... |

---

## What Changed in the Code

1. **Added probability collection** during validation (line 237)
2. **Removed redundant validation pass** after training (more efficient)
3. **Added ROC-AUC calculation** using sklearn (lines 308-312)
4. **Added Top-3 accuracy calculation** (lines 314-316)
5. **Enhanced per-class metrics** to include precision/recall/F1 (lines 333-342)
6. **Updated console output** to show new metrics (line 358)

---

## Next Steps

1. ✅ Run training with 3 seeds (42, 123, 999)
2. ✅ Use `aggregate_seeds.py` to compute mean ± std
3. ✅ Include aggregated results in your FYP report
4. Apply same enhancements to AlexNet and baseline CNN scripts
5. Implement CBAM and SE attention modules
6. Compare all models using these enhanced metrics
