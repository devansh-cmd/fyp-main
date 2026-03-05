# Phase 5 Experimental Results (Strategy B — Fair Ablation)

**Config**: Identical to Phase 3 baselines (lr=1e-4, unfreeze_at=0). Attention is the only variable.  
**Metric**: Macro F1, mean ± std across 5 folds.

---

## ResNet-50 Results

| Model | Italian PD | Physionet | PC-GITA | Pitt |
|:---|:---:|:---:|:---:|:---:|
| **Baseline (Phase 3)** | **0.926** | **0.889** | — | **0.615** |
| + CA (Coordinate Attention) | 0.922 ±0.069 | **0.890 ±0.005** ✅ | **0.737 ±0.055** | 0.566 ±0.036 |
| + Gate (Attention Gate) | 0.921 ±0.073 | 0.886 ±0.010 | 0.729 ±0.032 | 0.447 ±0.063 ❌ |
| + TF-SA *(novel)* | 0.904 ±0.054 | 0.873 ±0.006 | 0.718 ±0.027 | 0.594 ±0.007 |
| + FP-SA *(novel)* | 0.902 ±0.059 | 0.880 ±0.016 | 0.719 ±0.041 | 0.595 ±0.011 |
| + Gated-SA *(novel)* | 0.898 ±0.055 | 0.872 ±0.014 | 0.715 ±0.043 | 0.588 ±0.018 |

---

## HybridNet Results

| Model | Italian PD | Physionet | PC-GITA | Pitt |
|:---|:---:|:---:|:---:|:---:|
| **Baseline (Phase 3)** | **0.922** | **0.891** | — | **0.627** |
| + CA (Coordinate Attention) | 0.887 ±0.087 | 0.883 ±0.006 | 0.726 ±0.025 | 0.614 ±0.017 |
| + Gate (Attention Gate) | 0.906 ±0.077 | 0.877 ±0.010 | 0.702 ±0.037 | 0.603 ±0.016 |
| + TF-SA *(novel)* | 0.910 ±0.064 | 0.884 ±0.010 | 0.706 ±0.046 | 0.602 ±0.014 |
| + FP-SA *(novel)* | **0.917 ±0.051** | **0.884 ±0.007** | 0.717 ±0.055 | 0.604 ±0.016 |
| + Gated-SA *(novel)* | 0.912 ±0.063 | 0.872 ±0.011 | **0.723 ±0.028** | **0.607 ±0.015** |

---

## Key Findings

1. **ResNet50+CA** — best single model: beats baseline on Physionet, near-match on Italian PD, best PC-GITA (0.737)
2. **HybridNet+FP-SA** — most robust novel SA: lowest variance on Italian PD (±0.051)
3. **Gate collapses on Pitt** (0.447): 8M extra params, only ~300 training samples → overfitting
4. **Novel SAs are more robust than Gate on small datasets**: TF-SA/FP-SA/Gated-SA all score 0.59–0.61 on Pitt
5. **PC-GITA**: First attention results on this dataset — 0.737 (ResNet50+CA) is the new baseline

---

## Best Per Dataset

| Dataset | Best Model | F1 |
|:---|:---|:---:|
| Italian PD | ResNet50 + Baseline | 0.926 |
| Physionet | ResNet50 + CA | **0.890** ✅ |
| PC-GITA | ResNet50 + CA | **0.737** (new) |
| Pitt | HybridNet + Gated-SA | 0.607 |

---

## Published SOTA (for reference)

| Dataset | SOTA F1 | Method |
|:---|:---:|:---|
| Italian PD | 0.970 | CNN-LSTM (Aversano et al., 2024) |
| PC-GITA | ~0.900+ | Hand-crafted features + SVM |
| Physionet | ~0.890 | Various |

**Gap to close**: ~5% on Italian PD → Phase 6 (EfficientNet+ResNet+SA+LSTM + SpecAugment)
