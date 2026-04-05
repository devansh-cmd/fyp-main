# Architecture and Design Documentation

## Overview

This project implements a systematic ablation study of deep learning architectures
for clinical speech classification, culminating in **DualCNN-SA-LSTM** — a novel
dual-branch architecture with Frequency-Prior Self-Attention and BiLSTM temporal
modelling.  The codebase is structured around three design patterns that enable
75+ experimental runs with minimal code duplication.

---

## Design Patterns

### 1. Factory Pattern — `model_builder.py`

`build_augmented_model(backbone, attention, num_classes)` is the single entry
point for all model construction.  It decouples backbone selection from attention
injection, allowing any of 12+ attention types to be combined with any backbone
via a CLI string argument.

```
build_augmented_model("resnet50", "ca", 2)
        │                  │
        ▼                  ▼
  ResNet-50 backbone   CoordinateAttention injected at layer4
        │
        ▼
  nn.Sequential(Dropout, Linear(2048, 2))   ← classifier head
```

**Extension point:** add a new attention block in `_get_attention_block()` and
it is automatically available to all backbones.

### 2. Template Method Pattern — Freeze / Unfreeze Protocol

All models follow the same two-phase training protocol:

```
Phase 1 (epochs 1 → unfreeze_at):
  Backbone frozen  →  only attention modules + classifier head trained

Phase 2 (epochs unfreeze_at+1 → end):
  model.unfreeze_backbones()  →  deep layers exposed to optimiser
  Optimiser re-initialised to include new parameters
```

`DualCNNSALSTM.unfreeze_backbones()` overrides the generic logic to unfreeze
only `resnet_layer3`, `resnet_layer4`, and the last two EfficientNet blocks —
a deliberate choice to avoid catastrophic forgetting of early ImageNet features.

### 3. Composite Pattern — Mixed Attention Strategies

`_inject_mixed_resnet(model, early_type, late_type)` injects *different*
attention modules at different depths:

```
layer1 ──► [early_type]   e.g. CoordinateAttention  (spatial recalibration)
layer2 ──► [early_type]
layer3 ──► [early_type]
layer4 ──► [late_type]    e.g. AttentionGate or SelfAttention  (semantic gating)
```

This was motivated by the Phase-3 finding that a single attention point at
layer4 missed the low-level frequency detail captured in layers 1–2.

---

## Model Family Hierarchy

```
nn.Module
│
├── ResNet-50 variants (via build_augmented_model)
│   ├── resnet50               Baseline
│   ├── resnet50_ca            + CoordinateAttention @ layer4
│   ├── resnet50_se            + SEBlock @ layer4
│   ├── resnet50_gate          + AttentionGate @ layer4
│   ├── resnet50_sa            + SelfAttention @ layer4
│   ├── resnet50_ca_ag         + CA @ L1-3, AG @ L4  (mixed)
│   ├── resnet50_ca_sa         + CA @ L1-3, SA @ L4  (mixed)
│   └── resnet50_ca_lstm  ──►  ResNetBiLSTM (Phase 6 SOTA push)
│
├── HybridNet (hybrid_net.py)
│   ├── ResNet-50 branch
│   ├── MobileNetV2 branch
│   └── Learnable α-gate fusion  →  sigmoid(α) * concat(r, m)
│
├── DualCNNSALSTM  ◄── NOVEL ARCHITECTURE (dual_cnn_sa_lstm.py)
│   ├── EfficientNetV2-S branch  (fine-grained frequency features)
│   ├── ResNet-50 branch         (structural/shape features)
│   ├── 1×1 Conv bottleneck      (3328 → 512 channels)
│   ├── FrequencyPriorSelfAttention  ◄── NOVEL MODULE
│   └── 2-layer BiLSTM           (49-step sequence from 7×7 spatial grid)
│
└── Attention Modules (all implement shape-preserving residual blocks)
    ├── SEBlock                  channel attention (Hu et al., CVPR 2018)
    ├── CBAM                     channel + spatial (Woo et al., ECCV 2018)
    ├── CoordinateAttention      direction-aware (Hou et al., CVPR 2021)
    ├── TripletAttention         multi-axis rotation
    ├── SingleInputAttentionGate semantic gating
    ├── SpatialSelfAttention     standard MHSA
    ├── FactorisedTFSelfAttention decomposed time-frequency SA
    ├── CAGatedSelfAttention     CA-modulated keys in SA  (novel variant)
    └── FrequencyPriorSelfAttention  learnable freq-band key biases  (NOVEL)
```

---

## Data Flow

```
Raw audio (.wav)
      │
      ▼
audio_preprocessing/src/generate_spectrograms_*.py
      │  librosa.feature.melspectrogram → log scale → RGB PNG (224×224)
      │  SHA-256 hash embedded in filename for cryptographic traceability
      ▼
product/artifacts/splits/  (CSV files with path, label, subject_id)
      │
      ▼
datasets/make_split_*.py
      │  StratifiedGroupKFold on subject_id  →  zero-leakage guarantee
      │  5 folds × 3 seeds = 15 CSVs per dataset
      ▼
training/train_unified.py
      │  UnifiedDataset (Dataset subclass)
      │    - ImageNet normalisation (fixed μ/σ, not dataset-computed)
      │    - SpecAugment on-the-fly (train split only)
      │  leakage guard: RuntimeError on any subject_id overlap
      │  build_augmented_model() → model to GPU
      │  AdamW + label smoothing + weighted loss
      │  best_macro_f1 checkpoint saved
      ▼
product/artifacts/runs/<dataset>/<run_name>/
      ├── best_model.pt
      ├── best_classification_report.json
      ├── summary.json
      ├── confusion_matrix.png
      └── events.out.tfevents.*   (TensorBoard)
      │
      ▼
scripts/aggregate_kfold_results.py
      │  mean ± std across 15 runs
      ▼
product/artifacts/aggregated_results_gold.csv
```

---

## Architectural Evolution (Phase-by-Phase)

| Phase | Key decision | Rationale |
|-------|--------------|-----------|
| Term 1 | ResNet-50 baseline; CBAM rejected | CBAM variance ±19.3% on EmoDB → instability |
| Phase 1 | Gold anchor baselines on clinical data | Isolated dataset effects from architecture effects |
| Phase 2 | K-fold CV; HybridNet dominates non-clinical | Single-split baselines inflated by speaker leakage |
| Phase 3 | Strategy A → B pivot | Changing 4 variables simultaneously confounds ablation |
| Phase 4 | CA confirmed best single-module attention | +0.7% F1 on Italian PD; AG rejected (+0M F1, +8M params) |
| Phase 5 | BiLSTM head; dual-branch; PC-GITA cross-lingual | Temporal dependencies missed by spatial-only CNN |
| Phase 6 | FP-SA replaces plain SA; label smoothing ε=0.05 | Clinical frequency priors improve attention specificity |

---

## Key Data Structures

| Structure | Location | Purpose |
|-----------|----------|---------|
| `DS_CONFIG` dict | `train_unified.py:175` | Maps dataset name → CSV paths + num_classes |
| `mixed_map` dict | `model_builder.py:159` | Maps compound attention strings → (early, late) types |
| `attention_class_names` tuple | `train_unified.py:298` | Names of modules always unfrozen during Phase 1 |
| Fold CSVs | `artifacts/splits/` | 50 CSV files (5 folds × 5 datasets × train/val) |
| `aggregated_results_gold.csv` | `artifacts/` | Final 15-run mean ± std across all models/datasets |
