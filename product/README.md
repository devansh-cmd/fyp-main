# DualCNN-SA-LSTM — FYP Product

**A Novel Dual-Branch CNN with Frequency-Prior Self-Attention for Parkinson's Disease Speech Detection**

Devansh Dev · BSc Computer Science (AI) · Royal Holloway, University of London · 2026

---

## Quick Start

### 1. Environment setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -U pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r product/requirements.txt
```

### 2. Generate spectrograms

Run the appropriate script for each dataset (raw audio must be present in
`product/audio_preprocessing/data/`):

```bash
python product/audio_preprocessing/src/generate_spectrograms_italian.py
python product/audio_preprocessing/src/generate_spectrograms_pcgita.py
python product/audio_preprocessing/src/generate_spectrograms_emodb.py
python product/audio_preprocessing/src/generate_spectrograms.py   # ESC-50
```

### 3. Generate k-fold splits

```bash
python product/scripts/generate_kfold_splits.py --n_folds 5 --seed 42
```

### 4. Train a model

```bash
# Baseline ResNet-50 on Italian PD, fold 0
python product/training/train_unified.py \
    --dataset italian_pd \
    --model_type resnet50 \
    --fold 0 --seed 42 --epochs 30

# Novel architecture (DualCNN-SA-LSTM) with SpecAugment + label smoothing
python product/training/train_unified.py \
    --dataset italian_pd \
    --model_type dual_cnn_sa_lstm \
    --fold 0 --seed 42 --epochs 30 \
    --spec_augment --label_smoothing 0.05

# ResNet-50 + Coordinate Attention (best single-module ablation)
python product/training/train_unified.py \
    --dataset italian_pd \
    --model_type resnet50_ca \
    --fold 0 --seed 42 --epochs 30
```

### 5. Aggregate results across folds

```bash
python product/scripts/aggregate_kfold_results.py --n_folds 5
```

---

## Model Types

| `--model_type`        | Description                                      |
|-----------------------|--------------------------------------------------|
| `resnet50`            | ResNet-50 baseline (ImageNet pretrained)         |
| `resnet50_ca`         | ResNet-50 + Coordinate Attention                 |
| `resnet50_se`         | ResNet-50 + Squeeze-and-Excitation               |
| `resnet50_gate`       | ResNet-50 + Attention Gate                       |
| `resnet50_sa`         | ResNet-50 + Self-Attention                       |
| `resnet50_ca_ag`      | ResNet-50 + mixed CA (L1–3) + AG (L4)            |
| `resnet50_ca_sa`      | ResNet-50 + mixed CA (L1–3) + SA (L4)            |
| `resnet50_ca_lstm`    | ResNet-50 + CA + BiLSTM temporal head            |
| `dual_cnn_sa_lstm`    | **Novel: EfficientNetV2-S + ResNet-50 + FP-SA + BiLSTM** |
| `dual_cnn_lstm`       | Ablation: dual CNN + BiLSTM without FP-SA        |
| `hybrid`              | HybridNet: ResNet-50 + MobileNetV2 gated fusion  |
| `mobilenetv2`         | MobileNetV2 baseline                             |

---

## Supported Datasets

| `--dataset`   | Task                            | Classes |
|---------------|---------------------------------|---------|
| `italian_pd`  | Parkinson's Disease detection   | 2       |
| `pcgita`      | PD detection (Colombian Spanish)| 2       |
| `physionet`   | Heart sound classification      | 2       |
| `pitt`        | Dementia detection              | 2       |
| `emodb`       | Emotion recognition (German)    | 7       |
| `esc50`       | Environmental sound             | 50      |

---

## Running Tests

```bash
# Fast tests only (no weight downloads, runs in ~10 s)
pytest

# Full test suite including model forward-pass checks (~5 min, needs internet)
pytest -m slow
```

---

## Key Training Arguments

| Argument            | Default | Description                                      |
|---------------------|---------|--------------------------------------------------|
| `--epochs`          | 30      | Number of training epochs                        |
| `--lr`              | 1e-4    | Learning rate (AdamW)                            |
| `--dropout`         | 0.5     | Classifier head dropout                          |
| `--seed`            | 42      | Random seed for reproducibility                  |
| `--fold`            | None    | K-fold index (0–4); omit for single-split        |
| `--spec_augment`    | off     | Enable SpecAugment (time + frequency masking)    |
| `--label_smoothing` | 0.0     | Label smoothing ε (0.05 recommended for PD)      |
| `--weighted_loss`   | off     | Class-weighted loss for imbalanced datasets      |
| `--unfreeze_at`     | 0       | Epoch to unfreeze backbone deep layers           |

---

## Directory Structure

```
product/
├── models/                  All model architectures
│   ├── dual_cnn_sa_lstm.py  Novel DualCNN-SA-LSTM architecture
│   ├── freq_prior_attention.py  Frequency-Prior Self-Attention module
│   ├── model_builder.py     Factory pattern (20+ backbone+attention combos)
│   └── [other modules]      SE, CBAM, CA, AG, SA, TripletAttention, ...
├── training/
│   └── train_unified.py     Dataset-agnostic training engine
├── audio_preprocessing/     Spectrogram generation pipelines
├── datasets/                K-fold split generation scripts
├── scripts/                 Utilities: aggregation, leakage audit, GradCAM
├── tests/                   Pytest test suite
│   ├── test_models.py       Model shape-contract tests
│   ├── test_attention.py    Attention module tests
│   └── test_leakage.py      Data pipeline and label map tests
├── artifacts/
│   ├── splits/              All fold CSVs
│   └── runs/                Results, checkpoints, TensorBoard logs
└── requirements.txt
```

---

## Reproducibility

All experiments used `--seed 42`, `--seed 123`, and `--seed 999` across 5 folds
(15 runs per model). SHA-256 hashes in spectrogram filenames provide
cryptographic traceability. The zero-leakage guarantee is enforced at runtime:
training aborts with a `RuntimeError` if any subject appears in both train and
validation sets.
