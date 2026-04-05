# A Novel Dual-Branch CNN with FrequencyPriorSelfAttention for Parkinson's Disease Speech Detection

**Devansh Dev** · BSc Computer Science (Artificial Intelligence) · Royal Holloway, University of London · 2026
**Supervisor:** Dr. Li Zhang

---

## Overview

This project presents `dual_cnn_sa_lstm`, a novel deep learning architecture for automated Parkinson's Disease (PD) detection from speech spectrograms. The system fuses dual-branch convolutional feature extraction (ResNet50 + EfficientNetV2-S) with a custom FrequencyPriorSelfAttention (FP-SA) module, Coordinate Attention, and a Bidirectional LSTM temporal head.

All experiments are evaluated under speaker-independent 5-fold cross-validation with certified zero data leakage, producing N=15 observations per dataset (3 seeds × 5 folds) with Wilcoxon signed-rank statistical testing.

**Key result:** F1 = 0.964 ± 0.051 on the Italian PD dataset — the strongest reported result under a fully verifiable evaluation protocol.

---

## Repository Structure

```
├── product/                          Main codebase
│   ├── models/                       All model architectures
│   │   ├── dual_cnn_sa_lstm.py       Novel final architecture
│   │   ├── freq_prior_attention.py   FrequencyPriorSelfAttention module
│   │   ├── coordinate_attention.py   Coordinate Attention module
│   │   ├── model_builder.py          Factory (20+ backbone+attention combos)
│   │   ├── hybrid_net.py             HybridNet (alpha-gate fusion baseline)
│   │   └── [se_block, cbam, attention_gate, resnet_bilstm, ...]
│   ├── training/
│   │   └── train_unified.py          Dataset-agnostic training engine
│   ├── audio_preprocessing/
│   │   └── src/                      Spectrogram generation scripts per dataset
│   ├── datasets/                     K-fold split generation scripts
│   ├── scripts/                      Aggregation, leakage audit, GradCAM, Wilcoxon
│   ├── notebooks/                    Kaggle GPU execution notebooks
│   ├── tests/                        Pytest suite (22 tests)
│   ├── artifacts/
│   │   ├── splits/                   All 50 fold CSVs (5 datasets × 5 folds × train/val)
│   │   └── runs/                     Experiment results, checkpoints, TensorBoard logs
│   └── requirements.txt
├── documents/
│   ├── final report/                 LaTeX source for final report
│   │   ├── DevanshDev.final.tex      Main report source
│   │   └── figures/                  All report figures (PNG)
│   ├── Conference_Paper/             Submitted conference paper (LaTeX + PDF)
│   ├── DevanshDev-Plan.pdf           Project plan
│   └── Research/                     Reference papers
├── Interim_Submission/               Full interim submission archive
├── diary.md                          Project diary (weekly entries, Terms 1 and 2)
├── ARCHITECTURE.md                   System architecture overview
├── .gitlab-ci.yml                    CI/CD pipeline (lint, test, leakage audit, smoke test)
└── pytest.ini                        Test configuration
```

---

## Demonstration

- **Video Walkthrough:** [YouTube Link Placeholder](#)
- **Screenshots:** Available in the `screenshots/` directory.
- **User and Installation Manual:** Please refer to **Appendix B** of the final report PDF located in `documents/`.

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

### 2. Dataset setup

Raw audio datasets are not included in the repository due to ethics and size constraints. You must acquire them independently:
- **Italian PD**: Available publicly on [IEEE Dataport](https://ieee-dataport.org/open-access/italian-parkinsons-voice-and-speech).
- **PC-GITA**: Requires special permission. Request directly from Prof. Juan Rafael Orozco-Arroyave.
- **ESC-50**: Available on [GitHub](https://github.com/karolpiczak/ESC-50).
- **EmoDB**: Available from [TU Berlin](http://emodb.bilderbar.info/download/).

Once acquired, place them as follows:

| Dataset | Expected path |
|---------|---------------|
| Italian PD | `product/audio_preprocessing/data/Italian Parkinson's Voice and speech/` |
| PC-GITA | `product/audio_preprocessing/data/PC-GITA/` |
| ESC-50 | `product/audio_preprocessing/data/ESC-50/` |
| EmoDB | `product/audio_preprocessing/data/EmoDB-wav/` |

### 3. Generate spectrograms

```bash
python product/audio_preprocessing/src/generate_spectrograms_italian.py
python product/audio_preprocessing/src/generate_spectrograms_pcgita.py
python product/audio_preprocessing/src/generate_spectrograms_emodb.py
python product/audio_preprocessing/src/generate_spectrograms.py        # ESC-50
```

### 4. Generate k-fold splits

```bash
python product/scripts/generate_kfold_splits.py --n_folds 5 --seed 42
```

### 5. Train the final architecture

```bash
# Single fold (fold 0, seed 42)
python product/training/train_unified.py \
    --dataset italian_pd \
    --model_type dual_cnn_sa_lstm \
    --fold 0 --seed 42 --epochs 30 \
    --spec_augment --label_smoothing 0.05

# Full 3-seed × 5-fold matrix (Windows)
product\scripts\kfold_italian_pd.bat
```

### 6. Verify zero leakage

```bash
python product/scripts/verify_leakage.py
```

### 7. Run tests

```bash
pytest        # 22 fast tests, ~10 seconds
```

---

## Datasets

| Dataset | Language | Task | Subjects | Evaluation |
|---------|----------|------|----------|------------|
| Italian PD | Italian | PD vs HC | ~65 | Speaker-independent 5-fold |
| PC-GITA | Colombian Spanish | PD vs HC (DDK) | 101 | Speaker-independent 5-fold |
| ESC-50 | — | 50-class environmental sounds | 2000 clips | Grouped 5-fold |
| EmoDB | German | 7-class speech emotion | 10 actors | Speaker-independent 5-fold |

---

## Architecture

The `dual_cnn_sa_lstm` model processes 224×224 log-Mel spectrograms through:

1. **Dual-branch extraction** — ResNet50 (2048-dim) and EfficientNetV2-S (1280-dim) in parallel
2. **Channel projection** — 1×1 conv reduces concatenated (3328-dim) features to 512-dim
3. **FrequencyPriorSelfAttention** — 8-head self-attention with a learnable frequency-axis prior bias
4. **Coordinate Attention** — direction-aware channel recalibration on time and frequency axes independently
5. **BiLSTM temporal head** — 2-layer bidirectional LSTM over the 49-step spatial sequence
6. **Classifier** — dropout (0.5) + linear output

See `ARCHITECTURE.md` and `product/models/dual_cnn_sa_lstm.py` for full implementation details.

---

## Results Summary

| Dataset | Model | Mean F1 ± Std | N |
|---------|-------|---------------|---|
| Italian PD | dual_cnn_sa_lstm | **0.964 ± 0.051** | 15 |
| PC-GITA (cross-lingual) | dual_cnn_sa_lstm | 0.780 ± 0.056 | 15 |
| ESC-50 | HybridNet | 0.923 ± 0.006 | 15 |
| EmoDB | HybridNet | 0.976 ± 0.006 | 15 |

All results are produced under speaker-independent 5-fold cross-validation with certified zero leakage. Full per-fold breakdowns are in `product/artifacts/runs/` and in the final report (Chapter 7).

---

## CI/CD Pipeline

The `.gitlab-ci.yml` runs four stages on every push:

| Stage | What it does |
|-------|-------------|
| `setup` | Install Python 3.12 dependencies |
| `lint` | Ruff, Black formatting, mypy type checking |
| `audit` | Data leakage verification (`verify_leakage.py`) |
| `package` | One-epoch smoke test on EmoDB (end-to-end pipeline check) |

---

## Reproducibility

- **Seeds:** 42, 123, 999 used across all experiments
- **SHA-256 hashes** embedded in spectrogram filenames for cryptographic traceability back to source recordings
- **Split manifests** locked as SHA-256 checksums in `product/artifacts/splits/`
- **Zero-leakage enforced at runtime:** training aborts with `RuntimeError` on any subject overlap between train and validation sets
- **Static normalisation:** fixed ImageNet parameters used throughout; no training-set statistics are computed at runtime

---

## Troubleshooting

- If imports fail, check the Python interpreter is set to the virtual environment and run `pip install -r product/requirements.txt`
- To inspect training progress: `tensorboard --logdir product/artifacts/runs/`
- To check for data leakage across all splits: `python product/scripts/verify_leakage.py`

---

## Author

**Devansh Dev** — BSc Computer Science (Artificial Intelligence), Royal Holloway, University of London
Supervised by Dr. Li Zhang
