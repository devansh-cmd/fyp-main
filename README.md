# Final Year Project — Audio Classification with Attention-Augmented CNNs

Multi-domain audio classification system benchmarking CNN architectures with attention mechanisms (SE, CBAM) across **5 diverse datasets** spanning environmental sounds, speech emotion, and clinical speech pathology.

---

## Project Overview

The system classifies audio signals using log-mel spectrograms and transfer-learning CNNs.  
It evaluates **3 model architectures** across **5 datasets** using **Stratified 5-Fold Cross-Validation**, producing publication-grade Mean ± Std metrics.

### Datasets

| Dataset | Domain | Classes | Samples | Splitting Strategy |
|---------|--------|---------|---------|-------------------|
| **ESC-50** | Environmental Sound | 50 | 8,000 spectrograms | Clip-level `StratifiedKFold` |
| **EmoDB** | Speech Emotion | 7 | 3,210 spectrograms | Clip-level `StratifiedKFold` |
| **Italian PD** | Parkinson's Disease | 2 (PD/HC) | 831 spectrograms | Subject-grouped `StratifiedGroupKFold` |
| **Pitt Corpus** | Alzheimer's/Dementia | 2 (AD/Control) | 3,836 segments | Subject-grouped `StratifiedGroupKFold` |
| **PhysioNet 2016** | Heart Sound | 2 (Normal/Abnormal) | 3,153 records | Source+label `StratifiedKFold` |

### Models
- **ResNet-50** — ImageNet-pretrained backbone with custom classifier head
- **MobileNetV2** — Lightweight efficiency baseline
- **HybridNet** — Custom attention-augmented architecture

### Evaluation: 5-Fold Cross-Validation

```
┌────────────────────────────────────────────────────────┐
│          Stratified Grouped 5-Fold CV                  │
│                                                        │
│  Fold 0:  ████████████████████ │▒▒▒▒▒│                │
│  Fold 1:  ████████████████│▒▒▒▒▒│████                 │
│  Fold 2:  ████████████│▒▒▒▒▒│████████                 │
│  Fold 3:  ████████│▒▒▒▒▒│████████████                 │
│  Fold 4:  │▒▒▒▒▒│████████████████████                 │
│                                                        │
│  ████ = Train    ▒▒▒▒▒ = Validation                   │
│                                                        │
│  ✓ Zero subject leakage (clinical datasets)            │
│  ✓ Every subject validated exactly once                │
│  ✓ Stratified by diagnosis label                       │
│  ✓ Results: Mean ± Std across 5 folds                  │
│                                                        │
│  Experiment Matrix: 5 datasets × 3 models × 5 folds   │
│                   = 75 training runs                    │
└────────────────────────────────────────────────────────┘
```

---

## Data Setup

Since datasets are excluded from this repository, please set up the following:

1.  **ESC-50**: Download from [ESC-50 GitHub](https://github.com/karolpiczak/ESC-50)
    - Place `audio/` and `meta/esc50.csv` into: `product/audio_preprocessing/data/ESC-50/`
2.  **EmoDB**: Download from [EmoDB Kaggle](https://www.kaggle.com/piyushagni5/berlin-database-of-emotional-speech-emodb)
    - Place `.wav` files directly into: `product/audio_preprocessing/data/EmoDB-wav/`
3.  **Italian PD**: Place the dataset into: `product/audio_preprocessing/data/Italian Parkinson's Voice and speech/`
4.  **Pitt Corpus**: Place cookie-theft recordings into: `product/audio_preprocessing/data/English Pitt Corpus/`
5.  **PhysioNet 2016**: Place heart sound databases into: `product/audio_preprocessing/data/physionet.org/`

---

## Repository Structure

```
product/
├── audio_preprocessing/
│   ├── data/                          # Raw audio datasets (excluded from repo)
│   ├── src/                           # Spectrogram generation scripts
│   │   ├── audio_utils.py             # Common I/O and augmentation functions
│   │   ├── generate_spectrograms.py   # ESC-50 spectrogram generation
│   │   └── augment_audio.py           # Audio augmentation pipeline
│   └── outputs/                       # Generated spectrogram PNGs
│       ├── spectrograms/              # ESC-50
│       ├── spectrograms_emodb/        # EmoDB
│       ├── spectrograms_italian/      # Italian PD
│       ├── spectrograms_pitt/         # Pitt Corpus
│       └── spectrograms_physionet/    # PhysioNet
│
├── datasets/
│   ├── make_split.py                  # ESC-50 splits (single + K-Fold)
│   ├── make_split_emodb.py            # EmoDB splits
│   ├── make_split_italian.py          # Italian PD splits (subject-grouped)
│   ├── make_split_pitt.py             # Pitt Corpus splits (subject-grouped)
│   ├── make_split_physionet.py        # PhysioNet splits
│   └── esc50_png_dataset.py           # PyTorch Dataset for spectrograms
│
├── models/
│   ├── baseline_cnn.py                # Baseline CNN architecture
│   ├── cbam.py                        # CBAM attention module
│   ├── se_block.py                    # Squeeze-and-Excitation module
│   └── __init__.py
│
├── training/
│   └── train_unified.py               # Unified training pipeline (all datasets + K-Fold)
│
└── artifacts/
    ├── splits/                        # Train/Val CSVs (single-split + fold-indexed)
    └── runs/                          # Model checkpoints, logs, summaries

scripts/
├── generate_kfold_splits.py           # Generate all 50 fold CSV files
├── aggregate_kfold_results.py         # Compute Mean ± Std from fold results
├── run_kfold_experiments.bat          # Run full 75-experiment matrix
└── verify_leakage.py                  # Data leakage verification tool
```

---

## Workflow

### 1. Generate Spectrograms (once per dataset)

```powershell
python product/audio_preprocessing/src/generate_spectrograms.py           # ESC-50
python product/audio_preprocessing/src/generate_spectrograms_emodb.py     # EmoDB
python product/audio_preprocessing/src/generate_spectrograms_italian.py   # Italian PD
python product/audio_preprocessing/src/generate_spectrograms_pitt.py      # Pitt Corpus
python product/audio_preprocessing/src/generate_spectrograms_physionet.py # PhysioNet
```

### 2. Generate K-Fold Splits

```powershell
python scripts/generate_kfold_splits.py --n_folds 5 --seed 42
```

This creates **50 CSV files** (5 datasets × 5 folds × train/val), all pointing to `.png` spectrogram paths.

### 3. Run Experiments

**Full 75-run matrix (automated):**
```powershell
scripts\run_kfold_experiments.bat
```

**Single fold (manual):**
```powershell
python product/training/train_unified.py --dataset italian_pd --model_type resnet50 --fold 0 --epochs 30
```

**Legacy single-split mode (backward compatible):**
```powershell
python product/training/train_unified.py --dataset esc50 --model_type resnet50 --seed 42 --epochs 30
```

### 4. Aggregate Results

```powershell
python scripts/aggregate_kfold_results.py --n_folds 5
```

Outputs a Mean ± Std table in CSV and LaTeX format.

---

## Key Conventions

- **Paths** — handled via `pathlib.Path`. Avoid hardcoding.
- **File naming** — spectrogram PNGs: `<base>_orig.png`, `_noisy.png`, `_pitchUp2.png`, `_stretch0.9.png`.
- **Augmentation** — applied at the audio level before spectrogram conversion.
- **Subject independence** — clinical datasets (Italian PD, Pitt) split by `subject_id` to guarantee zero leakage.
- **Logging** — TensorBoard logs and model checkpoints saved under `product/artifacts/runs/`.

---

## Output Locations

| Artifact | Path |
|----------|------|
| Spectrogram images | `product/audio_preprocessing/outputs/spectrograms*/` |
| Train/Val split CSVs | `product/artifacts/splits/` |
| K-Fold CSVs | `product/artifacts/splits/*_fold{0-4}.csv` |
| Model checkpoints & logs | `product/artifacts/runs/` |
| Aggregated results | `product/artifacts/kfold_results.csv` |

---

## Environment Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -U pip
pip install numpy scipy librosa soundfile matplotlib pandas scikit-learn torch torchvision tensorboard pillow
```

---

## Troubleshooting

- If imports fail, ensure the correct Python interpreter is selected and dependencies installed:
  ```bash
  pip install -r requirements.txt
  ```
- To inspect training progress:
  ```bash
  tensorboard --logdir product/artifacts/runs/
  ```

---

## Author
**Devansh Dev** — BSc Computer Science (Artificial Intelligence)  
Royal Holloway, University of London