# Final Year Project — Audio Classification

This repository contains code and experiments for an audio classification project based on the **ESC-50** dataset.  
It includes preprocessing scripts, dataset handling, model definitions, and training workflows for reproducible benchmarking.

---

## Project Overview

The system classifies environmental sounds using log-mel spectrograms and convolutional neural networks (CNNs).  
The pipeline converts raw audio into spectrogram images, splits them into train/validation sets, and trains a baseline CNN model.

---

---

## Data Setup

Since the datasets are excluded from this repository (as per submission guidelines), please follow these steps to set up the data:

1.  **ESC-50 Dataset**: 
    - Download from: [ESC-50 GitHub](https://github.com/karolpiczak/ESC-50)
    - Place the `audio/` folder and `meta/esc50.csv` into:
      `product/audio_preprocessing/data/ESC-50/`
2.  **EmoDB (Berlin Emotional Speech Database)**:
    - Download from: [EmoDB Kaggle (mirror)](https://www.kaggle.com/piyushagni5/berlin-database-of-emotional-speech-emodb)
    - Place the `.wav` files **directly** into (do not include a sub-folder):
      `product/audio_preprocessing/data/EmoDB-wav/`

---

## Repository Structure

product/
│
├── audio_preprocessing/
│   ├── data/ESC-50/audio/         # Raw ESC-50 WAV files
│   ├── src/                       # Preprocessing scripts
│   │   ├── audio_utils.py         # Common I/O and augmentation functions
│   │   ├── generate_spectrograms.py
│   │   └── augment_audio.py
│   └── outputs/spectrograms/      # Generated spectrogram PNGs
│
├── datasets/
│   ├── make_split.py              # Creates train/val CSVs (ESC-50)
│   ├── make_split_emodb.py        # Creates train/val CSVs (EmoDB)
│   └── esc50_png_dataset.py       # PyTorch Dataset for spectrograms
│
├── models/
│   ├── baseline_cnn.py            # Baseline CNN architecture
│   ├── cbam.py                    # Convolutional Block Attention Module (Reusable)
│   ├── se_block.py                # Squeeze-and-Excitation Block (Reusable)
│   └── __init__.py
│
├── training/
│   ├── train_baseline.py          # Baseline training script
│   ├── resnet50_se.py             # ResNet-50 + SE training script
│   ├── resnet50_cbam.py           # ResNet-50 + CBAM training script
│   ├── Resnet50_t1.py             # Standard ResNet-50 training script
│   ├── alexnet_t1.py              # AlexNet training script
│   └── aggregate_seeds.py         # Results aggregation tool
│
└── artifacts/
    ├── splits/                    # train.csv / val.csv
    └── runs/                      # Model checkpoints, logs, TensorBoard data
```

## Advanced Targets

- **CBAM & SE Blocks:** Native implementation of attention mechanisms for audio feature refinement.
- **Data Leakage Resolution:** Stratified splitting based on source file metadata to prevent class pollution.
- **Aggregation Logic:** Multiple seed runs with statistical aggregation.

## System Architecture

The codebase follows a modular design:
- `models/`: Architecture definitions.
- `training/`: Training and validation loops.
- `datasets/`: Data loading and preprocessing.


## Workflow Summary

1. **Generate Spectrograms**

   Convert the raw audio into log-mel PNGs. This is required once per dataset.

   **For ESC-50:**
   ```powershell
   python product/audio_preprocessing/src/generate_spectrograms.py
   ```

   **For EmoDB:**
   ```powershell
   python product/audio_preprocessing/src/generate_spectrograms_emodb.py
   ```

2. **Run Batch Experiments (The "One-Click" Way)**

   Instead of running individual training commands, use these scripts. They will automatically handle the **train/val splitting** (preventing data leakage) and then run all experiments sequentially.

   - **ESC-50 Suite**: Run `run_all_experiments.bat`
     *(Trains ResNet-50 with/without SE and CBAM across 3 seeds)*

   - **EmoDB Suite**: Run `run_emodb_experiments.bat`
     *(Validates system robustness on Speech Emotion Recognition)*

*Note: For granular control, individual training scripts in `product/training/` can still be run manually using the commands detailed in the sections below.*

---

## Granular Control (Manual Execution)

For detailed evaluation of specific components or custom hyperparameter testing, you can run the internal scripts directly.

### 1. Manual Dataset Splitting
Use these to re-generate splits with different ratios or seeds.
- **ESC-50**:
  ```powershell
  python product/datasets/make_split.py --val_ratio 0.1 --seed 123
  ```
- **EmoDB**:
  ```powershell
  python product/datasets/make_split_emodb.py --spec_dir product/audio_preprocessing/outputs/spectrograms_emodb --out_dir product/artifacts/splits
  ```

### 2. Manual Model Training
All training scripts support standardized arguments (`--epochs`, `--batch_size`, `--lr`, `--seed`).
- **Baseline CNN**: 
  ```powershell
  python -m product.training.train_baseline --epochs 10 --batch_size 32
  ```
- **ResNet-50 + Attention (SE)**:
  ```powershell
  python -m product.training.resnet50_se --lr 1e-3 --run_name custom_test
  ```

### 3. Cross-Dataset Testing
To train any architecture on EmoDB instead of the default ESC-50, override the CSV paths:
```powershell
python -m product.training.resnet50_cbam `
  --train_csv product/artifacts/splits/train_emodb.csv `
  --val_csv product/artifacts/splits/val_emodb.csv
```

---

## 🚀 Marker's Quick Start (Batch Verification)

Once the data is set up and spectrograms are generated, use these "one-click" scripts to verify the core results:

1.  **ESC-50 Validation (Attention Models)**: 
    Run `run_all_experiments.bat` to execute the full test suite for ResNet-50 with SE and CBAM modules across 3 random seeds.
2.  **EmoDB Cross-Dataset Validation**: 
    Run `run_emodb_experiments.bat` to verify the system's robustness on a different domain (Speech Emotion Recognition).

*The results will be logged to `product/artifacts/runs/` and can be aggregated using `python -m product.training.aggregate_seeds`.*

---

## Key Conventions

- **Paths** — handled through `pathlib.Path` and helper functions in `audio_utils.py`. Avoid hardcoding paths.  
- **File naming** — spectrogram PNGs follow `<base>_orig.png`, `_noisy.png`, `_pitchUp2.png`, `_stretch0.9.png`.  
- **Audio loading** — uses `librosa.load(..., sr=None)` to preserve the original sampling rate.  
- **Augmentation** — implemented at the audio level (noise, pitch shift, time stretch) before spectrogram conversion.  
- **Logging** — TensorBoard logs stored in `runs/`, model checkpoints saved under `product/artifacts/runs/`.

---

## Output Locations

| Artifact | Path |
|-----------|------|
| Spectrogram images | `product/audio_preprocessing/outputs/spectrograms/` |
| Augmented WAVs | `product/audio_preprocessing/data/augmented_wav/` |
| Train/Val splits | `product/artifacts/splits/` |
| Model checkpoints & logs | `product/artifacts/runs/` |

---

## Troubleshooting

- If imports fail in VS Code/Pylance, ensure the correct Python interpreter is selected and dependencies installed:  
  ```bash
  pip install -r requirements.txt
  ```
- For `NameError` or missing imports in notebooks, verify `import os` or use `pathlib` consistently.
- To inspect training progress:  
  ```bash
  tensorboard --logdir runs/
  ```

---

## Environment Setup (example)

```bash
python -m venv .venv
.\.venv\Scriptsctivate
pip install -U pip
pip install numpy scipy librosa soundfile matplotlib pandas scikit-learn torch torchvision tensorboard pillow
```

---

## Author
**Devansh Dev** — BSc Computer Science (Artificial Intelligence)  
Royal Holloway, University of London