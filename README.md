# Final Year Project — Audio Classification

This repository contains code and experiments for an audio classification project based on the **ESC-50** dataset.  
It includes preprocessing scripts, dataset handling, model definitions, and training workflows for reproducible benchmarking.

---

## Project Overview

The system classifies environmental sounds using log-mel spectrograms and convolutional neural networks (CNNs).  
The pipeline converts raw audio into spectrogram images, splits them into train/validation sets, and trains a baseline CNN model.

---

## Repository Structure

```
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
│   ├── make_split.py              # Creates train/val CSVs
│   └── esc50_png_dataset.py       # PyTorch Dataset for spectrograms
│
├── models/
│   └── baseline_cnn.py            # Baseline CNN architecture
│
├── training/
│   └── train_baseline.py          # Main training script
│
└── artifacts/
    ├── splits/                    # train.csv / val.csv
    └── runs/                      # Model checkpoints, logs, TensorBoard data
```

- **`documents/`** — contains notes and `diary.md` (progress log).
- **`notebooks/`** — exploratory Jupyter notebooks for waveform and spectrogram checks.

---

## Workflow Summary

1. **Generate spectrograms**

   Convert WAVs to log-mel PNGs with augmentations (noise, pitch-up, time-stretch):

   ```powershell
   python product/audio_preprocessing/src/generate_spectrograms.py
   ```

   Output PNGs are saved under  
   `product/audio_preprocessing/outputs/spectrograms/`, e.g.:

   ```
   1-100032-A-0_orig.png
   1-100032-A-0_noisy.png
   1-100032-A-0_pitchUp2.png
   1-100032-A-0_stretch0.9.png
   ```

2. **Create train/validation splits**

   ```powershell
   python product/datasets/make_split.py `
     --spec_dir product/audio_preprocessing/outputs/spectrograms `
     --esc50_csv product/audio_preprocessing/data/ESC-50/meta/esc50.csv `
     --out_dir product/artifacts/splits --val_ratio 0.2
   ```

   Generates stratified `train.csv` and `val.csv` files.

3. **Train the baseline model**

   ```powershell
   python -m product.training.train_baseline --project_root . --epochs 10 --batch_size 32
   ```

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
