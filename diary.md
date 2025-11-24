# Project Diary (reverse chronological)
# 2025-11-20 EmoDB Integration + Augmented Spectrogram Generation

**Summary**
Set up the full EmoDB preprocessing pipeline and generated augmented spectrograms. Confirmed dataset structure, verified filename-based emotion labels, and produced a complete CSV for future training runs.

## Dataset Verification
- Successfully downloaded and mounted the EmoDB dataset.
- Confirmed presence of **535 WAV files** under `wav/`.
- Validated the official filename-label structure:
  - Emotion is always the **6th character** (W, L, E, A, F, T, N).
- Manually listened to several samples to confirm correct emotional categories and signal quality.
- Ensured all files follow the expected pattern (`SS T RR E V.wav`) with no corruption.

## Spectrogram Pipeline
- Implemented the dedicated EmoDB generation script and verified end-to-end functionality.
- Generated spectrograms for all 535 audio clips using the same Mel configuration as ESC-50.
- Ensured consistent output dimensions (224×224) across all PNGs.

## Augmentation
- Applied controlled augmentation per clip: **1 original + 5 variations** (noise, pitch shift, stretch, reverb, gain).
- Total generated: **3210 spectrograms**.
- Verified augmentation grouping logic — all augmented variants remain under the same clip ID, preventing leakage.
- Inspected random samples to confirm augmentation strength does not distort emotional prosody.

## Technical Notes
- Confirmed correct count before and after spectrogram generation.
- Successfully produced `emodb_all.csv` mapping each PNG to its emotion class.
- Verified paths, filenames, and label consistency across the entire dataset.

## Next Steps
1. Create grouped train/val splits for EmoDB.
2. Run first ResNet-50 baseline on the EmoDB spectrogram dataset.
3. Begin comparing ESC-50 vs EmoDB generalisation patterns.
4. Prepare for integration of attention models (CBAM, SE) using both datasets.


# 2025-11-19 ResNet50 Baseline Training

**Summary**
Completed ResNet50 transfer learning baseline with 3 random seeds (42, 123, 999).

**Results** 

| Seed | Val Acc | Macro F1 | Notes |
|------|---------|----------|-------|
| 42   | 89.4%   | 0.892    | First run, consistent performance |
| 123  | 89.3%   | 0.892    | Very consistent with seed 42 |
| 999  | 90.9%   | 0.909    | Best performance, better on quiet sounds |
| **Mean** | **89.9% ±0.7%** | **0.898** | Strong reproducibility |

## Key Observations

### Model Behavior
- Fast convergence: 85%+ by epoch 8
- Overfitting after epoch 18: 100% train acc, val plateaus at 89-90%
- Val loss best around epoch 24-25, then slight increase
- Early stopping at epoch 20-22 would be optimal

### Per-Class Performance
**Strong classes (>90% recall):**
- Siren, toilet_flush, thunderstorm, dog, fireworks, chainsaw

**Challenging classes:**
- Seed 42/123: insects (30) at 46-59%, mouse_click (33) at 47%
- Seed 999: door_wood_creaks (19), hen (29), laughing (32) at ~69%

**Interesting finding:** Seed 999 performed significantly better on quiet/transient sounds (insects 93.8% vs 46% in other seeds). Shows importance of multiple seeds.

## Technical Details
- Model: ResNet50 pretrained on ImageNet, fine-tuned classifier
- Optimizer: AdamW, lr=5e-4, weight_decay=1e-2
- Batch size: 64, Image size: 224x224
- Epochs: 30 (but could stop at 20)
- Data: ESC-50 spectrograms with augmentations (orig, noisy, pitch, stretch)

## Next Steps
1. Implement attention mechanisms (CBAM, SE-Net)
2. Run 40-50 epochs for attention models
3. Add early stopping (patience=10)
4. Consider increasing regularization (weight_decay=5e-2, more dropout)

## Issues Fixed Today
- CSV file had wrong paths (missing _orig, _noisy suffixes)
- Regenerated splits with make_split.py
- Fixed .gitignore to allow JSON/PT but block PNGs in runs folder

## GitLab CI/CD
- Finally got pipeline working after 18+ commits
- Simple pipeline: just install deps and run tests
- Excluded notebooks from linting (cells run out of order)

## 2025-11-13
**AlexNet Baseline (ESC-50, Clean Split) — Final Summary**
- Learning rate: 2.5e-4
- Weight decay: 1e-2
- Batch size: 64
- Epochs: 20
- Seed: 42
- Train/Val split: Non-leaky; all augmented variants grouped
**Results:**
- Peak Val Accuracy: ~72–73% (epoch 5–7)
- Final Train Accuracy: ~99%
- Final Val Accuracy: ~70% (overfitting evident)

Best Val Loss: ~1.0

**Confusion Matrix:** clean diagonal structure, no leakage patterns

**Interpretation:**
AlexNet learns ESC-50 quickly, overfits after epoch ~5, and reaches expected performance comparable to published baselines. This is now a valid baseline for comparison against ResNet-50, CBAM, SE, and EfficientNet.

## 2025-11-12 — Data Leakage Discovery & AlexNet Transfer-Learning Setup
- Implemented **AlexNet transfer-learning baseline** (T1 run) on clean ESC-50 splits.
- Conducted leakage audit to verify dataset integrity before training.

**Next Steps:**

- Proceed with AlexNet T1 transfer-learning run using clean splits.
- Compare metrics vs baseline CNN and prepare confusion-matrix visualisations.
Once AlexNet is stable, **begin ResNet-50 transfer-learning for cross-architecture comparison.**

## 2025-11-12
**Issue Discovered:**

While preparing the transfer learning phase for AlexNet, I ran a leakage audit script to verify the integrity of my training and validation splits.
The results were alarming — 1,171 overlapping clip IDs were found between train.csv and val.csv. **This meant that augmented versions of the same original clip (e.g., _orig, _noisy, _pitchUp, _stretch) were appearing in both splits.**

In other words, my baseline CNN had been trained and validated on overlapping data.
That explains the unusually high validation accuracy — the model wasn’t truly generalising; **it was effectively seeing the same sounds twice, just slightly modified.**

- Root Cause Analysis:
The original make_split.py script stratified the data per spectrogram image, not per unique clip ID.
Since I had generated multiple spectrograms for each audio clip (to increase training diversity), the script unknowingly distributed these variants across both train and val sets.

This violated a core rule of machine learning experimentation:

**No derivative of a training sample should ever appear in validation.** 

**Fix Implemented:**
- Re-wrote make_split.py to group by unique clip_id before stratified splitting.
- Regenerated clean train.csv and val.csv.
- Re-trained baseline CNN and logged new metrics.

## 2025-11-10 — Baseline Integration & Refinements

- Merged validated baseline CNN training branch into main (merge: integrate ESC-50 baseline CNN training).
- Added final run logs and confusion matrices under product/artifacts/runs.
- Extended .gitignore to exclude redundant generated files and experimental artifacts.
- Baseline CNN 20-epoch run results (lr = 5e-4, wd = 1e-2, bs = 32):
- Train loss: ↓ 3.2 → 1.0 | Val loss: ≈ 1.9 → 1.6
- Accuracy: ↑ to ~60 % (train) / 38–42 % (val)
- TensorBoard logs confirmed stable convergence, reproducible runs, and correct seed handling.

- Insight: Baseline now ready to serve as benchmark for upcoming transfer-learning models (AlexNet, ResNet-50).

## 2025-11-04 — Hyperparameter Sweep & Baseline Validation
- 2025-11-04 — Hyperparameter Sweep & Baseline Validation (Week 3 Milestone)
- Completed systematic learning-rate / batch-size sweep to stabilise the baseline CNN.
- Conducted five 5-epoch runs on the augmented ESC-50 subset (runs/sweep).
- Best configuration: **lr = 5 × 10⁻⁴, batch = 32.**
- Lower LR (1e-4) under-fit; higher LR (1e-3) caused mild validation oscillation.
- Batch 64 showed slower generalisation.
- Confirmed smooth convergence: loss ↓ 3.38 → 1.50, val acc ↑ 9 → 53 %.
- Verified checkpoint saving (**baseline_cnn_e5.pt**)
- Insight → **ESC-50 CNN learns reliably with modest regularisation; augmentation effective.**
## 2025-11-02 — TensorBoard Integration and Stable Training Loop
- **Added TensorBoard logging** for loss/accuracy tracking per epoch.
- Timestamped runs stored under product/artifacts/runs.
- Implemented model checkpoint saving after each training cycle.
- Rewrote training loop with **AdamW optimizer + gradient stability fixes**.
- Result → clean, monitorable training pipeline foundation for all future models (AlexNet, ResNet-50).

## 2025-10-30 — Baseline CNN Implementation
- Completed first functional CNN pipeline (spilt → dataset → model → train).
- Verified shape integrity ([B, 3, 224, 224] → [B, 50]).
- Conducted dummy dataset run to confirm end-to-end flow.
- Logged commit “feat: baseline CNN pipeline (splits > dataset > model > train loop)”.

### Week 2 Reflection
- **Key learning:** Transitioning from theory (Librosa basics) to a functioning pipeline revealed the importance of environment isolation and reproducible data paths. Migrating to `.venv` removed unexpected dependency conflicts and clarified version control.  
- **Technical insight:** Rewriting augmentations with NumPy + SciPy not only stabilised the workflow but deepened my understanding of how signal transformations actually modify spectral features.  
- **Challenges:** Frequent Numba/LLVM crashes within Anaconda caused major delays; debugging this required switching environments and manually tracing where JIT compilation occurred.  
- **Improvements made:** Modularised the preprocessing codebase, implemented error-tolerant loops, and added logging checkpoints to prevent silent failures.  
- **Next focus:** Move into **Week 3 (Model Training)** by implementing a baseline CNN and validating that the augmented dataset enhances model generalisation.  

## 2025-10-28 — Completed full augmentation pipeline (Week 2 milestone)
- Finalised and tested the entire audio preprocessing workflow end-to-end.  
  - Integrated **audio_utils.py**, **generate_spectrograms.py**, and **augment_audio.py** as modular scripts.  
  - Verified output paths and dataset structure align with ESC-50 conventions.  
- Resolved critical environment issues:  
  - Migrated from Anaconda to a clean `.venv` to eliminate LLVM/Numba conflicts.  
  - Rewrote augmentation logic using **NumPy + SciPy** only (no `librosa.effects`), improving runtime stability.  
- Successfully generated the complete spectrogram dataset:  
  - **2 000 original** `_orig.png` spectrograms from `generate_spectrograms.py`.  
  - **5 994 augmented** spectrograms (`_noisy`, `_pitchUp2`, `_stretch0.9`) → **≈ 7 994 total PNGs**.  
  - Progress checkpoints (`[50/2000]`, `[100/2000]`, …) confirmed smooth batch execution.  
- Clean Git history with clear milestones (latest: *“feat(audio): complete augmentation pipeline (8k spectrogram dataset ready)”*).  
- Next: begin **Week 3 — Baseline CNN Training** on the 8 k spectrogram dataset to establish initial accuracy benchmarks.

---

## 2025-10-27 — Stabilising augmentation scripts
- Debugged repeated LLVM / Numba errors during spectrogram generation.  
- Identified cause: Anaconda’s MKL + SVML vectorisation; resolved by disabling JIT and isolating environment.  
- Verified script functionality in `.venv` using pure Python build.  
- Ensured augmentation produced consistent spectrogram image dimensions (≈ 224 × 224 px).  
- Updated `.gitignore` to exclude generated PNGs and preview images.  
- Logged clean commit *“Update .gitignore to exclude generated spectrogram PNGs and debug waveform previews.”*

---

## 2025-10-26 — Codebase refactor & augmentation tests
- Refactored preprocessing scripts into `/src/` for modular imports.  
- Implemented augmentation pipeline prototypes:  
  - Noise injection, pitch shift (+2 semitones), and time-stretch (0.9×).  
- Initial test run produced partial spectrograms but revealed path inconsistencies.  
- Fixed directory handling and validated output save logic.  
- Commit milestone: *“Refined augmentation pipeline: added noise-based transformations and generated spectrogram outputs.”*

---

## 2025-10-25 — Augmentation pipeline setup
- Extended Week 2 goals to include automated augmentation.  
- Implemented and committed core transformations within `augment_audio.py`.  
- Achieved working generation of `_noisy`, `_pitchUp2`, `_stretch0.9` images for test subset.  
- Standardised file-naming conventions for traceability across experiments.  
- Commit: *“Add audio augmentation pipeline (noise, pitch shift, time stretch).”*

---

## 2025-10-24 — Mel / Log-Mel generation verification
- Validated Mel- and Log-Mel-spectrogram generation on multiple ESC-50 clips.  
- Ensured consistent sampling rate (22 050 Hz) and output resolution (~224 × 224).  
- Produced example notebooks visualising transformations.  
- Commit: *“Added Mel & Log-Mel spectrogram generation notebook.”*

---

## 2025-10-21 — Waveform visualisation
- Completed waveform inspection notebook (`01_load_and_visualize.ipynb`).  
- Verified amplitude scaling, sampling rate integrity, and display of multiple audio examples.  
- Exported representative waveform figures for report documentation.  
- Commit: *“Added waveform visualisation notebook and manual export.”
## 2025-10-20 — Dataset verification + documentation
- Successfully verified the ESC-50 dataset integrity.
  - 2000 metadata entries matched with 2000 .wav files.
- Verified metadata structure (esc50.csv) and directory consistency.
- Added a short verification notebook (00_dataset_check.ipynb) and logged results in dataset_check.txt.
- Created /documents/references/librosa_docs.md summarising all Librosa functions used in preprocessing.
-Structured commits to mark milestone (Verified ESC-50 dataset, Add Librosa documentation summary).
-Repository now fully aligned for Week 2 tasks (waveform visualization, spectrogram generation, augmentation).

## 2025-10-15 — Research: Librosa & preprocessing fundamentals
- Explored the Librosa library in detail for audio preprocessing tasks.
- Understood how to load audio, inspect sampling rates, and visualise waveforms.
- Studied Mel-spectrogram and Log-Mel conversion pipeline.
- Reviewed augmentation methods: noise injection, pitch shift, and time stretch.
- Compiled notes into librosa_docs.md covering examples.

## 2025-10-08 — Research: neural + audio foundations

-Focused on understanding CNNs and their application to audio.
Covered:
  - How neurons and activation functions enable non-linear learning.
  - How stacked layers form deep networks capable of complex pattern recognition.
  - CNN structure: convolution, pooling, and dense layers for feature extraction and classification.
  - Concept of transfer learning using pre-trained CNNs **(e.g., ResNet-50) for ESC-50.**
  - Audio workflow: waveform → Mel-spectrogram → CNN for sound classification.


## 2025-10-03 — Kickoff + scope confirmation
- Met with Li Zhang; agreed starting point:
  - Dataset: **ESC-50**, later extend to real-world (e.g., medical) audio.
  - Models: **CNN on spectrograms** → **ResNet-50 / AlexNet** with transfer learning; explore **attention**.
  - Metrics: **Accuracy, AUC, Confusion Matrix** (scikit-learn).
  - Language/tooling: **Python** (librosa, PyTorch/TensorFlow, sklearn).
- Action items: 
  - Set up repo per RHUL rules (diary at root, product/, documents/, .gitignore).
  - Prepare spectrogram pipeline stub and EDA notebook.
  - Read up: ESC-50 docs, spectrogram basics, ResNet-50 for audio.

## 2025-10-02 — Admin & planning
- Decided core aim: benchmark CNN baselines vs transfer learning on ESC-50.
- Outlined evaluation protocol and notebook structure.
