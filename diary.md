# Project Diary (reverse chronological)

### 🔍 Week 2 Reflection
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
- Compiled notes into librosa_docs.md covering relevant API calls and examples.

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
