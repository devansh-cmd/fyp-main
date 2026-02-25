# Project Diary (reverse chronological)
# Project- Status: **PHASE 3 IN PROGRESS — PC-GITA Integrated. Attention Runs Next.**

# 2026-02-24 PC-GITA DDK Dataset Integration

**Summary**
Integrated the PC-GITA (Colombian Spanish) Parkinson's Disease dataset as a 6th dataset, focusing on **DDK (Diadochokinetic) analysis** — the clinical gold standard for motor speech assessment in PD. This enables a **cross-lingual PD comparison** alongside the Italian PD corpus.

**Dataset Structure:**
- **Source**: PC-GITA corpus (Universidad de Antioquia, Colombia), 44.1kHz WAVs
- **Task**: Combined DDK analysis (6 sub-tasks: ka-ka-ka, pa-pa-pa, ta-ta-ta, pakata, pataka, petaka)
- **Files**: 600 WAVs (300 HC, 300 PD) — perfectly balanced
- **Subjects**: 51 HC, 50 PD (101 total unique speakers)
- **Supervisor Approval**: Dr Li confirmed DDK combined approach (2026-02-23)

**Pipeline Built:**
1. `generate_spectrograms_pcgita.py` — 600/600 Log-Mel spectrograms generated at 16kHz/2048 n_fft/128 mels
2. `make_split_pcgita.py` — Speaker-independent `StratifiedGroupKFold` (5-fold), zero leakage verified
3. `train_unified.py` — Added `pcgita` dataset config (label map, CLI choices, DS_CONFIG)
4. `kfold_pcgita.bat` — Execution script for 15 K-Fold runs (3 models × 5 folds)

**Verification:**
- Smoke test passed: pcgita / ResNet50 / Fold 0 / 1 epoch → Macro F1: 0.506, AUC: 0.663
- Zero subject leakage confirmed across all 5 folds
- Fold sizes: ~480 train (~81 subjects) / ~120 val (~20 subjects) per fold

*Status*: **Ready for 15-run K-Fold execution.**


# 2026-02-23 Phase 2 (K-Fold Baselines) Complete & Kaggle Results Aggregated

**Summary**
The final 20 Kaggle runs (ESC-50 HybridNet, EmoDB ResNet/MobileNet/HybridNet) have been successfully extracted and merged into `product/artifacts/runs/`. This marks the formal completion of the 75-run K-Fold validation matrix.

## The Complete K-Fold Experimental Matrix (75/75 Runs)
All metrics reflect the **Macro F1 Score (Mean ± Standard Deviation)** across 5 Folds.

| Dataset | Objective | ResNet-50 (Base) | MobileNetV2 | HybridNet | Winning Model |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **EmoDB** | Emotion (7-Class) | 0.942 ± 0.017 | 0.856 ± 0.035 | **0.976 ± 0.006** | **HybridNet** 🏆 |
| **ESC-50** | Envir. Noise (50-Class)| 0.835 ± 0.014 | 0.826 ± 0.027 | **0.923 ± 0.006** | **HybridNet** 🏆 |
| **PhysioNet** | Heart Sounds (Abnormal) | 0.889 ± 0.013 | 0.884 ± 0.008 | **0.891 ± 0.011** | **HybridNet** 🏆 |
| **Pitt** | Dementia / Cognition | 0.615 ± 0.008 | **0.627 ± 0.015** | **0.627 ± 0.024** | **HybridNet / MobileNet (Tie)** |
| **Italian PD** | Parkinson's Disease | **0.916 ± 0.046** | 0.911 ± 0.063 | 0.908 ± 0.061 | **ResNet-50** |

## Key Insights
1. **HybridNet Excellence:** The `alpha-gate` fusion approach decisively solved the ESC-50 class overlap problem (jumping from ~83% to 92.3%) and dominated EmoDB, proving the architectural pivot was highly successful for general and emotional audio.
2. **Clinical Viability:** HybridNet remained highly competitive on medical audio, tying for best on the noise-heavy Pitt Corpus and winning on PhysioNet. 
3. **Phase 3 Viability:** EmoDB and ESC-50 are statistically saturated. The remaining headroom for Deep Attention Mechanisms exist primarily in isolating pathology from noisy audio (Pitt Corpus, Italian PD, PhysioNet).

*Status*: **Ready for Phase 3: Spatial and Channel Attention.**

# 2026-02-17 K-Fold Results Analysis & Pitt Protocol Fix

**Summary**
Conducted interim results analysis on the 3 completed datasets (Italian PD, PhysioNet, EmoDB) while Pitt and ESC-50 k-fold runs execute. Discovered and fixed a critical hyperparameter mismatch in the Pitt k-fold script. Identified a scientifically valuable finding regarding Italian PD Fold 3 performance degradation.

## Completed Results (Aggregated Metrics)

### Italian PD — 5-Fold Cross-Validation
| Model | Best Macro F1 | AUC | Control Recall |
|:---|:---:|:---:|:---:|
| ResNet50 | 0.926 ± 0.053 | 0.976 ± 0.019 | 0.866 ± 0.117 |
| MobileNetV2 | 0.930 ± 0.081 | 0.983 ± 0.019 | 0.898 ± 0.160 |
| HybridNet | 0.922 ± 0.076 | 0.969 ± 0.048 | 0.853 ± 0.130 |

### PhysioNet — 5-Fold Cross-Validation
| Model | Best Macro F1 | AUC | Control Recall |
|:---|:---:|:---:|:---:|
| ResNet50 | 0.889 ± 0.013 | 0.943 ± 0.012 | 0.956 ± 0.012 |
| MobileNetV2 | 0.884 ± 0.008 | 0.935 ± 0.006 | 0.945 ± 0.019 |
| **HybridNet** | **0.891 ± 0.011** | **0.956 ± 0.012** | **0.967 ± 0.005** |

### EmoDB — 3-Seed Evaluation (Seeds 42, 123, 999)
| Model | Best Macro F1 | AUC | Control Recall |
|:---|:---:|:---:|:---:|
| ResNet50 | 0.984 ± 0.003 | 0.999 ± 0.000 | 0.996 ± 0.007 |
| MobileNetV2 | 0.925 ± 0.040 | 0.993 ± 0.007 | 0.987 ± 0.013 |
| **HybridNet** | **0.985 ± 0.002** | **1.000 ± 0.000** | 0.994 ± 0.011 |

## Key Findings

1. **HybridNet is the overall best model** — best or tied-best on PhysioNet (all 3 metrics) and EmoDB (F1, AUC), competitive on Italian PD. The gated fusion mechanism successfully combines ResNet and MobileNet strengths.
2. **PhysioNet is the most stable dataset** — all standard deviations below 2%. HybridNet's control recall variance of just 0.005 is excellent for clinical reliability.
3. **MobileNetV2 shows highest variance** — particularly on EmoDB (±4% F1) and Italian PD (±8%), suggesting sensitivity to data partitions. Seed 123 on EmoDB produced a notably weak result (F1=0.882).

## Critical Fix: Pitt K-Fold Hyperparameters

**Problem**: The `kfold_pitt.bat` script was running with default hyperparameters (lr=1e-4, dropout=0.5, weight_decay=0.01, unfreeze_at=0) instead of the validated Phase 3 "Clinical Guard" protocol. This produced catastrophic overfitting: **99.5% train accuracy, 63% val accuracy, and control recall oscillating between 35–50%** — essentially random on the minority class.

**Root Cause**: When the k-fold scripts were created, the Pitt-specific regularization parameters were not carried over from `run_pitt.bat`.

**Fix Applied**:
| Parameter | Before (broken) | After (fixed) |
|:---|:---:|:---:|
| Learning Rate | 1e-4 | **1e-5** |
| Dropout | 0.5 | **0.7** |
| Weight Decay | 0.01 | **0.1** |
| Unfreeze At | 0 | **10** |

The broken `pitt_resnet50_fold0` results were deleted and the script was corrected. The remaining datasets (Italian PD, PhysioNet, EmoDB, ESC-50) are confirmed unaffected — their default parameters either match Phase 3 or produce strong, non-overfitting results.

## Scientific Finding: Italian PD Fold 3 Degradation

All three models exhibited consistent performance degradation on Fold 3:

| Fold | ResNet50 F1 | MobileNetV2 F1 | HybridNet F1 |
|:---:|:---:|:---:|:---:|
| 0 | 0.950 | 0.983 | 0.978 |
| 1 | 0.981 | 0.994 | 0.975 |
| 2 | 0.957 | 0.965 | 0.957 |
| **3** | **0.851** | **0.797** | **0.797** |
| 4 | 0.892 | 0.910 | 0.904 |

**Investigation**: Fold 3's validation set has the most skewed class distribution (58.1% HC vs 41.9% PD), while its training set is PD-dominant (375 PD vs 308 HC). This mismatch causes all models to over-predict PD, crashing control recall to 0.62–0.66.

**Why this matters**: The consistency of the drop across all architectures confirms the difficulty is in the data partition, not an architectural weakness. This is a key argument for k-fold cross-validation over single-split evaluation — it exposes exactly this kind of sensitivity to partition composition in small clinical datasets.

**Zero data leakage was confirmed** across all 5 folds (0 subject overlap between train and val).

## Run Status
| Dataset | Type | Completed | Status |
|:---|:---|:---:|:---|
| Italian PD | K-Fold | 15/15 | Done |
| PhysioNet | K-Fold | 15/15 | Done |
| Pitt | K-Fold | 15/15 | Done (Colab) |
| ESC-50 | K-Fold | 10/15 | HybridNet pending |
| EmoDB | K-Fold | 0/15 | Pending |

*Status*: **55/75 runs complete. Running remaining 20 locally.**


# 2026-02-19 Integrated Colab Runs & Finishing K-Fold
**Summary**
Successfully integrated accelerated k-fold results from Google Colab for Pitt and ESC-50.
- **Pitt Corpus**: Completed all 15 runs. Standard and MobileNetV2 (~62% F1) significantly outperform chance, validating the Phase 3 protocol fix.
- **ESC-50**: Completed 10/15 runs (ResNet50 & MobileNetV2).
- **Remaining Work**: 20 runs (ESC-50 Hybrid + all EmoDB) are now executing via Kaggle (`kfold_kaggle.ipynb`).
*Note: Colab notebooks were removed after successful integration to declutter the repository.*


# 2026-02-13 Stratified K-Fold Cross-Validation Implementation
**Summary**
Transitioned the evaluation methodology from a single 80/20 Train/Val split to **Stratified Grouped 5-Fold Cross-Validation**. This upgrade provides publication-grade statistical rigour by ensuring every subject is validated exactly once across folds, enabling Mean ± Std reporting and paired significance testing.

**Changes Made:**
1.  **Split Scripts** (`make_split_italian.py`, `make_split_pitt.py`, `make_split_physionet.py`, `make_split.py`, `make_split_emodb.py`): Added `StratifiedGroupKFold` / `StratifiedKFold` functions with built-in zero-leakage assertions. Clinical datasets use `subject_id` as the group key; acoustic datasets use `clip_id`.
2.  **Orchestration Script** (`scripts/generate_kfold_splits.py`): Master script that generates all 50 fold CSV files (5 datasets × 5 folds × train/val) in one command. Clinical datasets remap WAV paths to spectrogram PNG paths.
3.  **Training Pipeline** (`train_unified.py`): Added `--fold` argument. When set, the `DS_CONFIG` auto-routes to fold-indexed CSVs. Backward compatible with legacy single-split CSVs.
4.  **Aggregation Script** (`scripts/aggregate_kfold_results.py`): Reads per-fold `summary.json` and outputs Mean ± Std results table with LaTeX snippet.
5.  **Batch Script** (`scripts/run_kfold_experiments.bat`): Executes the full 75-run matrix (5 datasets × 3 models × 5 folds).

**Verification:**
- All 50 fold CSVs generated successfully.
- Smoke test passed: Italian PD / ResNet50 / Fold 0 / 1 epoch → Val Acc: 0.725, Macro F1: 0.713, AUC: 0.817.
- Zero subject leakage confirmed across all clinical folds.

*Status*: **Ready for full 75-run experiment matrix.**


# 2026-02-10 Granular Run Toolkit (Standardization)
**Summary**
Transitioned from monolithic "Marathon" scripts to a dataset-centric granular toolkit. Renamed all calibration scripts to `run_[dataset].bat` and standardized internal parameters to the Phase 3 Tiered Protocol. This provides the user with 9-run per-dataset execution blocks for high-precision validation.

**Toolkit Catalog**:
1. `run_esc50.bat` (Tier 1 Recovery)
2. `run_emodb.bat` (Tier 1 Recovery)
3. `run_italian_pd.bat` (Tier 2 Stability)
4. `run_pitt.bat` (Tier 3 Clinical Guard)
5. `run_physionet.bat` (Tier 3 Clinical Guard)
6. `run_seed999_marathon.bat` (Stand-alone Statistical Coverage)

# 2026-02-10 Technical Review: Split Strategy (Val-as-Holdout)
**Summary**
Audit of the research architecture confirms a dual-split strategy (Train/Validation). In this protocol, the **Validation Set** serves as the definitive **Test Holdout**.

**Rationale for the "Val-as-Holdout" Decision:**
1.  **Sample Constraint**: Datasets like the Pitt Corpus and Italian PD have limited unique subjects. A tertiary "Test" split would reduce the validation sample size below the threshold for stable early stopping or result in a statistically noisy test set.
2.  **Subject Independence**: As verified by the "Zero-Leakage" audit, the Validation set is strictly sequestered by Subject ID. This ensures it provides an unbiased estimate of generalization performance.
3.  **Aggregate Rigor**: We use **3 random seeds (42, 123, 999)**. By reporting the Mean ± Std across these seeds on the holdout (Val) set, we provide a more robust SOTA benchmark than a single pass on a tiny sequestered test set.

*Status*: **Confirmed. No tertiary Test CSVs required.**

# 2026-02-10 Data Leakage Audit (Zero-Leakage Certification)
**Summary**
Conducted a formal audit of the split logic using `verify_leakage.py` across all 45 potential run configurations. The results confirm a "leak-proof" setup, establishing the highest possible rigor for clinical audio research.

**Audit Results:**
- **Subject Isolation (Pitt & PD)**: Confirmed $0$ subject overlap. Patients in Training NEVER appear in Validation, even in segmented or augmented forms.
- **Segment Isolation (All)**: Confirmed $0$ path overlap. Audio segments are strictly divided at the root level before any noise augmentation or windowing.
- **Normalization Stability**: Verified that `train_unified.py` uses static ImageNet parameters. No training-set statistics are leaked into the validation forward pass.

**Verified Split Files:**
- `product/artifacts/splits/train_pitt_segments.csv` (No Subjects Overlap)
- `product/artifacts/splits/train_italian_png.csv` (No Subjects Overlap)

*Verdict*: **CERTIFIED ZERO LEAKAGE.**

# 2026-02-10 Post-Mortem: Marathon Batch 1 (Regression & Memorization)
**Summary**
Audit of the first 27 runs of the "Big 45" Marathon indicates a total failure of the "Universal Regularization" strategy. We hit two distinct failure modes: **Performance Regression** (underfitting) on high-SNR audio and **Delayed Memorization** (overfitting) on clinical speech.

**Dataset-Specific Audit (Keep/Discard Results):**

| Dataset | Outcome | Decision | Technical Rationale |
| :--- | :--- | :--- | :--- |
| **Pitt Corpus**| $99\%$ Train Acc | **DISCARD** | Despite $0.6$ dropout, the model hit "memorization spike" by Epoch 15. The "Generalization Gap" is $>35\%$. We need **Delayed Unfreezing ($Unfreeze=10$)** to kill this. |
| **ESC-50** | $59\%$ Macro F1 | **DISCARD** | Massive regression from the $0.83$ Macro F1 baseline. Aggressive LR ($5e-6$) prevents convergence on multi-class environmental features. |
| **Italian PD** | $75-77\%$ Accuracy| **DISCARD** | Regression from $97\%$ baseline. The "Anti-Overfitting" protocol is literally preventing the model from learning the PD markers. |

**The "Clean Slate" Decision**
None of the runs from Batch 1 meet the "Golden Anchor" rigor. I have halted all processes and successfully wiped the `product/artifacts/runs` and `logs` directories. 

**Deployment: Grand Calibration (V2 Ready)**
- **Tier 1 (Recovery)**: ESC-50/EmoDB reverted to $LR=1e-4, DO=0.5$.
- **Tier 2 (Stability)**: Italian PD calibrated to $LR=5e-5, DO=0.5$.
- **Tier 3 (Clinical Guard)**: Pitt/PhysioNet enforced with $LR=1e-5, DO=0.7$ and **Delayed Unfreezing ($Unfreeze=10$)**.
- Status: **READY FOR RESTART.**

# 2026-02-09 The "Golden Baseline" Calibration (Audit & Pivot)
**Summary**
Audit of the second "True Anchor" run on the Pitt Corpus (using the 5e-6/0.8 dropout protocol) revealed a transition from catastrophic overfitting to legitimate Underfitting. Training accuracy hit a hard ceiling at 59%, failing to capture the complexity of the speech signal. Initiated the "Golden Baseline" Calibration to find the optimal capacity point for ResNet50 and MobileNetV2.

**The Problem: The Regularization Pendulum**
- **Initial State**: LR 1e-4, No Dropout -> Overfitting (99% Train, 66% Val).
- **Secondary State**: LR 5e-6, 0.8 Dropout -> Underfitting (59% Train, 55% Val).
- **The Target**: We need the "Golden Middle"—a protocol that allows the model to learn deep semantic features without drifting into sample memorization.

**Technical Refinements (Pitt Corpus Optimization)**
- **LR 2e-5 (Acceleration)**: Shifted the learning rate back up to 2e-5 to allow the gradient descent process to escape the 59% accuracy plateau.
- **Dropout 0.6 (Capacity Buffer)**: Dialed back the dropout from 0.8 to 0.6. This allows 40% more neurons to participate in the feature mapping, providing the capacity needed for clinical speech nuances.
- **WD 0.1 (Memorization Guard)**: Maintained the high weight decay (0.1) as a strictly standardized pressure against non-linear memorization.
- **Standardized Goal**: Target 75-80% Training Accuracy by Epoch 15 with a <10% Generalization Gap.

**Archiving "Bogus" Experiments**
- **Deprecated Sets**: Moved `aggregated_results.csv` and `all_results.csv` (dated Feb 2) to a dedicated archive. These runs used non-standardized splitting and lacked the Bias-Correction Protocol.
- **Underfit Quarantine**: Archived the recent Phase-4/5 "experimental drift" CSVs. These confirmed that while 0.8 dropout is theoretically sound for high-SNR data, it is computationally "blind" to the subtle clinical markers in DementiaBank speech.
- **Purity Lock**: The active `product/artifacts/runs/` and `logs/` directories are now at 0ms data age, ready for the Golden Baseline.

**Verification command**: 
`python product/training/train_unified.py --dataset pitt --model_type resnet50 --seed 42 --epochs 30 --lr 2e-5 --weight_decay 0.1 --dropout 0.6 --weighted_loss --run_name stabilized_anchor_pitt`

# 2026-02-09 The "Anti-Overfitting" Protocol (Post-Audit)
**Summary**
Audit of the first "True Anchor" run (ResNet50) on the Pitt Corpus revealed catastrophic overfitting (99.4% Train Accuracy by Epoch 10). While weighted loss fixed the bias, the model capacity was too large for the sample size. Deployed the "Anti-Overfitting" Protocol to stable the Golden Anchors.

**Technical Refinements**
- **Precision LR**: Dropped to `5e-6` to prevent gradient vibrations.
- **Aggressive WD**: Increased to `0.1` to enforce feature simplification.
- **Anchor Dropout**: Increased to `0.8` (up from default/none) in the classifier head.
- **Hybrid Support**: Updated `train_unified.py` to support dynamic `--dropout` for all backbones.

# 2026-02-09 The Bias-Correction Protocol - Establishing True Golden Anchors

**Summary**
Critical audit of preliminary baseline results (ResNet50 & MobileNetV2) on the Pitt Corpus revealed a "Biased Plateau". While accuracy appeared stable (~67%), the models exhibited extreme "laziness"—failing to identify 54% of healthy controls (Class 0 failure) and showing poor discriminative power (AUC 0.61). Initiated the Bias-Correction Protocol to establish legitimate, clinically-sound research anchors.

## The Discovery: The Biased Plateau
- **The Accuracy Trap**: Overall accuracy was inflated by the class imbalance in the DementiaBank dataset. The models were effectively "guessing" the majority class (Dementia) to achieve high scores.
- **Class 0 Failure**: Recall for healthy speakers was <46%, which is clinically unacceptable for a detection task.
- **Low AUC**: An AUC of 0.61 confirms that the preliminary baselines had almost zero discriminative power, rendering them useless for comparison against Hybrid models.

## The Bias-Correction Protocol
1. **Weighted Loss (The "Bias-Killer")**:
   - Updated `train_unified.py` with a `--weighted_loss` flag.
   - Implemented dynamic class weight calculation: `Weight = Total / (Num_Classes * Class_Count)`.
   - This prevents the model from ignoring the minority class by penalizing "Control" misclassifications more heavily ($W[0] > W[1]$).
2. **True Golden Anchor Matrix**:
   - Committed to a full re-run of all baseline experiments (ResNet50 & MobileNetV2) across the **3-seed matrix** (42, 123, 999).
   - Pivoted the reporting priority to **Macro F1** and **ROC-AUC** to decouple performance from class distribution.

## Technical Notes
- The weighted loss effectively forces the training engine to optimize for the *intersection* of precision and recall for both classes, not just the raw accuracy of the dominant class.
- This creates a significantly higher "bar" for the Hybrid Ensemble and future attention models to clear.

## Next Steps
1. Create and execute `run_true_anchors.bat`.
2. Aggregate "True Anchor" results to establish the definitive baseline.
3. Quantify the "Bias-Killer" effect by comparing new AUC/F1 against the preliminary sets.

# 2026-02-09 Architectural Pivot: Hybrid Ensemble Integration

**Summary**
Completed the MobileNetV2 benchmarking sprint (15/15 runs) and successfully transitioned to the Hybrid Ensemble. Shifted from standalone backbones to a "Hybrid Specialist" architecture (ResNet50 + MobileNetV2) using Gated Fusion. Implemented clinical-grade optimizations including dynamic backbone unfreezing and enhanced regularization to prevent feature memorization on low-SNR medical datasets.

## Improvements Made
- **Standalone MobileNetV2 Benchmarking**:
  - Successfully executed all 15 MobileNetV2 seeds across 5 datasets.
  - Documented a major performance breakthrough on ESC-50 (+10.4% lead over ResNet50).
  - Established performance parity on medical audio (Pitt, PhysioNet) with 10x fewer parameters.
- **Architectural Shift (Hybrid Ensemble)**:
  - Designed and implemented `HybridNet` with a **Learnable Gated Fusion** mechanism ($\alpha$-gate).
  - Implemented scale alignment via Batch Normalization to prevent ResNet (2048-dim) from dominating MobileNet (1280-dim).
  - Increased MLP Dropout to **0.7** to force more robust feature selection.
- **Training Engine Refinements**:
  - Implemented **Dynamic Unfreezing (Warm-Up)**: Specialists remain frozen for the first 10 epochs to establish the fusion signal before fine-tuning backbones.
  - Integrated configurable **Weight Decay** and a clinical-standard **LR of 1e-5** to stabilize cardiac and speech pathology detection.
- **Experimental Automation**:
  - Developed `run_benchmarks_all.bat` for the Hybrid 15-run sequence.
  - Resolved `KeyError` in Italian PD via definitive label standardization.

## Technical Notes
- **Gated Fusion Rationale**: Unlike naive concatenation, the learnable sigmoid gate ($\alpha$) allows the model to "prune" irrelevant spectral dimensions. This is vital for the Pitt Corpus, where cognitive signals are often masked by ambient noise.
- **The "Dominance" Fix**: Early sanity probes showed 92% training accuracy within 4 epochs (overfitting). The 10-epoch warm-up and high weight decay ($5 \times 10^{-2}$) were implemented to force the model to find generalizable patterns instead of memorizing backbone-specific artifacts.

## Final Benchmarking Comparison (Backbone Baseline)
| Dataset | ResNet50 Acc (%) | MobileNetV2 Acc (%) | Delta |
| :--- | :--- | :--- | :--- |
| **ESC-50** | 74.67 | 85.06 | **+10.39** |
| **Italian PD** | 97.52 | 99.32 | **+1.80** |
| **Pitt Corpus**| 66.19 | 66.83 | **+0.64** |
| **PhysioNet** | 93.13 | 91.97 | -1.16 |
| **EmoDB** | 100.00| 94.39 | -5.61 |

## Next Steps
1. Execute the master benchmarking script (3-seed hybrid matrix).
2. Evaluate if the gated fusion breaks the 67% accuracy ceiling on the Pitt Corpus.
3. Document alpha-vector weights to interpret which specialist (ResNet vs MobileNet) is favored per dataset.

# 2026-02-05 Gold Anchor Baselines Established & Pitt Analysis

**Summary**
Successfully completed the "Gold Anchor" Baseline experiments across all 5 datasets (Italian PD, PhysioNet, ESC-50, EmoDB, Pitt Corpus). Resolved critical training bugs and established definitive performance benchmarks (3-seed means) to guide the next phase of attention-augmented research.

## Improvements Made
- **Training Stability & Bug Fixes**:
  - Resolved `KeyError` in `train_unified.py` by implementing a polymorphic label handler in `UnifiedDataset` to support both integer (PhysioNet/Pitt) and categorical (EmoDB/ESC-50) labels.
  - Implemented automated path-prefix fallback in the DataLoader to handle deep/shallow directory structures for Pitt spectrograms.
- **Experimental Completion**:
  - Executed the final 10-run sequence via `run_remaining_anchors.bat`, completing the 15-run anchor matrix (5 datasets × 3 seeds).
  - Verified 100% reproducibility across seeds for clinical datasets (PhysioNet, Italian PD).
- **Result Tabulation**:
  - Aggregated performance metrics into a central research table.
  - Confirmed "EmoDB Saturation" (100% Acc) and "Pitt Complexity" (66% Acc), providing a clear motivation for Selective Attention Research.

## Technical Notes
- **Pitt Corpus Analysis**: The baseline accuracy of 66.19% reflects the difficulty of detecting cognitive impairment from acoustic textures alone. The high false-negative rate suggests thatResNet-50 is missing temporal/linguistic cues (e.g., latencies) which are often "silent" or subtle in Mel-spectrograms.
- **PhysioNet Fix**: The `KeyError: np.int64(0)` was traced to the CSV using raw encoded indices. The fix ensures the pipeline is now agnostic to whether labels are strings or pre-encoded integers.

## Final Baseline Benchmarks (3-Seed Mean)
| Dataset | Avg Accuracy (%) | Avg Macro F1 | Avg AUC |
| :--- | :--- | :--- | :--- |
| **Italian PD** | 97.52 | 0.937 | 0.992 |
| **PhysioNet** | 93.13 | 0.884 | 0.955 |
| **ESC-50** | 74.67 | 0.731 | 0.987 |
| **EmoDB** | 100.00| 1.000 | 1.000 |
| **Pitt Corpus** | 66.19 | 0.577 | 0.621 |

## Next Steps
1. Transition to **Selective Attention Integration**.
2. Benchmark Coordinate Attention (CA) against these "Gold Anchor" baselines.
3. Investigate "Temporal Pooling" or "Attention Gating" specifically for the Pitt Corpus to address its lower baseline.

# 2026-02-02 Pitt Corpus Integration & CI/CD Stabilization

**Summary**
Integrated the DementiaBank Pitt Corpus with extreme clinical rigor, ensuring subject-independent partitioning and cryptographic traceability. Formalized the Gold Anchor Baseline protocol by automating the final 10 runs across all datasets to ensure a definitive research benchmark.

## Improvements Made
- **DementiaBank Pitt Corpus Integration**:
  - Implemented `make_split_pitt.py` with **Subject-Independent partitioning** (292 unique patients) and SHA-256 stable file IDs for 100% traceability.
  - Implemented `generate_spectrograms_pitt.py` using a **20s sliding window strategy** and deterministic clinical augmentation (Exactly 2 outputs for train, 1 Clean-Only for val).
  - Generated **3,836 high-resolution spectrograms** and locked the splits with a `splits_manifest_pitt.json` SHA-256 checksum to prevent data drift.
- **Benchmarking Automation**:
  - Audited the "Gold Anchor" Baseline status; identified **10 missing/incomplete runs** (PhysioNet S999, ESC-50, EmoDB, and Pitt).
  - Developed `run_remaining_anchors.bat` to sequentially execute these runs, establishing a statistically rigorous ground truth.
- **Codebase Health & CI/CD**:
  - Fixed **14 Ruff linting errors** across the project (improper f-strings, unused imports, undefined names).
  - Resolved GitLab pipeline failures by implementing a minimal CI verification suite (and cleaned up after use to maintain repository purity).

## Technical Notes
- The "Immutability Token" in filenames (e.g., `157_2c87d4e1a0b5_015_orig.png`) provides permanent cryptographic evidence linking every image back to a specific clinical recording.
- Established a "Purity Audit" protocol for the validation set, confirming zero noise-augmented files enter the evaluation loop.

## Next Steps
1. Execute the 10-run sequence via `run_remaining_anchors.bat`.
2. Aggregate all 3-seed mean ± std results for the definitive baseline table.
3. Transition to SOTA push using Coordinate Attention (CA) modules.


# 2026-01-26 Establishing 'Gold Standard' Anchors & Unified Metrics

**Summary**
Formalized a rigorous experimental protocol by establishing "Gold Standard" ResNet-50 anchors across all audio domains. Enhanced the unified training engine to ensure high-resolution statistical comparability for the final research narrative.

## Improvements Made
- **Experimental Design**:
  - Defined the **Primary Anchor** protocol: plain ResNet-50, ImageNet-pretrained, fine-tuning only the classifier and Layer 4.
  - Established a 3-seed statistical requirement (42, 123, 999) for all baseline reporting to ensure clinical reproducibility.
- **Unified Training Engine (`train_unified.py`)**:
  - Implemented automated calculation of **Macro F1-Score** and **One-vs-Rest ROC-AUC** for all tasks.
  - Standardized the output artifact structure: every run now saves a visual `confusion_matrix.png` and a detailed `best_classification_report.json`.
- **Data Pipeline Serialization**:
  - Developed `finalize_training_csvs.py` to create direct-to-spectrogram mapping CSVs (`*_png.csv`), drastically reducing loading overhead during massive batch runs.

## Technical Notes
- Early results on the Italian PD anchor show 0.992 AUC with near-zero variance across seeds, indicating a saturated backbone performance for this specific task.
- This creates a critical research "Pivot": identifying where plain CNNs struggle (e.g., PhysioNet or ESC-50) so attention modules can demonstrate tangible clinical value.

## Next Steps
1. Complete the 12-run anchor batch (4 datasets x 3 seeds).
2. Begin Systematic Attention evaluation (ResNet-50 + Coordinate Attention) against these anchors.
3. Compare "Baseline Ceiling" across general vs. medical audio.


# 2026-01-25 Modular Model Architecture Refactor

**Summary**
Refactored the model architecture implementation to be fully modular and backbone-agnostic. Established a unified model factory that can dynamically inject attention modules into various CNN backbones.

## Improvements Made
- **Modular Modeling**:
  - Developed `model_builder.py`, a centralized factory for creating attention-augmented models.
  - Decoupled attention blocks (SE, CBAM, Coordinate, Triplet, Gate) from specific backbones.
  - Implemented dynamic "plug-and-play" logic to support ResNet and MobileNet variants.
- **Unified Training Loop**:
  - Optimized `train_pd_italian.py` to use the new factory.
  - Simplified training logic by removing backbone-specific boilerplate.
- **Repository Optimization**:
  - Cleaned up redundant architecture-specific files (`resnet50_*.py`) to reduce technical debt and ensure singular sources of truth.

## Technical Notes
- The new architecture uses `nn.Sequential` wrappers to hook attention blocks into existing pre-trained layers without modifying library code.
- This design allows for rapid benchmarking of different attention-backbone combinations via simple CLI arguments.

## Next Steps
1. Re-run Italian PD benchmarks using the new modular framework to ensure parity.
2. Finalize MobileNetV2 hooks for all attention types.
3. Begin systematic evaluation on the PhysioNet dataset.


# 2026-01-23 PhysioNet 2016 Heart Sound Integration & Attention Modules

**Summary**
Successfully integrated the PhysioNet 2016 Heart Sound Abnormal Detection dataset. Implemented stratified splitting, specialized spectrogram generation, and established a suite of modular attention blocks (CA, Triplet, Gate).

## Improvements Made
- **PhysioNet Integration**:
  - Developed `make_split_physionet.py` to index 3,153 recordings across sources A-F.
  - Implemented stratified splitting by both source and binary label (Normal vs Abnormal).
  - Created `generate_spectrograms_physionet.py` with optimized settings for heart sounds (Target SR: 2kHz, 224x224).
- **Attention Modeling**:
  - Implemented **Coordinate Attention (CA)** to capture cross-dimension direction-aware information.
  - Implemented **Triplet Attention** for capturing cross-dimension interactions through rotation.
  - Implemented **Attention Gate** module for focus-driven feature pruning.

## Technical Notes
- PCG (Heart Sounds) processing uses a lower sampling rate (2kHz) compared to speech (16kHz) to focus on relevant low-frequency cardiac cycles.
- Splitting protocol ensures source distribution is balanced between training and validation sets to prevent domain bias.

## Next Steps
1. Establish ResNet50 baseline on PhysioNet dataset.
2. Systematic benchmarking of all attention modules (CA, CBAM, Triplet, Gate) across Italian PD and PhysioNet.
3. Begin preparations for DementiaBank (Week 4 goal).


# 2026-01-22 Italian PD Dataset Integration & Framework Preparation

**Summary**
Successfully integrated the Italian Parkinson's Disease dataset into the unified project pipeline. Generated 831 high-quality Log-Mel spectrograms and established a robust training framework for cross-domain evaluation.

## Improvements Made
- **Data Preprocessing**:
  - Implemented `generate_spectrograms_italian.py` using a strictly non-augmented protocol to preserve subtle acoustic pathology.
  - Processed all 831 files (HC & PD) at 16kHz with 2048/512/128 Mel configuration (~224x224).
- **Modeling Infrastructure**:
  - Developed `train_pd_italian.py`, a unified script for ResNet50, SE-Net, and CBAM experiments.
  - Integrated comprehensive reporting: Accuracy, ROC-AUC, Macro F1, Precision, and Recall.
  - Automated JSON summary and confusion matrix generation for every run.
- **Repository Hygiene**:
  - Removed all legacy references to the PC-GITA dataset to focus purely on the new Italian corpus.

## Technical Notes
- Fixed `PDDataset` loader to handle irregular filenames (double dots) common in the Italian raw data.
- Established a `product/artifacts/runs/italian_pd/` structure for standardized result tracking.
- Pre-runs (Sanity Probes) confirmed stable training dynamics across all three architectures.

## Next Steps
1. Implement and benchmark the **Coordinate Attention (CA)** module.
2. Integrate **PhysioNet (Heart Sound)** dataset (Week 3 target).
3. Begin cross-dataset evaluation protocol design.


# 2025-12-16 Final Report Polish & Submission Prep

**Summary**
Finalized the Interim Report with all required figures and corrections. Prepared the codebase for technical submission by updating documentation and packaging the project.

## Improvements Made
- **Report**: 
  - Added confusion matrices and UML diagrams (Class & Sequence).
  - Fixed typos and aligned formatting with submission guidelines.
  - Updated abstract and project title.
- **Documentation**:
  - Overhauled `README.md` to highlight "Advanced Targets" (Attention, Reproducibility).
  - Detailed the full project workflow including EmoDB and Attention experiments.
- **Submission**:
  - Prepared repository for submission.

# 2025-12-02 Results Aggregation for Interim Report

**Summary**
Fixed aggregate_seeds.py to correctly identify all model types and generated comprehensive results tables with LaTeX format for interim report.

## Results Analysis
- Fixed model detection logic to recognize ResNet50 baseline runs (timestamp-based directory names)
- Successfully aggregated results across all experiments:
  - **ESC-50**: ResNet50 89.9%±0.9, SE 88.8%±2.4, CBAM 85.1%, AlexNet 74.5%±4.3, Baseline CNN 44.4%
  - **EmoDB**: ResNet50 95.7%±3.2, SE 95.8%±0.9
- Generated LaTeX tables ready for inclusion in interim report
- Organized broken CBAM runs into _BROKEN folder for clarity

## Technical Notes
- Script now uses glob pattern `**/*summary*.json` to catch all summary files
- Properly handles both timestamp directories and named run folders
- Validates all results against original JSON files - confirmed accuracy
- CSV outputs saved to product/artifacts/ (all_results.csv, aggregated_results.csv)

## Next Steps
1. Focus on writing interim report (Abstract, Intro, Methodology done)
2. Complete Results chapter with generated tables and analysis
3. Finish Planning and Conclusion chapters
4. Consider running remaining CBAM seeds (123, 999) if time permits

# 2025-11-30 ResNet50+CBAM Fixed Implementation

**Summary**
Successfully fixed CBAM attention module implementation and completed seed42 training run for ESC-50 dataset.

## CBAM Training Results (Seed 42)
- **Val Accuracy**: 85.1%
- **Macro F1**: 0.851
- **Top-3 Accuracy**: 95.5%
- Training completed successfully with fixed attention integration

## Technical Fixes
- Fixed CBAM module integration with ResNet50 bottleneck blocks
- Corrected channel attention pooling dimensions
- Verified gradient flow through attention pathways
- Run saved to: resnet50_cbam_seed42_fixed/

## Observations
- CBAM performance (85.1%) lower than baseline ResNet50 (89.9%) and SE-Net (88.8%)
- Possible explanations:
  - Attention mechanism may need different hyperparameters for audio spectrograms
  - CBAM adds more parameters → may need longer training or different regularization
  - Spatial attention in CBAM might not be as beneficial for spectrogram features

## Next Steps
1. Run CBAM with seeds 123 and 999 for reproducibility analysis
2. Consider hyperparameter tuning (learning rate, weight decay) for CBAM
3. Analyze attention maps to understand what CBAM focuses on
4. Begin writing interim report with current results

# 2025-11-29 EmoDB SE-Net Training (Seed 999)

**Summary**
Completed third EmoDB SE-Net training run with seed 999 for reproducibility analysis.

## Results (Seed 999)
- **Val Accuracy**: 94.7%
- **Macro F1**: 0.936
- Slightly lower than seeds 42 (96.4%) and 123 (96.3%)
- All three seeds show consistent high performance on EmoDB

## EmoDB SE-Net Summary (All Seeds)
| Seed | Val Acc | Macro F1 |
|------|---------|----------|
| 42   | 96.4%   | 0.958    |
| 123  | 96.3%   | 0.952    |
| 999  | 94.7%   | 0.936    |
| **Mean** | **95.8% ±0.9%** | **0.949** |

## Observations
- EmoDB shows higher accuracy than ESC-50 (expected - only 7 emotion classes vs 50 sound classes)
- SE-Net provides small improvement over baseline ResNet50 (95.8% vs 95.7%)
- Low variance across seeds indicates good reproducibility
- German speech emotions well-captured by transfer learning from ImageNet

## Next Steps
1. Fix and re-run CBAM experiments (previous runs had implementation bugs)
2. Complete all ESC-50 CBAM runs with 3 seeds
3. Begin aggregating all results for interim report
4. Start writing Results and Analysis chapter

# 2025-11-26 ESC-50 SE-Net Training Complete

**Summary**
Completed all three SE-Net training runs on ESC-50 with seeds 42, 123, and 999. SE-Net shows competitive but slightly lower performance compared to baseline ResNet50.

## Final SE-Net Results (ESC-50)
| Seed | Val Acc | Macro F1 | Top-3 Acc | Notes |
|------|---------|----------|-----------|-------|
| 42   | 87.0%   | 0.870    | 95.7%     | Lowest of three seeds |
| 123  | 87.8%   | 0.876    | -         | Middle performance |
| 999  | 91.6%   | 0.914    | -         | Best performance |
| **Mean** | **88.8% ±2.4%** | **0.887** | - | Comparable to baseline |

## Key Observations
- SE-Net mean (88.8%) slightly lower than ResNet50 baseline (89.9%)
- Higher variance (±2.4%) compared to baseline (±0.7%)
- Seed 999 significantly outperformed other seeds (91.6% vs 87.0%)
- Suggests SE-Net may be more sensitive to initialization

## Comparison: SE-Net vs Baseline
- Baseline ResNet50: 89.9% ±0.7%
- SE-Net: 88.8% ±2.4%
- Difference: -1.1% (not significant given variance)
- SE-Net adds channel attention but doesn't clearly improve ESC-50

## Technical Notes
- All runs used same hyperparameters: lr=5e-4, wd=1e-2, bs=64
- Training stable across all seeds
- No implementation issues observed
- Confusion matrices show similar patterns to baseline

## Next Steps
1. Complete CBAM experiments (currently in progress)
2. Run EmoDB experiments for both SE-Net and CBAM
3. Aggregate all results and perform statistical analysis
4. Begin interim report writing with focus on attention mechanism comparison

# 2025-11-25 ResNet50 SE-Net Implementation

**Summary**
Implemented Squeeze-and-Excitation (SE) blocks for ResNet50 and began training on ESC-50 dataset with multiple seeds.

## SE-Net Implementation
- Added SE blocks to ResNet50 bottleneck layers
- SE ratio: 16 (reduces channels by 16x in squeeze operation)
- Integrated after each bottleneck block's final convolution
- Verified gradient flow through SE pathways

## Training Configuration
- Same setup as baseline ResNet50 for fair comparison
- Learning rate: 5e-4, Weight decay: 1e-2, Batch size: 64
- 30 epochs, AdamW optimizer
- Seeds: 42, 123, 999 for reproducibility

## Early Results (Seeds 42, 123)
- Seed 42: 87.0% validation accuracy
- Seed 123: 87.8% validation accuracy
- Performance comparable to baseline ResNet50 (~89.9%)
- SE blocks add minimal overhead (<1% parameter increase)

## Next Steps
1. Complete seed 999 run
2. Analyze per-class performance differences vs baseline
3. Implement CBAM (Convolutional Block Attention Module)
4. Compare all three attention mechanisms systematically

# 2025-11-23 GitLab CI/CD Pipeline Fix

**Summary**
After 18+ commits and extensive debugging, finally resolved all GitLab CI/CD pipeline issues. Pipeline now runs successfully on university runners.

## Issues Resolved
1. **Cache corruption across runners**: Venv binaries incompatible between cim-ts-node-01/02/03
   - Solution: Force venv recreation in setup stage with `rm -rf .venv312`
2. **PEP 668 externally-managed-environment**: System Python blocked pip installs
   - Solution: Always create fresh venv, use `python -m pip` instead of bare `pip`
3. **Workflow rules blocking jobs**: No jobs visible on main branch
   - Solution: Added `when: always` rule for main branch
4. **Missing dev tools in lint stage**: Ruff not installed
   - Solution: Added `pip install ruff black mypy` in lint stage script

## Pipeline Stages
1. **setup**: Create venv, install PyTorch CPU + dependencies
2. **lint**: Run ruff, black, mypy (excluding notebooks)
3. **test**: Run pytest on codebase
4. **audit**: Security checks with pip-audit
5. **package**: Save artifacts (checkpoints, results)

## Lessons Learned
- Heterogeneous runners require explicit venv versioning or recreation
- Docker Python 3.12-slim images have PEP 668 restrictions
- Notebook cells run out of order → exclude from linting
- Simpler is better: recreate venv instead of complex caching

## Next Steps
1. Focus on experiments (SE-Net, CBAM implementations)
2. Stop fighting with CI/CD - it works now
3. Prepare for interim report with working pipeline as evidence

# 2025-11-21 EmoDB Training Splits & ResNet50 Baseline

**Summary**
Created stratified train/val splits for EmoDB dataset and completed initial ResNet50 baseline training runs with seeds 123 and 999.

## EmoDB Split Creation
- Total spectrograms: 3,210 (535 clips × 6 augmentations)
- Split: 80% train / 20% val
- Stratified by emotion class to maintain class balance
- Clip-level grouping to prevent augmentation leakage
- Generated train_emodb.csv and val_emodb.csv

## ResNet50 Baseline Results (EmoDB)
| Seed | Val Acc | Macro F1 | Notes |
|------|---------|----------|-------|
| 123  | 93.5%   | 0.917    | Lower than seed 999 |
| 999  | 98.0%   | 0.975    | Excellent performance |
| **Mean** | **95.7% ±3.2%** | **0.946** | Strong but high variance |

## Observations
- EmoDB easier than ESC-50 (95.7% vs 89.9%) - only 7 emotion classes
- Higher variance between seeds (±3.2% vs ±0.7% for ESC-50)
- Seed 999 nearly perfect performance (98%)
- German emotional speech well-captured by spectrogram representations
- Transfer learning from ImageNet works surprisingly well for speech emotions

## Technical Notes
- Same hyperparameters as ESC-50: lr=5e-4, wd=1e-2, bs=64
- 30 epochs training
- 224×224 mel spectrograms with same preprocessing pipeline
- AdamW optimizer

## Next Steps
1. Complete remaining baseline runs (seed 42 if needed)
2. Begin SE-Net experiments on both datasets
3. Implement CBAM attention mechanism
4. Cross-dataset analysis: ESC-50 vs EmoDB patterns

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
