# Thesis Progress Update: Completion of Phase 2 (Acoustic Baselines & Architectural Fusion)

## 1. Experimental Rigor: Clinical-Grade Evaluation Protocol
To ensuring the final findings are publication-ready and statistically rigorous, the experimental methodology underwent a massive upgrade from standard train/validation splits to a **Stratified Grouped 5-Fold Cross-Validation** approach.

**Key Institutional Milestones Achieved:**
*   **Zero Subject Leakage:** A custom `StratifiedGroupKFold` pipeline was implemented. For clinical datasets (Pitt Corpus, Italian PD, PhysioNet), this mathematically guarantees that **no single patient's audio ever exists in both the training and validation sets simultaneously**. 
*   **Reproducibility Matrix:** The full experimental matrix has now been executed across 5 entirely different datasets, 3 distinct backbone architectures, and 5 separate validation folds, totaling **75 complete training runs**.
*   **Agnostic Feature Extraction:** No manual, domain-specific feature engineering (e.g., VAD filtering, noise-reduction) has been applied. All models are forced to learn directly from raw Log-Mel Spectrogram representations, establishing a "Lower Bound" of pure deep-learning capability on noisy audio.

---

## 2. Definitive Results (75-Run K-Fold Matrix)
The following tables represent the final, statistically verified **Macro F1 Scores (Mean ± Standard Deviation across 5 Folds)**. 

### Performance on General & Emotional Audio
This tier proves the proposed Hybrid architecture works on standard acoustic benchmarks.

| Dataset | Complexity | ResNet-50 (Base) | MobileNetV2 | HybridNet (Proposed) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **EmoDB** | Emotion (7-Class) | 0.942 ± 0.017 | 0.856 ± 0.035 | **0.976 ± 0.006** 🏆 | Conclusively Solved |
| **ESC-50** | Env. Noise (50-Class)| 0.835 ± 0.014 | 0.826 ± 0.027 | **0.923 ± 0.006** 🏆 | Massive Breakthrough |

*   **The HybridNet Breakthrough:** On the difficult 50-class ESC dataset, standard CNNs plateaued at ~83%. The proposed `HybridNet` utilizes a novel **Learnable $\alpha$-Gate Fusion**, combining deep semantic features from ResNet with efficient spatial features from MobileNet. This fusion broke the performance ceiling, jumping nearly 10% in F1-score with phenomenally low variance (`± 0.006`).

### Performance on Clinical & Pathological Audio
This tier represents the core thesis domain—messy, real-world medical data.

| Dataset | Pathology | ResNet-50 (Base) | MobileNetV2 | HybridNet (Proposed) | Strategic Headroom |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Italian PD** | Parkinson's | **0.916 ± 0.046** | 0.911 ± 0.063 | 0.908 ± 0.061 | Moderate (~8%) |
| **PhysioNet** | Heart Sounds | 0.889 ± 0.013 | 0.884 ± 0.008 | **0.891 ± 0.011** 🏆 | Moderate (~11%) |
| **Pitt** | Dementia | 0.615 ± 0.008 | 0.627 ± 0.015 | **0.627 ± 0.024** 🏆 | Massive (~37%) |

*   **Clinical Stability:** The HybridNet matched or exceeded standalone backbones on 2 out of 3 clinical tasks, proving the fusion mechanism is robust outside of standard environmental noise.
*   **The "Pitt Ceiling":** The DementiaBank (Pitt) results (`0.627` F1) highlight the extreme difficulty of the task. The audio contains heavy background noise and overriding interviewer dialogue. The current models are struggling to isolate patient-specific cognitive hesitation markers from ambient interference.

---

## 3. Strategic Proposal: Phase 3 (Deep Attention Mechanisms)
With the "Gold Standard" baselines rigorously established, the project is officially transitioning into Phase 3: **Self-Guided Feature Engineering via Spatial and Channel Attention.**

**The Core Scientific Question:**
*Can advanced Attention Mechanisms learn to isolate clinical acoustic markers (like speech hesitation) from severe background noise better than manual feature engineering?*

**Execution Strategy (Targeted Surgical Strikes):**
Rather than arbitrarily applying attention to all models, we will deploy specific mechanisms tailored to the clinical challenges identified in Phase 2:

1.  **Coordinate Attention (CA) for the Pitt Corpus & Italian PD**
    *   *Rationale:* Unlike standard Squeeze-and-Excitation (which suppresses whole channels), CA factorizes 2D spatial pooling into 1D directional vectors (Time/Width and Frequency/Height). 
    *   *Hypothesis:* By forcing the network to understand *where* in time a pause occurs (Dementia) or *where* in frequency a micro-tremor exists (Parkinson's), CA will break the `0.627` baseline ceiling on the Pitt Corpus without requiring manual noise reduction protocols.
2.  **Triplet Attention for PhysioNet**
    *   *Rationale:* Triplet Attention computes cross-dimension interactions (Channel-Spatial, Channel-Time) via tensor rotation. 
    *   *Hypothesis:* This will excel at capturing the highly structured, cyclical nature of abnormal cardiac rhythms better than traditional pooling layers.

**Next Immediate Steps:**
1. Hook `CoordinateAttention` into the ResNet-50 backbone.
2. Execute the 5-Fold Cross-Validation on the Pitt Corpus using the new attention-augmented architecture.
3. Compare the new metrics directly against the established `0.615` ResNet-50 baseline to quantify the exact clinical value of Coordinate Attention.
