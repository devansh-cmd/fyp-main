@echo off
:: Phase 0: Ground Truth Anchor (GOLD STANDARD RESET)
:: Model: Plain ResNet50
:: Protocol: No Augmentation, 15 Epochs, 3 Seeds
:: Purpose: Establishes the definitive non-attention benchmark with static label mapping.

set SPLITS=product/artifacts/splits/
set EPOCHS=15

echo --- STARTING GOLD STANDARD PHASE 0 ANCHORS ---

:: 1. ESC-50 (ALL 3 SEEDS)
echo Running ESC-50 ...
for %%S in (42 123 999) do (
    python product/training/train_unified.py --dataset esc50 --model_type resnet50 --seed %%S --epochs %EPOCHS% --run_name phase0_esc50_anchor_s%%S --train_csv %SPLITS%train_no_aug.csv --val_csv %SPLITS%val_no_aug.csv
)

:: 2. EmoDB (ALL 3 SEEDS)
echo Running EmoDB ...
for %%S in (42 123 999) do (
    python product/training/train_unified.py --dataset emodb --model_type resnet50 --seed %%S --epochs %EPOCHS% --run_name phase0_emodb_anchor_s%%S --train_csv %SPLITS%train_emodb_no_aug.csv --val_csv %SPLITS%val_emodb_no_aug.csv
)

:: NOTE: Italian PD and PhysioNet are already COMPLETE and verified clean.
:: They are preserved in the runs/ folder but skipped here to prioritize GPU time.

echo GOLD STANDARD PHASE 0 ANCHOR RUNS COMPLETE.
