@echo off
:: Cumulative Gold Anchor Baseline Execution
:: Model: ResNet50 (Anchor)
:: Protocol: No Augmentation / Deterministic Clinical Augmentation
:: Epochs: 15 (Standardized)

set SPLITS=product/artifacts/splits/
set EPOCHS=15

echo --- STARTING REMAINING GOLD ANCHOR BASELINES ---

:: 1. PhysioNet (Clean-up Seed 999)
echo [1/10] Running PhysioNet S999...
python product/training/train_unified.py --dataset physionet --model_type resnet50 --seed 999 --epochs %EPOCHS% --run_name anchor_phys_s999 --train_csv %SPLITS%train_physionet_png.csv --val_csv %SPLITS%val_physionet_png.csv

:: 2. ESC-50 (3 Seeds)
for %%S in (42 123 999) do (
    echo [%%S] Running ESC-50 S%%S...
    python product/training/train_unified.py --dataset esc50 --model_type resnet50 --seed %%S --epochs %EPOCHS% --run_name phase0_esc50_anchor_s%%S --train_csv %SPLITS%train_no_aug.csv --val_csv %SPLITS%val_no_aug.csv
)

:: 3. EmoDB (3 Seeds)
for %%S in (42 123 999) do (
    echo [%%S] Running EmoDB S%%S...
    python product/training/train_unified.py --dataset emodb --model_type resnet50 --seed %%S --epochs %EPOCHS% --run_name phase0_emodb_anchor_s%%S --train_csv %SPLITS%train_emodb_no_aug.csv --val_csv %SPLITS%val_emodb_no_aug.csv
)

:: 4. Pitt Corpus (3 Seeds - NEW)
for %%S in (42 123 999) do (
    echo [%%S] Running Pitt Corpus S%%S...
    python product/training/train_unified.py --dataset pitt --model_type resnet50 --seed %%S --epochs %EPOCHS% --run_name anchor_pitt_s%%S --train_csv %SPLITS%train_pitt_segments.csv --val_csv %SPLITS%val_pitt_segments.csv
)

echo --- ALL REMAINING BASELINES COMPLETE ---
pause
