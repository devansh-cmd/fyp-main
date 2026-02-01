@echo off
:: Phase 1: SE Calibration (Global Baseline)
:: Model: ResNet50 + SE
:: Protocol: No Augmentation, 15 Epochs, 3 Seeds

set SPLITS=product/artifacts/splits/
set EPOCHS=15

echo --- STARTING PHASE 1: SE CALIBRATION ---

:: 1. ESC-50
echo Running ESC-50 ...
for %%S in (42 123 999) do (
    python product/training/train_unified.py --dataset esc50 --model_type resnet50_se --seed %%S --epochs %EPOCHS% --run_name phase1_esc50_se_s%%S --train_csv %SPLITS%train_no_aug.csv --val_csv %SPLITS%val_no_aug.csv
)

:: 2. EmoDB
echo Running EmoDB ...
for %%S in (42 123 999) do (
    python product/training/train_unified.py --dataset emodb --model_type resnet50_se --seed %%S --epochs %EPOCHS% --run_name phase1_emodb_se_s%%S --train_csv %SPLITS%train_emodb_no_aug.csv --val_csv %SPLITS%val_emodb_no_aug.csv
)

:: 3. Italian PD
echo Running Italian PD ...
for %%S in (42 123 999) do (
    python product/training/train_unified.py --dataset italian_pd --model_type resnet50_se --seed %%S --epochs %EPOCHS% --run_name phase1_italian_pd_se_s%%S --train_csv %SPLITS%train_italian_png.csv --val_csv %SPLITS%val_italian_png.csv
)

:: 4. PhysioNet
echo Running PhysioNet ...
for %%S in (42 123 999) do (
    python product/training/train_unified.py --dataset physionet --model_type resnet50_se --seed %%S --epochs %EPOCHS% --run_name phase1_physionet_se_s%%S --train_csv %SPLITS%train_physionet_png.csv --val_csv %SPLITS%val_physionet_png.csv
)

echo PHASE 1 COMPLETE.
