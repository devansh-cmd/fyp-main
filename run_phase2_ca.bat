@echo off
:: Phase 2: Coordinate Attention (Strategic Discovery)
:: Model: ResNet50 + CA
:: Protocol: No Augmentation, 15 Epochs, 3 Seeds

set SPLITS=product/artifacts/splits/
set EPOCHS=15

echo --- STARTING PHASE 2: COORDINATE ATTENTION ---

:: 1. PhysioNet (MANDATORY - PRIMARY HYPOTHESIS)
echo Running PhysioNet ...
for %%S in (42 123 999) do (
    python product/training/train_unified.py --dataset physionet --model_type resnet50_ca --seed %%S --epochs %EPOCHS% --run_name phase2_physionet_ca_s%%S --train_csv %SPLITS%train_physionet_png.csv --val_csv %SPLITS%val_physionet_png.csv
)

:: 2. ESC-50 (MANDATORY - SECONDARY HYPOTHESIS)
echo Running ESC-50 ...
for %%S in (42 123 999) do (
    python product/training/train_unified.py --dataset esc50 --model_type resnet50_ca --seed %%S --epochs %EPOCHS% --run_name phase2_esc50_ca_s%%S --train_csv %SPLITS%train_no_aug.csv --val_csv %SPLITS%val_no_aug.csv
)

:: 3. EmoDB (Completeness)
echo Running EmoDB ...
for %%S in (42 123 999) do (
    python product/training/train_unified.py --dataset emodb --model_type resnet50_ca --seed %%S --epochs %EPOCHS% --run_name phase2_emodb_ca_s%%S --train_csv %SPLITS%train_emodb_no_aug.csv --val_csv %SPLITS%val_emodb_no_aug.csv
)

:: 4. Italian PD (Completeness)
echo Running Italian PD ...
for %%S in (42 123 999) do (
    python product/training/train_unified.py --dataset italian_pd --model_type resnet50_ca --seed %%S --epochs %EPOCHS% --run_name phase2_italian_pd_ca_s%%S --train_csv %SPLITS%train_italian_png.csv --val_csv %SPLITS%val_italian_png.csv
)

echo PHASE 2 COMPLETE.
