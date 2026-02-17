@echo off
setlocal enabledelayedexpansion

:: ==========================================================
:: ITALIAN PD PHASE 3 RUN (STABLE CLINICAL RECOVERY)
:: Protocol: LR 5e-5 | Dropout 0.5 | WD 0.05 | Unfreeze 0
:: ==========================================================

set DATASET=italian_pd
set MODELS=resnet50 mobilenetv2 hybrid
set SEEDS=42 123 999

echo [STARTING] %DATASET% Marathon...

for %%m in (%MODELS%) do (
    for %%s in (%SEEDS%) do (
        echo [RUNNING] Model: %%m ^| Seed: %%s
        python "%~dp0..\..\product\training\train_unified.py" --dataset %DATASET% --model_type %%m --seed %%s --epochs 30 --batch_size 32 --weighted_loss --lr 5e-5 --dropout 0.5 --weight_decay 0.05 --unfreeze_at 0
    )
)

echo [COMPLETE] %DATASET% Marathon finished.
pause

