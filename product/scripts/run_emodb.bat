@echo off
setlocal enabledelayedexpansion

:: ==========================================================
:: EMODB PHASE 3 RUN (HIGH-CAPACITY RECOVERY)
:: Protocol: LR 1e-4 | Dropout 0.5 | WD 0.01 | Unfreeze 0
:: ==========================================================

set DATASET=emodb
set MODELS=resnet50 mobilenetv2 hybrid
set SEEDS=42 123 999

echo [STARTING] %DATASET% Marathon...

for %%m in (%MODELS%) do (
    for %%s in (%SEEDS%) do (
        echo [RUNNING] Model: %%m ^| Seed: %%s
        python "%~dp0..\..\product\training\train_unified.py" --dataset %DATASET% --model_type %%m --seed %%s --epochs 30 --batch_size 32 --weighted_loss --lr 1e-4 --dropout 0.5 --weight_decay 0.01 --unfreeze_at 0
    )
)

echo [COMPLETE] %DATASET% Marathon finished.
pause

