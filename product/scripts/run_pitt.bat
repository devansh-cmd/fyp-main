@echo off
setlocal enabledelayedexpansion

:: ==========================================================
:: PITT CORPUS PHASE 3 RUN (HIGH-RISK GOLD STANDARD)
:: Protocol: LR 1e-5 | Dropout 0.7 | WD 0.1 | UNFREEZE 10
:: ==========================================================

set DATASET=pitt
set MODELS=resnet50 mobilenetv2 hybrid
set SEEDS=42 123 999

echo [STARTING] %DATASET% Marathon...

for %%m in (%MODELS%) do (
    for %%s in (%SEEDS%) do (
        echo [RUNNING] Model: %%m ^| Seed: %%s
        python "%~dp0..\..\product\training\train_unified.py" --dataset %DATASET% --model_type %%m --seed %%s --epochs 30 --batch_size 32 --weighted_loss --lr 1e-5 --dropout 0.7 --weight_decay 0.1 --unfreeze_at 10
    )
)

echo [COMPLETE] %DATASET% Marathon finished.
pause

