@echo off
setlocal enabledelayedexpansion

:: ==========================================================
:: MASTER REMAINING RUNS SCRIPT (PHASE 3)
:: Executes all 34 missing/incomplete runs from the audit.
:: ==========================================================

echo [STARTING] Remaining Phase 3 Runs...

:: --- ITALIAN PD (Tier 2 Protocol) ---
:: Protocol: LR 5e-5 | Dropout 0.5 | WD 0.05 | Unfreeze 0
set DATASET=italian_pd
echo [PROCESSING] %DATASET%

:: Missing resnet50 s999 (failed/empty)
echo [RUNNING] Model: resnet50 ^| Seed: 999
python "%~dp0..\..\product\training\train_unified.py" --dataset %DATASET% --model_type resnet50 --seed 999 --epochs 30 --batch_size 32 --weighted_loss --lr 5e-5 --dropout 0.5 --weight_decay 0.05 --unfreeze_at 0

:: Missing mobilenetv2
for %%s in (42 123 999) do (
    echo [RUNNING] Model: mobilenetv2 ^| Seed: %%s
    python "%~dp0..\..\product\training\train_unified.py" --dataset %DATASET% --model_type mobilenetv2 --seed %%s --epochs 30 --batch_size 32 --weighted_loss --lr 5e-5 --dropout 0.5 --weight_decay 0.05 --unfreeze_at 0
)

:: Missing hybrid
for %%s in (42 123 999) do (
    echo [RUNNING] Model: hybrid ^| Seed: %%s
    python "%~dp0..\..\product\training\train_unified.py" --dataset %DATASET% --model_type hybrid --seed %%s --epochs 30 --batch_size 32 --weighted_loss --lr 5e-5 --dropout 0.5 --weight_decay 0.05 --unfreeze_at 0
)

:: --- ESC-50 (Tier 1 Protocol) ---
:: Protocol: LR 1e-4 | Dropout 0.5 | WD 0.01 | Unfreeze 0
set DATASET=esc50
echo [PROCESSING] %DATASET%

:: Missing mobilenetv2
for %%s in (42 123 999) do (
    echo [RUNNING] Model: mobilenetv2 ^| Seed: %%s
    python "%~dp0..\..\product\training\train_unified.py" --dataset %DATASET% --model_type mobilenetv2 --seed %%s --epochs 30 --batch_size 32 --weighted_loss --lr 1e-4 --dropout 0.5 --weight_decay 0.01 --unfreeze_at 0
)

:: Missing hybrid
for %%s in (42 123 999) do (
    echo [RUNNING] Model: hybrid ^| Seed: %%s
    python "%~dp0..\..\product\training\train_unified.py" --dataset %DATASET% --model_type hybrid --seed %%s --epochs 30 --batch_size 32 --weighted_loss --lr 1e-4 --dropout 0.5 --weight_decay 0.01 --unfreeze_at 0
)

:: --- PITT CORPUS (Tier 3 Protocol) ---
:: Protocol: LR 1e-5 | Dropout 0.7 | WD 0.1 | UNFREEZE 10
set DATASET=pitt
echo [PROCESSING] %DATASET%

:: Missing hybrid
for %%s in (42 123 999) do (
    echo [RUNNING] Model: hybrid ^| Seed: %%s
    python "%~dp0..\..\product\training\train_unified.py" --dataset %DATASET% --model_type hybrid --seed %%s --epochs 30 --batch_size 32 --weighted_loss --lr 1e-5 --dropout 0.7 --weight_decay 0.1 --unfreeze_at 10
)

:: --- EMODB (Tier 1 Protocol) ---
:: Protocol: LR 1e-4 | Dropout 0.5 | WD 0.01 | Unfreeze 0
set DATASET=emodb
echo [PROCESSING] %DATASET%

set MODELS=resnet50 mobilenetv2 hybrid
for %%m in (%MODELS%) do (
    for %%s in (42 123 999) do (
        echo [RUNNING] Model: %%m ^| Seed: %%s
        python "%~dp0..\..\product\training\train_unified.py" --dataset %DATASET% --model_type %%m --seed %%s --epochs 30 --batch_size 32 --weighted_loss --lr 1e-4 --dropout 0.5 --weight_decay 0.01 --unfreeze_at 0
    )
)

:: --- PHYSIONET (Tier 3 Protocol) ---
:: Protocol: LR 1e-5 | Dropout 0.7 | WD 0.1 | UNFREEZE 10
set DATASET=physionet
echo [PROCESSING] %DATASET%

set MODELS=resnet50 mobilenetv2 hybrid
for %%m in (%MODELS%) do (
    for %%s in (42 123 999) do (
        echo [RUNNING] Model: %%m ^| Seed: %%s
        python "%~dp0..\..\product\training\train_unified.py" --dataset %DATASET% --model_type %%m --seed %%s --epochs 30 --batch_size 32 --weighted_loss --lr 1e-5 --dropout 0.7 --weight_decay 0.1 --unfreeze_at 10
    )
)

echo [ALL REMAINING RUNS COMPLETE]
pause

