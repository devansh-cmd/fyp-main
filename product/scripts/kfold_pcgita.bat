@echo off
REM PC-GITA DDK K-Fold Cross-Validation (5 folds x 3 models = 15 runs)
REM Speaker-independent splits, same protocol as Italian PD (Tier 2 Stability)

echo ========================================
echo PC-GITA DDK K-Fold Experiments
echo ========================================

set EPOCHS=30
set BATCH=32
set LR=5e-5
set DROPOUT=0.5

for %%M in (resnet50 mobilenetv2 hybrid) do (
    for %%F in (0 1 2 3 4) do (
        echo.
        echo --- Running pcgita_%%M_fold%%F ---
        python product\training\train_unified.py --dataset pcgita --model_type %%M --fold %%F --epochs %EPOCHS% --batch_size %BATCH% --lr %LR% --dropout %DROPOUT%
        echo --- Finished pcgita_%%M_fold%%F ---
    )
)

echo ========================================
echo All PC-GITA K-Fold runs complete!
echo ========================================
