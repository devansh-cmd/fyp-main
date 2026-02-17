@echo off
cd /d %~dp0\..\..
REM ============================================================
REM  K-Fold Cross-Validation Experiment Matrix
REM  5 Datasets x 3 Models x 5 Folds = 75 Runs
REM ============================================================

echo ========================================
echo  K-Fold Cross-Validation: 75-Run Matrix
echo ========================================

set PYTHON=python
set TRAIN_SCRIPT=product\training\train_unified.py
set EPOCHS=30
set BATCH_SIZE=32

REM --- Datasets and their specific flags ---
set DATASETS=esc50 emodb italian_pd physionet pitt
set MODELS=resnet50 mobilenetv2 hybrid

REM --- Loop through all combinations ---
for %%D in (esc50 emodb italian_pd physionet pitt) do (
    for %%M in (resnet50 mobilenetv2 hybrid) do (
        for %%F in (0 1 2 3 4) do (
            echo.
            echo ============================================================
            echo  Running: %%D / %%M / Fold %%F
            echo ============================================================
            
            REM Pitt Corpus uses weighted loss due to class imbalance
            if "%%D"=="pitt" (
                %PYTHON% %TRAIN_SCRIPT% --dataset %%D --model_type %%M --fold %%F --epochs %EPOCHS% --batch_size %BATCH_SIZE% --weighted_loss
            ) else (
                %PYTHON% %TRAIN_SCRIPT% --dataset %%D --model_type %%M --fold %%F --epochs %EPOCHS% --batch_size %BATCH_SIZE%
            )
            
            if errorlevel 1 (
                echo [ERROR] Failed: %%D / %%M / Fold %%F
                echo Failed: %%D / %%M / Fold %%F >> kfold_errors.log
            ) else (
                echo [OK] Completed: %%D / %%M / Fold %%F
            )
        )
    )
)

echo.
echo ============================================================
echo  All 75 runs complete. Aggregating results...
echo ============================================================

%PYTHON% product\scripts\aggregate_kfold_results.py --n_folds 5

echo.
echo Done! Results saved to product\artifacts\kfold_results.csv
pause
