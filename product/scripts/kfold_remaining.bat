@echo off
setlocal enabledelayedexpansion
cd /d %~dp0\..\..
REM ========================================
REM  ALL REMAINING K-FOLD RUNS (33 total)
REM  Auto-skips completed runs
REM ========================================

set PYTHON=python
set TRAIN=product\training\train_unified.py
set COUNT=0
set TOTAL=33

echo ========================================
echo  Remaining K-Fold Runs: %TOTAL%
echo ========================================
echo.
echo  Pitt:  13 runs (resnet50 f2-4, mobilenetv2 f0-4, hybrid f0-4)
echo  ESC50:  5 runs (hybrid f0-4)
echo  EmoDB: 15 runs (all models f0-4)
echo.
echo  Italian PD: COMPLETE
echo  PhysioNet:  COMPLETE
echo ========================================
echo.

REM ---- PITT (13 remaining) ----
REM Clinical protocol: aggressive regularisation
echo [DATASET] Pitt Corpus
for %%M in (resnet50 mobilenetv2 hybrid) do (
    for %%F in (0 1 2 3 4) do (
        if not exist "product\artifacts\runs\pitt\pitt_%%M_fold%%F\summary.json" (
            set /a COUNT+=1
            echo.
            echo [!COUNT!/%TOTAL%] pitt / %%M / fold%%F
            %PYTHON% %TRAIN% --dataset pitt --model_type %%M --fold %%F --epochs 30 --batch_size 32 --weighted_loss --lr 1e-5 --dropout 0.7 --weight_decay 0.1 --unfreeze_at 10
            if errorlevel 1 (
                echo [ERROR] pitt / %%M / fold%%F >> kfold_errors.log
            )
        ) else (
            echo [SKIP] pitt / %%M / fold%%F (already done^)
        )
    )
)

REM ---- ESC-50 (5 remaining) ----
echo.
echo [DATASET] ESC-50
for %%M in (resnet50 mobilenetv2 hybrid) do (
    for %%F in (0 1 2 3 4) do (
        if not exist "product\artifacts\runs\esc50\esc50_%%M_fold%%F\summary.json" (
            set /a COUNT+=1
            echo.
            echo [!COUNT!/%TOTAL%] esc50 / %%M / fold%%F
            %PYTHON% %TRAIN% --dataset esc50 --model_type %%M --fold %%F --epochs 30 --batch_size 32
            if errorlevel 1 (
                echo [ERROR] esc50 / %%M / fold%%F >> kfold_errors.log
            )
        ) else (
            echo [SKIP] esc50 / %%M / fold%%F (already done^)
        )
    )
)

REM ---- EmoDB (15 remaining) ----
echo.
echo [DATASET] EmoDB
for %%M in (resnet50 mobilenetv2 hybrid) do (
    for %%F in (0 1 2 3 4) do (
        if not exist "product\artifacts\runs\emodb\emodb_%%M_fold%%F\summary.json" (
            set /a COUNT+=1
            echo.
            echo [!COUNT!/%TOTAL%] emodb / %%M / fold%%F
            %PYTHON% %TRAIN% --dataset emodb --model_type %%M --fold %%F --epochs 30 --batch_size 32
            if errorlevel 1 (
                echo [ERROR] emodb / %%M / fold%%F >> kfold_errors.log
            )
        ) else (
            echo [SKIP] emodb / %%M / fold%%F (already done^)
        )
    )
)

echo.
echo ========================================
echo  ALL REMAINING K-FOLD RUNS COMPLETE
echo ========================================
if exist kfold_errors.log (
    echo.
    echo ERRORS LOGGED:
    type kfold_errors.log
)

