@echo off
cd /d %~dp0\..\..
REM PhysioNet K-Fold: 3 models x 5 folds = 15 runs (~10 hrs)
echo ========================================
echo  PhysioNet - K-Fold (15 runs)
echo ========================================

set PYTHON=python
set TRAIN=product\training\train_unified.py

for %%M in (resnet50 mobilenetv2 hybrid) do (
    for %%F in (0 1 2 3 4) do (
        echo.
        echo [Running] physionet / %%M / Fold %%F
        %PYTHON% %TRAIN% --dataset physionet --model_type %%M --fold %%F --epochs 30 --batch_size 32
        if errorlevel 1 (
            echo [ERROR] physionet / %%M / Fold %%F >> kfold_errors.log
        )
    )
)

echo.
echo PhysioNet K-Fold complete!
pause

