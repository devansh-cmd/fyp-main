@echo off
cd /d %~dp0\..\..
REM Pitt Corpus K-Fold RESUME: Skips fold0/fold1 for resnet50 (already complete)
echo ========================================
echo  Pitt Corpus - K-Fold RESUME (13 runs)
echo ========================================

set PYTHON=python
set TRAIN=product\training\train_unified.py
set ARGS=--dataset pitt --epochs 30 --batch_size 32 --weighted_loss --lr 1e-5 --dropout 0.7 --weight_decay 0.1 --unfreeze_at 10

REM ResNet50: folds 2,3,4 remaining
for %%F in (2 3 4) do (
    echo.
    echo [Running] pitt / resnet50 / Fold %%F
    %PYTHON% %TRAIN% --model_type resnet50 --fold %%F %ARGS%
    if errorlevel 1 (
        echo [ERROR] pitt / resnet50 / Fold %%F >> kfold_errors.log
    )
)

REM MobileNetV2: all 5 folds
for %%F in (0 1 2 3 4) do (
    echo.
    echo [Running] pitt / mobilenetv2 / Fold %%F
    %PYTHON% %TRAIN% --model_type mobilenetv2 --fold %%F %ARGS%
    if errorlevel 1 (
        echo [ERROR] pitt / mobilenetv2 / Fold %%F >> kfold_errors.log
    )
)

REM Hybrid: all 5 folds
for %%F in (0 1 2 3 4) do (
    echo.
    echo [Running] pitt / hybrid / Fold %%F
    %PYTHON% %TRAIN% --model_type hybrid --fold %%F %ARGS%
    if errorlevel 1 (
        echo [ERROR] pitt / hybrid / Fold %%F >> kfold_errors.log
    )
)

echo.
echo Pitt Corpus K-Fold RESUME complete!
pause
