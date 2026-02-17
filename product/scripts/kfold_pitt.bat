@echo off
cd /d %~dp0\..\..
REM Pitt Corpus K-Fold: 3 models x 5 folds = 15 runs (~12.5 hrs)
echo ========================================
echo  Pitt Corpus - K-Fold (15 runs)
echo ========================================

set PYTHON=python
set TRAIN=product\training\train_unified.py

for %%M in (resnet50 mobilenetv2 hybrid) do (
    for %%F in (0 1 2 3 4) do (
        echo.
        echo [Running] pitt / %%M / Fold %%F
        %PYTHON% %TRAIN% --dataset pitt --model_type %%M --fold %%F --epochs 30 --batch_size 32 --weighted_loss --lr 1e-5 --dropout 0.7 --weight_decay 0.1 --unfreeze_at 10
        if errorlevel 1 (
            echo [ERROR] pitt / %%M / Fold %%F >> kfold_errors.log
        )
    )
)

echo.
echo Pitt Corpus K-Fold complete!
pause

