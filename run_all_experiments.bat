@echo off
REM Run all ESC-50 attention model experiments (CBAM + SE, 3 seeds each)
REM This will run 6 training jobs in parallel

echo ========================================
echo Starting ESC-50 Attention Model Experiments
echo ========================================
echo.
echo Running 6 experiments in parallel:
echo   - ResNet50+CBAM (seeds: 42, 123, 999)
echo   - ResNet50+SE (seeds: 42, 123, 999)
echo.
echo Each experiment takes ~2 hours
echo Total time: ~2 hours (parallel execution)
echo.
echo Press Ctrl+C to cancel, or any key to start...
pause

REM Create logs directory
if not exist "logs" mkdir logs

echo.
echo Starting experiments...
echo.

REM ResNet50+CBAM - Seed 42
start "CBAM-42" /MIN cmd /c "python -m product.training.resnet50_cbam --train_csv product/artifacts/splits/train.csv --val_csv product/artifacts/splits/val.csv --epochs 30 --batch_size 64 --lr 5e-4 --seed 42 --run_name resnet50_cbam_seed42 > logs/cbam_seed42.log 2>&1"

REM ResNet50+CBAM - Seed 123
start "CBAM-123" /MIN cmd /c "python -m product.training.resnet50_cbam --train_csv product/artifacts/splits/train.csv --val_csv product/artifacts/splits/val.csv --epochs 30 --batch_size 64 --lr 5e-4 --seed 123 --run_name resnet50_cbam_seed123 > logs/cbam_seed123.log 2>&1"

REM ResNet50+CBAM - Seed 999
start "CBAM-999" /MIN cmd /c "python -m product.training.resnet50_cbam --train_csv product/artifacts/splits/train.csv --val_csv product/artifacts/splits/val.csv --epochs 30 --batch_size 64 --lr 5e-4 --seed 999 --run_name resnet50_cbam_seed999 > logs/cbam_seed999.log 2>&1"

REM ResNet50+SE - Seed 42
start "SE-42" /MIN cmd /c "python -m product.training.resnet50_se --train_csv product/artifacts/splits/train.csv --val_csv product/artifacts/splits/val.csv --epochs 30 --batch_size 64 --lr 5e-4 --seed 42 --run_name resnet50_se_seed42 > logs/se_seed42.log 2>&1"

REM ResNet50+SE - Seed 123
start "SE-123" /MIN cmd /c "python -m product.training.resnet50_se --train_csv product/artifacts/splits/train.csv --val_csv product/artifacts/splits/val.csv --epochs 30 --batch_size 64 --lr 5e-4 --seed 123 --run_name resnet50_se_seed123 > logs/se_seed123.log 2>&1"

REM ResNet50+SE - Seed 999
start "SE-999" /MIN cmd /c "python -m product.training.resnet50_se --train_csv product/artifacts/splits/train.csv --val_csv product/artifacts/splits/val.csv --epochs 30 --batch_size 64 --lr 5e-4 --seed 999 --run_name resnet50_se_seed999 > logs/se_seed999.log 2>&1"

echo.
echo ========================================
echo All 6 experiments started!
echo ========================================
echo.
echo Check progress in separate windows or logs:
echo   - logs/cbam_seed42.log
echo   - logs/cbam_seed123.log
echo   - logs/cbam_seed999.log
echo   - logs/se_seed42.log
echo   - logs/se_seed123.log
echo   - logs/se_seed999.log
echo.
echo Results will be saved to:
echo   - product/artifacts/runs/resnet50_cbam_seed42/
echo   - product/artifacts/runs/resnet50_cbam_seed123/
echo   - product/artifacts/runs/resnet50_cbam_seed999/
echo   - product/artifacts/runs/resnet50_se_seed42/
echo   - product/artifacts/runs/resnet50_se_seed123/
echo   - product/artifacts/runs/resnet50_se_seed999/
echo.
echo Press any key to exit this window...
pause
