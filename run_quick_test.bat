@echo off
REM Quick test run (5 epochs) to verify scripts work before full experiments

echo ========================================
echo Quick Test - CBAM and SE (5 epochs)
echo ========================================
echo.
echo This will run 2 quick tests in parallel:
echo   - ResNet50+CBAM (5 epochs, ~15 min)
echo   - ResNet50+SE (5 epochs, ~15 min)
echo.
echo Press Ctrl+C to cancel, or any key to start...
pause

if not exist "logs" mkdir logs

echo.
echo Starting test runs...
echo.

start "Test-CBAM" cmd /c "python -m product.training.resnet50_cbam --train_csv product/artifacts/splits/train.csv --val_csv product/artifacts/splits/val.csv --epochs 5 --batch_size 64 --lr 5e-4 --seed 42 --run_name test_cbam_5ep > logs/test_cbam.log 2>&1"

start "Test-SE" cmd /c "python -m product.training.resnet50_se --train_csv product/artifacts/splits/train.csv --val_csv product/artifacts/splits/val.csv --epochs 5 --batch_size 64 --lr 5e-4 --seed 42 --run_name test_se_5ep > logs/test_se.log 2>&1"

echo.
echo Test runs started!
echo Check logs: logs/test_cbam.log and logs/test_se.log
echo.
pause
