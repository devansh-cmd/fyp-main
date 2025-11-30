@echo off
REM Run all EmoDB experiments SEQUENTIALLY
REM Train all 5 models on EmoDB dataset (3 seeds each = 15 experiments)

echo ========================================
echo EmoDB Experiments - All Models
echo ========================================
echo.
echo Running 15 experiments SEQUENTIALLY:
echo   - Baseline CNN (seeds: 42, 123, 999)
echo   - AlexNet (seeds: 42, 123, 999)
echo   - ResNet50 (seeds: 42, 123, 999)
echo   - ResNet50+CBAM (seeds: 42, 123, 999)
echo   - ResNet50+SE (seeds: 42, 123, 999)
echo.
echo Each experiment: ~30-60 min
echo Total time: ~10-12 hours (sequential)
echo.
echo Press Ctrl+C to cancel, or any key to start...
pause

echo.
echo Starting EmoDB experiments...
echo.

REM Baseline CNN
echo [1/15] Baseline CNN - seed 42...
.venv312\Scripts\python.exe -m product.training.train_baseline --train_csv product/artifacts/splits/train_emodb.csv --val_csv product/artifacts/splits/val_emodb.csv --epochs 30 --batch_size 64 --lr 5e-4 --seed 42 --run_name emodb_baseline_seed42

echo [2/15] Baseline CNN - seed 123...
.venv312\Scripts\python.exe -m product.training.train_baseline --train_csv product/artifacts/splits/train_emodb.csv --val_csv product/artifacts/splits/val_emodb.csv --epochs 30 --batch_size 64 --lr 5e-4 --seed 123 --run_name emodb_baseline_seed123

echo [3/15] Baseline CNN - seed 999...
.venv312\Scripts\python.exe -m product.training.train_baseline --train_csv product/artifacts/splits/train_emodb.csv --val_csv product/artifacts/splits/val_emodb.csv --epochs 30 --batch_size 64 --lr 5e-4 --seed 999 --run_name emodb_baseline_seed999

REM AlexNet
echo [4/15] AlexNet - seed 42...
.venv312\Scripts\python.exe -m product.training.alexnet_t1 --train_csv product/artifacts/splits/train_emodb.csv --val_csv product/artifacts/splits/val_emodb.csv --epochs 30 --batch_size 64 --lr 5e-4 --seed 42 --run_name emodb_alexnet_seed42

echo [5/15] AlexNet - seed 123...
.venv312\Scripts\python.exe -m product.training.alexnet_t1 --train_csv product/artifacts/splits/train_emodb.csv --val_csv product/artifacts/splits/val_emodb.csv --epochs 30 --batch_size 64 --lr 5e-4 --seed 123 --run_name emodb_alexnet_seed123

echo [6/15] AlexNet - seed 999...
.venv312\Scripts\python.exe -m product.training.alexnet_t1 --train_csv product/artifacts/splits/train_emodb.csv --val_csv product/artifacts/splits/val_emodb.csv --epochs 30 --batch_size 64 --lr 5e-4 --seed 999 --run_name emodb_alexnet_seed999

REM ResNet50
echo [7/15] ResNet50 - seed 42...
.venv312\Scripts\python.exe -m product.training.Resnet50_t1 --train_csv product/artifacts/splits/train_emodb.csv --val_csv product/artifacts/splits/val_emodb.csv --epochs 30 --batch_size 64 --lr 5e-4 --seed 42 --run_name emodb_resnet50_seed42

echo [8/15] ResNet50 - seed 123...
.venv312\Scripts\python.exe -m product.training.Resnet50_t1 --train_csv product/artifacts/splits/train_emodb.csv --val_csv product/artifacts/splits/val_emodb.csv --epochs 30 --batch_size 64 --lr 5e-4 --seed 123 --run_name emodb_resnet50_seed123

echo [9/15] ResNet50 - seed 999...
.venv312\Scripts\python.exe -m product.training.Resnet50_t1 --train_csv product/artifacts/splits/train_emodb.csv --val_csv product/artifacts/splits/val_emodb.csv --epochs 30 --batch_size 64 --lr 5e-4 --seed 999 --run_name emodb_resnet50_seed999

REM ResNet50+CBAM
echo [10/15] ResNet50+CBAM - seed 42...
.venv312\Scripts\python.exe -m product.training.resnet50_cbam --train_csv product/artifacts/splits/train_emodb.csv --val_csv product/artifacts/splits/val_emodb.csv --epochs 30 --batch_size 64 --lr 5e-4 --seed 42 --run_name emodb_cbam_seed42

echo [11/15] ResNet50+CBAM - seed 123...
.venv312\Scripts\python.exe -m product.training.resnet50_cbam --train_csv product/artifacts/splits/train_emodb.csv --val_csv product/artifacts/splits/val_emodb.csv --epochs 30 --batch_size 64 --lr 5e-4 --seed 123 --run_name emodb_cbam_seed123

echo [12/15] ResNet50+CBAM - seed 999...
.venv312\Scripts\python.exe -m product.training.resnet50_cbam --train_csv product/artifacts/splits/train_emodb.csv --val_csv product/artifacts/splits/val_emodb.csv --epochs 30 --batch_size 64 --lr 5e-4 --seed 999 --run_name emodb_cbam_seed999

REM ResNet50+SE
echo [13/15] ResNet50+SE - seed 42...
.venv312\Scripts\python.exe -m product.training.resnet50_se --train_csv product/artifacts/splits/train_emodb.csv --val_csv product/artifacts/splits/val_emodb.csv --epochs 30 --batch_size 64 --lr 5e-4 --seed 42 --run_name emodb_se_seed42

echo [14/15] ResNet50+SE - seed 123...
.venv312\Scripts\python.exe -m product.training.resnet50_se --train_csv product/artifacts/splits/train_emodb.csv --val_csv product/artifacts/splits/val_emodb.csv --epochs 30 --batch_size 64 --lr 5e-4 --seed 123 --run_name emodb_se_seed123

echo [15/15] ResNet50+SE - seed 999...
.venv312\Scripts\python.exe -m product.training.resnet50_se --train_csv product/artifacts/splits/train_emodb.csv --val_csv product/artifacts/splits/val_emodb.csv --epochs 30 --batch_size 64 --lr 5e-4 --seed 999 --run_name emodb_se_seed999

echo.
echo ========================================
echo All 15 EmoDB experiments completed!
echo ========================================
echo.
echo Results saved to product/artifacts/runs/emodb_*
echo.
pause
