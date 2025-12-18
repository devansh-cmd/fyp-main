@echo off
REM Run all ESC-50 attention model experiments SEQUENTIALLY (one after another)
REM This prevents CPU overload by running one experiment at a time

echo ========================================
echo ESC-50 Attention Model Experiments
echo ========================================
echo.
echo Running 6 experiments SEQUENTIALLY:
echo   - ResNet50+CBAM (seeds: 42, 123, 999)
echo   - ResNet50+SE (seeds: 42, 123, 999)
echo.
echo Each experiment: ~2 hours
echo Total time: ~12 hours (sequential)
echo.
echo Press Ctrl+C to cancel, or any key to start...
pause

echo.
echo [0/6] Generating train/validation splits...
python product/datasets/make_split.py --spec_dir product/audio_preprocessing/outputs/spectrograms --esc50_csv product/audio_preprocessing/data/ESC-50/esc50.csv --out_dir product/artifacts/splits --val_ratio 0.2

echo.
echo Starting experiments sequentially...
echo.

REM ResNet50+CBAM - Seed 42
echo [1/6] Running CBAM seed 42...
python -m product.training.resnet50_cbam --train_csv product/artifacts/splits/train.csv --val_csv product/artifacts/splits/val.csv --epochs 30 --batch_size 64 --lr 5e-4 --seed 42 --run_name resnet50_cbam_seed42

REM ResNet50+CBAM - Seed 123
echo [2/6] Running CBAM seed 123...
python -m product.training.resnet50_cbam --train_csv product/artifacts/splits/train.csv --val_csv product/artifacts/splits/val.csv --epochs 30 --batch_size 64 --lr 5e-4 --seed 123 --run_name resnet50_cbam_seed123

REM ResNet50+CBAM - Seed 999
echo [3/6] Running CBAM seed 999...
python -m product.training.resnet50_cbam --train_csv product/artifacts/splits/train.csv --val_csv product/artifacts/splits/val.csv --epochs 30 --batch_size 64 --lr 5e-4 --seed 999 --run_name resnet50_cbam_seed999

REM ResNet50+SE - Seed 42
echo [4/6] Running SE seed 42...
python -m product.training.resnet50_se --train_csv product/artifacts/splits/train.csv --val_csv product/artifacts/splits/val.csv --epochs 30 --batch_size 64 --lr 5e-4 --seed 42 --run_name resnet50_se_seed42

REM ResNet50+SE - Seed 123
echo [5/6] Running SE seed 123...
python -m product.training.resnet50_se --train_csv product/artifacts/splits/train.csv --val_csv product/artifacts/splits/val.csv --epochs 30 --batch_size 64 --lr 5e-4 --seed 123 --run_name resnet50_se_seed123

REM ResNet50+SE - Seed 999
echo [6/6] Running SE seed 999...
python -m product.training.resnet50_se --train_csv product/artifacts/splits/train.csv --val_csv product/artifacts/splits/val.csv --epochs 30 --batch_size 64 --lr 5e-4 --seed 999 --run_name resnet50_se_seed999

echo.
echo ========================================
echo All 6 experiments completed!
echo ========================================
echo.
echo Results saved to:
echo   - product/artifacts/runs/resnet50_cbam_seed42/
echo   - product/artifacts/runs/resnet50_cbam_seed123/
echo   - product/artifacts/runs/resnet50_cbam_seed999/
echo   - product/artifacts/runs/resnet50_se_seed42/
echo   - product/artifacts/runs/resnet50_se_seed123/
echo   - product/artifacts/runs/resnet50_se_seed999/
echo.
pause
