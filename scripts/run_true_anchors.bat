@echo off
cd /d "%~dp0.."
setlocal enabledelayedexpansion

:: The Bias-Correction Protocol (Establishing True Golden Anchors)
:: Goal: Benchmarking with class-weighting to establish legitimate clinical baselines
:: Protocols: 30 Epochs, Weighted Loss Enabled, 3 Seeds

echo ==========================================================
echo STARTING THE BIAS-CORRECTION PROTOCOL: TRUE GOLDEN ANCHORS (30 RUNS)
echo ==========================================================

:: Models to benchmark
set MODELS=resnet50 mobilenetv2

:: Datasets in priority order
for %%d in (pitt esc50 italian_pd physionet emodb) do (
    for %%m in (!MODELS!) do (
        for %%s in (42 123 999) do (
            echo [RUNNING] Dataset: %%d ^| Model: %%m ^| Seed: %%s
            
            if "%%d"=="pitt" (
                :: "Golden Baseline" Pitt Optimization (Balanced Capacity)
                python product/training/train_unified.py --dataset %%d --model_type %%m --seed %%s --epochs 30 --batch_size 32 --weighted_loss --lr 2e-5 --weight_decay 0.1 --dropout 0.6
            ) else (
                :: High-Regularization Anchor Protocol (Anti-Overfitting)
                python product/training/train_unified.py --dataset %%d --model_type %%m --seed %%s --epochs 30 --batch_size 32 --weighted_loss --lr 5e-6 --weight_decay 0.1 --dropout 0.8
            )
        )
    )
)

echo ==========================================================
echo TRUE ANCHOR RUNS COMPLETE. AGGREGATING RESULTS...
echo ==========================================================

python product/training/aggregate_seeds.py

echo DONE.
pause
