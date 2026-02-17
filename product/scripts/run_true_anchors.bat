@echo off
cd /d "%~dp0..\..\.."
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
                :: High-Risk Gold Protocol (Pitt): Delayed Unfreezing
                python product/training/train_unified.py --dataset %%d --model_type %%m --seed %%s --epochs 30 --batch_size 32 --weighted_loss --lr 1e-5 --weight_decay 0.1 --dropout 0.7 --unfreeze_at 10
            ) else if "%%d"=="physionet" (
                :: High-Risk Gold Protocol (PhysioNet): Delayed Unfreezing
                python product/training/train_unified.py --dataset %%d --model_type %%m --seed %%s --epochs 30 --batch_size 32 --weighted_loss --lr 1e-5 --weight_decay 0.1 --dropout 0.7 --unfreeze_at 10
            ) else if "%%d"=="italian_pd" (
                :: Stable Clinical Recovery (PD)
                python product/training/train_unified.py --dataset %%d --model_type %%m --seed %%s --epochs 30 --batch_size 32 --weighted_loss --lr 5e-5 --weight_decay 0.01 --dropout 0.5
            ) else (
                :: High-Capacity Recovery (ESC-50, EmoDB)
                python product/training/train_unified.py --dataset %%d --model_type %%m --seed %%s --epochs 30 --batch_size 32 --weighted_loss --lr 1e-4 --weight_decay 0.01 --dropout 0.5
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

