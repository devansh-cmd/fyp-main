@echo off
cd /d "%~dp0.."
setlocal enabledelayedexpansion

:: Deep Hybrid Ensemble Benchmarking
:: Dual-Backbone: ResNet50 + MobileNetV2
:: Fusion Strategy: Learnable Gated Fusion (alpha-mask)
:: Protocol: 30 Epochs, Batch Size 16 (Verified Stable), 15 Runs

echo ==========================================================
echo STARTING HYBRID ENSEMBLE BENCHMARKS (30 RUNS TOTAL)
echo ==========================================================

:: Datasets in priority order for the research narrative
for %%d in (pitt esc50 italian_pd physionet emodb) do (
    for %%s in (42 123 999) do (
        echo [RUNNING] Dataset: %%d ^| Model: hybrid ^| Seed: %%s
        python product/training/train_unified.py --dataset %%d --model_type hybrid --seed %%s --epochs 30 --batch_size 16 --lr 1e-5 --weight_decay 5e-2 --unfreeze_at 10 --weighted_loss --dropout 0.7
    )
)

echo ==========================================================
echo HYBRID ENSEMBLE RUNS COMPLETE. AGGREGATING FINAL RESULTS...
echo ==========================================================

python product/training/aggregate_seeds.py

echo DONE.
pause
