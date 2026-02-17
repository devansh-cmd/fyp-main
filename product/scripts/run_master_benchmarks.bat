@echo off
cd /d "%~dp0..\..\.."
setlocal enabledelayedexpansion

:: ==========================================================
:: THE "BIG 45" MASTER BENCHMARKING SUITE
:: Goal: Execute all Anchors and Hybrid models in one go
:: Protocol: Weighted Loss (Bias-Correction) + 3 Seeds Matrix
:: ==========================================================

echo ==========================================================
echo [1/2] STARTING TRUE GOLDEN ANCHORS (30 RUNS)
echo ==========================================================

set ANCHOR_MODELS=resnet50 mobilenetv2

for %%d in (pitt esc50 italian_pd physionet emodb) do (
    for %%m in (!ANCHOR_MODELS!) do (
        for %%s in (42 123 999) do (
            set SKIP_RUN=0
            if "%%d"=="italian_pd" if "%%s"=="999" set SKIP_RUN=1
            
            if "!SKIP_RUN!"=="1" (
                echo [SKIPPING] Dataset: %%d ^| Seed: %%s ^(Time Constraint^)
            ) else (
                echo [RUNNING] Dataset: %%d ^| Model: %%m ^| Seed: %%s
                
                if "%%d"=="pitt" (
                    :: Tier 3: High-Risk Gold (Pitt)
                    python product/training/train_unified.py --dataset %%d --model_type %%m --seed %%s --epochs 30 --batch_size 32 --weighted_loss --lr 1e-5 --dropout 0.7 --weight_decay 0.1 --unfreeze_at 10
                ) else if "%%d"=="physionet" (
                    :: Tier 3: High-Risk Gold (PhysioNet)
                    python product/training/train_unified.py --dataset %%d --model_type %%m --seed %%s --epochs 30 --batch_size 32 --weighted_loss --lr 1e-5 --dropout 0.7 --weight_decay 0.1 --unfreeze_at 10
                ) else if "%%d"=="italian_pd" (
                    :: Tier 2: Stable Clinical (PD)
                    python product/training/train_unified.py --dataset %%d --model_type %%m --seed %%s --epochs 30 --batch_size 32 --weighted_loss --lr 5e-5 --dropout 0.5 --weight_decay 0.05 --unfreeze_at 0
                ) else (
                    :: Tier 1: High-Capacity (ESC-50, EmoDB)
                    python product/training/train_unified.py --dataset %%d --model_type %%m --seed %%s --epochs 30 --batch_size 32 --weighted_loss --lr 1e-4 --dropout 0.5 --weight_decay 0.01 --unfreeze_at 0
                )
            )
        )
    )
)

echo ==========================================================
echo [2/2] STARTING HYBRID ENSEMBLE MARATHON (15 RUNS)
echo ==========================================================

for %%d in (pitt esc50 italian_pd physionet emodb) do (
    for %%s in (42 123 999) do (
        set SKIP_RUN=0
        if "%%d"=="italian_pd" if "%%s"=="999" set SKIP_RUN=1
        
        if "!SKIP_RUN!"=="1" (
            echo [SKIPPING] Dataset: %%d ^| Seed: %%s ^(Time Constraint^)
        ) else (
            echo [RUNNING] Dataset: %%d ^| Model: hybrid ^| Seed: %%s
            :: Hybrid specialized flags: low LR, high Weight Decay, Clinical Warm-up
            python product/training/train_unified.py --dataset %%d --model_type hybrid --seed %%s --epochs 30 --batch_size 16 --weighted_loss --lr 1e-5 --weight_decay 5e-2 --unfreeze_at 10 --dropout 0.7
        )
    )
)

echo ==========================================================
echo MASTER BENCHMARKS COMPLETE. AGGREGATING FINAL COMPARISON...
echo ==========================================================

python product/training/aggregate_seeds.py

echo ALL RUNS FINISHED. REPORT READY IN artifacts/aggregated_results_gold.csv
pause

