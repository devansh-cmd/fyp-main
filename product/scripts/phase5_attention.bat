@echo off
REM Phase 5: Attention Ablation Study
REM Runs CA and Gate on ResNet-50 and HybridNet
REM Only on datasets with headroom: Pitt, Italian PD, PhysioNet, PC-GITA
REM Total: 4 datasets x 2 attention x 2 backbones x 5 folds = 80 runs

echo ========================================
echo Phase 5: Attention Ablation Study
echo ========================================

REM --- Tier 3 Clinical Guard (Pitt) ---
set PITT_LR=1e-5
set PITT_DO=0.7
set PITT_UNFREEZE=10

REM --- Tier 2 Stability (Italian PD, PC-GITA, PhysioNet) ---
set T2_LR=5e-5
set T2_DO=0.5

REM ============ RESNET50 + CA ============
echo.
echo ===== ResNet50 + Coordinate Attention =====

for %%F in (0 1 2 3 4) do (
    echo --- pitt resnet50_ca fold%%F ---
    python product\training\train_unified.py --dataset pitt --model_type resnet50_ca --fold %%F --epochs 30 --lr %PITT_LR% --dropout %PITT_DO% --weight_decay 0.1 --unfreeze_at %PITT_UNFREEZE% --weighted_loss
)

for %%F in (0 1 2 3 4) do (
    echo --- italian_pd resnet50_ca fold%%F ---
    python product\training\train_unified.py --dataset italian_pd --model_type resnet50_ca --fold %%F --epochs 30 --lr %T2_LR% --dropout %T2_DO%
)

for %%F in (0 1 2 3 4) do (
    echo --- physionet resnet50_ca fold%%F ---
    python product\training\train_unified.py --dataset physionet --model_type resnet50_ca --fold %%F --epochs 30 --lr %T2_LR% --dropout %T2_DO%
)

for %%F in (0 1 2 3 4) do (
    echo --- pcgita resnet50_ca fold%%F ---
    python product\training\train_unified.py --dataset pcgita --model_type resnet50_ca --fold %%F --epochs 30 --lr %T2_LR% --dropout %T2_DO%
)

REM ============ RESNET50 + GATE ============
echo.
echo ===== ResNet50 + Attention Gate =====

for %%F in (0 1 2 3 4) do (
    echo --- pitt resnet50_gate fold%%F ---
    python product\training\train_unified.py --dataset pitt --model_type resnet50_gate --fold %%F --epochs 30 --lr %PITT_LR% --dropout %PITT_DO% --weight_decay 0.1 --unfreeze_at %PITT_UNFREEZE% --weighted_loss
)

for %%F in (0 1 2 3 4) do (
    echo --- italian_pd resnet50_gate fold%%F ---
    python product\training\train_unified.py --dataset italian_pd --model_type resnet50_gate --fold %%F --epochs 30 --lr %T2_LR% --dropout %T2_DO%
)

for %%F in (0 1 2 3 4) do (
    echo --- physionet resnet50_gate fold%%F ---
    python product\training\train_unified.py --dataset physionet --model_type resnet50_gate --fold %%F --epochs 30 --lr %T2_LR% --dropout %T2_DO%
)

for %%F in (0 1 2 3 4) do (
    echo --- pcgita resnet50_gate fold%%F ---
    python product\training\train_unified.py --dataset pcgita --model_type resnet50_gate --fold %%F --epochs 30 --lr %T2_LR% --dropout %T2_DO%
)

REM ============ HYBRID + CA ============
echo.
echo ===== HybridNet + Coordinate Attention =====

for %%F in (0 1 2 3 4) do (
    echo --- pitt hybrid_ca fold%%F ---
    python product\training\train_unified.py --dataset pitt --model_type hybrid_ca --fold %%F --epochs 30 --lr %PITT_LR% --dropout %PITT_DO% --weight_decay 0.1 --unfreeze_at %PITT_UNFREEZE% --weighted_loss
)

for %%F in (0 1 2 3 4) do (
    echo --- italian_pd hybrid_ca fold%%F ---
    python product\training\train_unified.py --dataset italian_pd --model_type hybrid_ca --fold %%F --epochs 30 --lr %T2_LR% --dropout %T2_DO%
)

for %%F in (0 1 2 3 4) do (
    echo --- physionet hybrid_ca fold%%F ---
    python product\training\train_unified.py --dataset physionet --model_type hybrid_ca --fold %%F --epochs 30 --lr %T2_LR% --dropout %T2_DO%
)

for %%F in (0 1 2 3 4) do (
    echo --- pcgita hybrid_ca fold%%F ---
    python product\training\train_unified.py --dataset pcgita --model_type hybrid_ca --fold %%F --epochs 30 --lr %T2_LR% --dropout %T2_DO%
)

REM ============ HYBRID + GATE ============
echo.
echo ===== HybridNet + Attention Gate =====

for %%F in (0 1 2 3 4) do (
    echo --- pitt hybrid_gate fold%%F ---
    python product\training\train_unified.py --dataset pitt --model_type hybrid_gate --fold %%F --epochs 30 --lr %PITT_LR% --dropout %PITT_DO% --weight_decay 0.1 --unfreeze_at %PITT_UNFREEZE% --weighted_loss
)

for %%F in (0 1 2 3 4) do (
    echo --- italian_pd hybrid_gate fold%%F ---
    python product\training\train_unified.py --dataset italian_pd --model_type hybrid_gate --fold %%F --epochs 30 --lr %T2_LR% --dropout %T2_DO%
)

for %%F in (0 1 2 3 4) do (
    echo --- physionet hybrid_gate fold%%F ---
    python product\training\train_unified.py --dataset physionet --model_type hybrid_gate --fold %%F --epochs 30 --lr %T2_LR% --dropout %T2_DO%
)

for %%F in (0 1 2 3 4) do (
    echo --- pcgita hybrid_gate fold%%F ---
    python product\training\train_unified.py --dataset pcgita --model_type hybrid_gate --fold %%F --epochs 30 --lr %T2_LR% --dropout %T2_DO%
)

echo ========================================
echo Phase 5 Complete! All 80 attention runs finished.
echo ========================================
