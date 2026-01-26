@echo off
:: Batch script to run Gold Standard Anchor Baselines (3 Seeds each)

echo --- STARTING ANCHOR RUNS: ITALIAN PD ---
python product/training/train_unified.py --dataset italian_pd --model_type resnet50 --seed 42 --run_name anchor_pd_s42
python product/training/train_unified.py --dataset italian_pd --model_type resnet50 --seed 123 --run_name anchor_pd_s123
python product/training/train_unified.py --dataset italian_pd --model_type resnet50 --seed 999 --run_name anchor_pd_s999

echo --- STARTING ANCHOR RUNS: PHYSIONET ---
python product/training/train_unified.py --dataset physionet --model_type resnet50 --seed 42 --run_name anchor_phys_s42
python product/training/train_unified.py --dataset physionet --model_type resnet50 --seed 123 --run_name anchor_phys_s123
python product/training/train_unified.py --dataset physionet --model_type resnet50 --seed 999 --run_name anchor_phys_s999

echo --- STARTING ANCHOR RUNS: EMODB ---
python product/training/train_unified.py --dataset emodb --model_type resnet50 --seed 42 --run_name anchor_emodb_s42
python product/training/train_unified.py --dataset emodb --model_type resnet50 --seed 123 --run_name anchor_emodb_s123
python product/training/train_unified.py --dataset emodb --model_type resnet50 --seed 999 --run_name anchor_emodb_s999

echo --- STARTING ANCHOR RUNS: ESC-50 ---
python product/training/train_unified.py --dataset esc50 --model_type resnet50 --seed 42 --run_name anchor_esc50_s42
python product/training/train_unified.py --dataset esc50 --model_type resnet50 --seed 123 --run_name anchor_esc50_s123
python product/training/train_unified.py --dataset esc50 --model_type resnet50 --seed 999 --run_name anchor_esc50_s999

echo ALL ANCHOR RUNS COMPLETE.
