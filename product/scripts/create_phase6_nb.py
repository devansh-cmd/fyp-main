"""
Regenerate phase6_sota_push_kaggle.ipynb.
Run this script locally to update the notebook before packing for Kaggle.
"""
import json
from pathlib import Path

NB_PATH = Path(__file__).parent.parent / "notebooks" / "phase6_sota_push_kaggle.ipynb"

SETUP_CELL = r"""import os
import shutil
import zipfile
import subprocess
import sys
from pathlib import Path

KAGGLE_INPUT_DIR = Path('/kaggle/input')
WORK_DIR = Path('/kaggle/working/PROJECT')

if WORK_DIR.exists():
    shutil.rmtree(WORK_DIR)
WORK_DIR.mkdir(parents=True, exist_ok=True)

# Check if Kaggle kept it as a zip or auto-extracted it
zips = list(KAGGLE_INPUT_DIR.glob('**/*.zip'))

if zips:
    ZIP_PATH = zips[0]
    print(f'Found zip: {ZIP_PATH}')
    print('Extracting... (may take 2-3 mins)')
    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        zf.extractall('/kaggle/working/')
    print('Done extracting.')
else:
    print('No zip found — assuming Kaggle auto-extracted the dataset.')
    product_dirs = list(KAGGLE_INPUT_DIR.glob('**/product'))
    if not product_dirs:
        raise FileNotFoundError('Could not find the dataset contents!')
    src_dir = product_dirs[0].parent
    print(f'Copying files from {src_dir} to {WORK_DIR}...')
    shutil.copytree(src_dir, WORK_DIR, dirs_exist_ok=True)
    print('Done copying.')
"""

PATH_FIX_CELL = r"""os.chdir(WORK_DIR)
print(f'CWD: {os.getcwd()}')

ROOT = WORK_DIR
SPLITS = ROOT / 'product/artifacts/splits'
WIN_PREFIX = r'C:\FYP\PROJECT'

fixed = 0
for csv_file in SPLITS.glob('*.csv'):
    text = csv_file.read_text()
    if WIN_PREFIX in text or '\\' in text:
        text = text.replace(WIN_PREFIX, str(ROOT))
        text = text.replace('\\', '/')
        csv_file.write_text(text)
        fixed += 1
print(f'Fixed paths in {fixed} CSV files.')
"""

CLEANUP_CELL = r"""import shutil
runs_root = ROOT / "product/artifacts/runs"
cleaned = 0
PHASE6_TAGS = ["_resnet50_", "_resnet50_ca_", "_resnet50_ca_lstm_", "_dual_cnn_lstm_"]
for ds in ["italian_pd"]:
    ds_dir = runs_root / ds
    if not ds_dir.exists():
        continue
    for run_dir in list(ds_dir.iterdir()):
        name = run_dir.name
        # Only wipe runs that collide with our naming scheme (same model, seed, fold)
        # — leave other datasets/seeds untouched
        if any(tag in f"_{name}_" for tag in PHASE6_TAGS) and "_s42_" in name:
            shutil.rmtree(run_dir)
            cleaned += 1
print(f"Cleaned {cleaned} stale Phase 6 run directories.")
"""

EXPERIMENT_CELL = r"""# ============================================================
# Phase 6 Stage A: Italian PD only, seed=42, 5 folds
# 3 models × 5 folds = 15 runs
# Goal: Verify ResNetBiLSTM beats ResNet50_CA (>0.930 F1)
# ============================================================

SPEC_ARGS = ["--spec_augment", "--spec_aug_T", "40", "--spec_aug_F", "40"]

# lr tuned lower for LSTM arms (slower convergence with temporal head)
STANDARD_ARGS = ["--lr", "1e-4", "--dropout", "0.5", "--unfreeze_at", "0"] + SPEC_ARGS
LSTM_ARGS     = ["--lr", "5e-5", "--dropout", "0.5", "--unfreeze_at", "5"] + SPEC_ARGS

# (model_type, extra_args, epochs)
EXPERIMENTS = [
    ("resnet50",         STANDARD_ARGS, 15),  # reference baseline + SpecAugment
    ("resnet50_ca",      STANDARD_ARGS, 15),  # best Phase 5 + SpecAugment
    ("resnet50_ca_lstm", LSTM_ARGS,     20),  # SOTA push: CA + BiLSTM head (Phase 6 MAIN)
]

DATASET = "italian_pd"
SEED    = 42

total = len(EXPERIMENTS) * 5
done  = 0
print(f"Phase 6 Stage A: {total} runs ({len(EXPERIMENTS)} models × 5 folds on {DATASET})")

for model_type, extra_args, epochs in EXPERIMENTS:
    for fold in range(5):
        run_name = f"{DATASET}_{model_type}_s{SEED}_fold{fold}"
        summary_path = ROOT / f"product/artifacts/runs/{DATASET}/{run_name}/summary.json"
print(f"Phase 6 Stage B: {total} runs ({len(EXPERIMENTS)} models × {len(SEEDS)} seeds × 5 folds on {DATASET})")

for exp in EXPERIMENTS:
    for seed in SEEDS:
        for fold in range(5):
            run_name = f"{DATASET}_{exp['name']}_s{seed}_fold{fold}"
            summary_path = ROOT / f"product/artifacts/runs/{DATASET}/{run_name}/summary.json"

            if summary_path.exists():
                print(f"SKIP {run_name} (already done)")
                done += 1
                continue

            print(f"\n>>> RUNNING {run_name} (epochs={exp['epochs']})")
            cmd = [
                sys.executable, "-u", "product/training/train_unified.py",
                "--dataset",    DATASET,
                "--model_type", exp['name'],
                "--fold",       str(fold),
                "--epochs",     str(exp['epochs']),
                "--batch_size", "16",
                "--seed",       str(seed),
                "--run_name",   run_name,
                "--drop_last",
                "--lr",         str(exp['lr']),
                "--weight_decay", str(exp['wd']),
                "--spec_augment",
                "--label_smoothing", str(exp['label_smoothing']),
            ]

            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1
            )
            for line in process.stdout:
                print(line, end="")
            process.wait()

            status = "DONE" if process.returncode == 0 else "ERROR"
            print(f"--- {status}: {run_name} ---")
            done += 1
            print(f"Progress: {done}/{total}")
"""

AGGREGATE_CELL = r"""# Aggregate Stage A results into a summary table
import glob

print("\n=== Phase 6 Stage A Results (Italian PD) ===\n")
print(f"{'Model':<25} {'Fold':<6} {'Best F1':>8} {'Final F1':>9} {'AUC':>7}")
print("-" * 60)

import json as _json
from pathlib import Path as _P

runs_root = ROOT / "product/artifacts/runs/italian_pd"
for model_type, _, _ in EXPERIMENTS:
    fold_f1s = []
    for fold in range(5):
        run_name = f"italian_pd_{model_type}_s42_fold{fold}"
        summary_p = runs_root / run_name / "summary.json"
        if summary_p.exists():
            s = _json.loads(summary_p.read_text())
            f1  = s.get("best_macro_f1", 0.0)
            ff1 = s.get("final_macro_f1", 0.0)
            auc = s.get("final_auc", 0.0)
            fold_f1s.append(f1)
            print(f"{model_type:<25} {fold:<6} {f1:>8.4f} {ff1:>9.4f} {auc:>7.4f}")
        else:
            print(f"{model_type:<25} {fold:<6} {'MISSING':>8}")
    if fold_f1s:
        import numpy as np
        print(f"  → mean ± std: {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")
    print()
"""

ZIP_CELL = r"""# Zip all results for download
output_zip = '/kaggle/working/phase6_results.zip'
print(f'Zipping results...')

with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
    runs_dir = ROOT / 'product/artifacts/runs'
    for f in runs_dir.glob('**/*'):
        if f.is_file() and f.suffix != '.pt':  # skip large model weights
            zipf.write(f, f.relative_to(runs_dir.parent.parent))

print(f'Done! Download phase6_results.zip from the Output tab.')
print(f'Stage A complete: 15 runs on Italian PD.')
print(f'EXIT CRITERIA: resnet50_ca_lstm must achieve F1 > 0.930 to proceed to Stage B.')
"""

def make_code_cell(source, cell_id):
    lines = [l + "\n" for l in source.strip().splitlines()]
    lines[-1] = lines[-1].rstrip("\n")  # no trailing newline on last line
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": lines,
    }

def make_md_cell(text, cell_id):
    lines = [l + "\n" for l in text.strip().splitlines()]
    lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": lines,
    }

TITLE_MD = """\
# Phase 6: SOTA Push — ResNet+CA+BiLSTM + SpecAugment

**Stage A**: Italian PD only | seed=42 | 5 folds | 3 models × 5 folds = **15 runs**

| Model | Role |
|---|---|
| `resnet50` + SpecAugment | Reference baseline with augmentation |
| `resnet50_ca` + SpecAugment | Best Phase 5 attention + augmentation |
| `resnet50_ca_lstm` + SpecAugment | **SOTA push: CA + BiLSTM temporal head** |

### Exit Criteria
To proceed to full 3-seed Stage B, `resnet50_ca_lstm` must achieve:
1. Macro F1 **> 0.930** (meaningful uplift over Phase 5 baseline 0.926)
2. Statistical significance vs CA baseline in paired Wilcoxon test.
"""

nb = {
    "cells": [
        make_md_cell(TITLE_MD, "title"),
        make_code_cell(SETUP_CELL, "setup"),
        make_code_cell(PATH_FIX_CELL, "path_fix"),
        make_code_cell(CLEANUP_CELL, "cleanup"),
        make_code_cell(EXPERIMENT_CELL, "experiments"),
        make_code_cell(AGGREGATE_CELL, "aggregate"),
        make_code_cell(ZIP_CELL, "zip_results"),
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB_PATH.write_text(json.dumps(nb, indent=4, ensure_ascii=False), encoding="utf-8")
print(f"Written: {NB_PATH}  ({NB_PATH.stat().st_size:,} bytes)")
print(f"Cells: {len(nb['cells'])}")
