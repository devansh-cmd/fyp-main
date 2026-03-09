"""
Regenerate phase6_sota_push_kaggle.ipynb.
Run this script locally to update the notebook before packing for Kaggle.
"""
import json
from pathlib import Path

NB_PATH = Path(__file__).parent.parent / "notebooks" / "phase6_sota_push_kaggle.ipynb"

# ── CHANGE THIS to match your Kaggle username and dataset slug ──
KAGGLE_USER = "devanshdev01"
DATASET_SLUG = "phase6-italian-pd-clean"

SETUP_CELL = r"""import os
import shutil
import zipfile
import subprocess
import sys
from pathlib import Path

# ── Hardcoded dataset path — avoids stale-file bugs from Kaggle versioning ──
KAGGLE_USER = '""" + KAGGLE_USER + r"""'
DATASET_SLUG = '""" + DATASET_SLUG + r"""'
EXPECTED_DIR = Path('/kaggle/input') / 'datasets' / KAGGLE_USER / DATASET_SLUG
if not EXPECTED_DIR.exists():
    # Fallback: some Kaggle setups mount at /kaggle/input/SLUG directly
    EXPECTED_DIR = Path('/kaggle/input') / DATASET_SLUG
WORK_DIR = Path('/kaggle/working/PROJECT')

if WORK_DIR.exists():
    shutil.rmtree(WORK_DIR)
WORK_DIR.mkdir(parents=True, exist_ok=True)

# Look for a zip first, then assume auto-extracted
zips = list(EXPECTED_DIR.glob('*.zip'))

if zips:
    ZIP_PATH = zips[0]
    print(f'Found zip: {ZIP_PATH}')
    print('Extracting... (may take 2-3 mins)')
    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        zf.extractall('/kaggle/working/')
    print('Done extracting.')
else:
    print(f'No zip found — copying from {EXPECTED_DIR}...')
    if not EXPECTED_DIR.exists():
        raise FileNotFoundError(
            f'Dataset not found at {EXPECTED_DIR}! '
            f'Make sure the Kaggle dataset slug is "{DATASET_SLUG}"'
        )
    shutil.copytree(EXPECTED_DIR, WORK_DIR, dirs_exist_ok=True)
    print('Done copying.')
"""

PATH_FIX_CELL = r"""os.chdir(WORK_DIR)
print(f'CWD: {os.getcwd()}')

ROOT = WORK_DIR
SPLITS = ROOT / 'product/artifacts/splits'

# ── Leakage guard: assert 0 subject overlap for every fold ──
import pandas as pd
leakage_found = False
for fold in range(5):
    train_csv = SPLITS / f'train_italian_fold{fold}.csv'
    val_csv = SPLITS / f'val_italian_fold{fold}.csv'
    if not train_csv.exists() or not val_csv.exists():
        print(f'WARN: fold{fold} CSVs missing, skipping check')
        continue
    t_subj = set(pd.read_csv(train_csv)['subject_id'])
    v_subj = set(pd.read_csv(val_csv)['subject_id'])
    overlap = t_subj & v_subj
    if overlap:
        print(f'CRITICAL: fold{fold} has {len(overlap)} overlapping subjects: {overlap}')
        leakage_found = True
    else:
        print(f'fold{fold}: OK (train={len(t_subj)} subjects, val={len(v_subj)} subjects, overlap=0)')

if leakage_found:
    raise RuntimeError('DATA LEAKAGE DETECTED — aborting. Delete and re-upload the Kaggle dataset.')
print('All folds passed leakage check.')
"""

CLEANUP_CELL = r"""import shutil
runs_root = ROOT / "product/artifacts/runs"
cleaned = 0
PHASE6_TAGS = ["_resnet50_ca_lstm_", "_dual_cnn_sa_lstm_"]
for ds in ["italian_pd"]:
    ds_dir = runs_root / ds
    if not ds_dir.exists():
        continue
    for run_dir in list(ds_dir.iterdir()):
        name = run_dir.name
        if any(tag in f"_{name}_" for tag in PHASE6_TAGS):
            shutil.rmtree(run_dir)
            cleaned += 1
print(f"Cleaned {cleaned} stale Phase 6 run directories.")
"""

SEEDS = [42, 123, 999]

EXPERIMENTS = [
    {
        "name": "resnet50_ca_lstm",
        "epochs": 20,
        "lr": 1e-4,
        "wd": 1e-2,
        "label_smoothing": 0.05,
    },
    {
        "name": "dual_cnn_sa_lstm",
        "epochs": 20,
        "lr": 1e-4,
        "wd": 1e-2,
        "label_smoothing": 0.05,
    }
]

DATASET = "italian_pd"
total = len(EXPERIMENTS) * len(SEEDS) * 5

# Build the experiment block dynamically so we don't have scoping issues
EXPERIMENT_CELL = f"""# ============================================================
# Phase 6 Stage B: Italian PD, 3 seeds, 5 folds
# 2 models × 3 seeds × 5 folds = 30 runs
# Goal: Statistical validation of dual_cnn_sa_lstm and resnet50_ca_lstm with 0.05 label smoothing
# ============================================================
import sys
import subprocess
from pathlib import Path

ROOT = Path('/kaggle/working/PROJECT')
total = {total}
done = 0
print(f"Phase 6 Stage B: {{total}} runs ({len(EXPERIMENTS)} models × {len(SEEDS)} seeds × 5 folds on {DATASET})")

"""

for exp in EXPERIMENTS:
    for seed in SEEDS:
        for fold in range(5):
            run_name = f"{DATASET}_{exp['name']}_s{seed}_fold{fold}"
            EXPERIMENT_CELL += f"""
run_name = "{run_name}"
summary_path = ROOT / f"product/artifacts/runs/italian_pd/{{run_name}}/summary.json"

if summary_path.exists():
    print(f"SKIP {{run_name}} (already done)")
    done += 1
else:
    print(f"\\n>>> RUNNING {{run_name}}")
    cmd = [
        sys.executable, "-u", "product/training/train_unified.py",
        "--dataset", "{DATASET}",
        "--model_type", "{exp['name']}",
        "--fold", "{fold}",
        "--epochs", "{exp['epochs']}",
        "--batch_size", "16",
        "--seed", "{seed}",
        "--run_name", "{run_name}",
        "--drop_last",
        "--lr", "{exp['lr']}",
        "--weight_decay", "{exp['wd']}",
        "--spec_augment",
        "--label_smoothing", "{exp['label_smoothing']}",
    ]

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in process.stdout:
        print(line, end="")
    process.wait()

    status = "DONE" if process.returncode == 0 else "ERROR"
    print(f"--- {{status}}: {{run_name}} ---")
    done += 1
    print(f"Progress: {{done}}/{{total}}")
"""


AGGREGATE_CELL = r"""# Aggregate Stage A results into a summary table
import json as _json
import numpy as np
from pathlib import Path as _P

print("\n=== Phase 6 Stage B Results (Italian PD) ===\n")
print(f"{'Model':<25} {'Fold':<6} {'Best F1':>8} {'Final F1':>9} {'AUC':>7}")
print("-" * 60)

runs_root = ROOT / "product/artifacts/runs/italian_pd"
MODEL_NAMES = """ + repr([e["name"] for e in EXPERIMENTS]) + r"""
SEEDS = """ + repr(SEEDS) + r"""

for model_type in MODEL_NAMES:
    for seed in SEEDS:
        fold_f1s = []
        for fold in range(5):
            run_name = f"italian_pd_{model_type}_s{seed}_fold{fold}"
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
# Phase 6: SOTA Push — Stage B

**Stage B**: Italian PD only | 3 seeds (42, 123, 999) | 5 folds | 2 models × 3 seeds × 5 folds = **30 runs**

| Model | Role |
|---|---|
| `resnet50_ca_lstm` | Phase 6 Stage A Baseline |
| `dual_cnn_sa_lstm` | **Novel architecture: Dual CNN + Self-Attention + BiLSTM** |

### Goal
Statistically validate the novel `dual_cnn_sa_lstm` against `resnet50_ca_lstm` across 3 seeds with label smoothing factor $0.05$.
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
