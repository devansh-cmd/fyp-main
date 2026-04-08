"""
Regenerate Stage B SOTA Push notebooks for all 4 datasets and 2 models.
Splits workloads by model + dataset to prevent the 12-hour Kaggle execution limit.
"""
import json
from pathlib import Path

# ── CHANGE THIS to match your Kaggle username ──
KAGGLE_USER = "devanshdev01"

def generate_setup_cell(dataset):
    dataset_slug_name = dataset.replace('_', '-')
    dataset_slug = f"phase6-{dataset_slug_name}-clean"
    return """import os
import glob
import shutil
import zipfile
import subprocess
import sys
from pathlib import Path

WORK_DIR = Path('/kaggle/working/PROJECT')
if WORK_DIR.exists():
    shutil.rmtree(WORK_DIR)
WORK_DIR.mkdir(parents=True, exist_ok=True)

# Find the dataset dynamically by looking for directories in /kaggle/input/
try:
    available_dirs = [d for d in os.listdir('/kaggle/input') if os.path.isdir(os.path.join('/kaggle/input', d))]
except FileNotFoundError:
    available_dirs = []

if not available_dirs:
    raise FileNotFoundError("NO DATASETS ATTACHED! Please click the + Add Data button in the right sidebar (Input tab).")

# Just grab the first dataset folder it finds (Kaggle mounts attached datasets here)
dataset_folder_name = available_dirs[0]
EXPECTED_DIR = Path('/kaggle/input') / dataset_folder_name

print(f"Dataset securely located at: {EXPECTED_DIR}")

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
    print(f'No zip found — scanning auto-extracted contents from {EXPECTED_DIR}...')
    # Kaggle sometimes nests the upload into another directory arbitrarily.
    # Find the actual 'product' directory containing our code & splits.
    product_dirs = list(EXPECTED_DIR.rglob('product'))
    
    if not product_dirs:
        raise FileNotFoundError(f"Could not find the 'product' directory inside {EXPECTED_DIR}!")
        
    actual_product_dir = product_dirs[0]
    
    # We want /kaggle/working/PROJECT/product/...
    target_product_dir = WORK_DIR / 'product'
    
    shutil.copytree(actual_product_dir, target_product_dir, dirs_exist_ok=True)
    print(f'Done copying from {actual_product_dir} to {target_product_dir}')
"""

def generate_path_fix_cell(dataset):
    if dataset == "italian_pd":
        prefix = "italian_"
    elif dataset == "esc50":
        prefix = ""
    else:
        prefix = f"{dataset}_"

    return f"""os.chdir(WORK_DIR)
print(f'CWD: {{os.getcwd()}}')

ROOT = WORK_DIR
SPLITS = ROOT / 'product/artifacts/splits'

# ── Leakage guard: assert 0 subject overlap for every fold ──
import pandas as pd
leakage_found = False

for fold in range(5):
    train_csv = SPLITS / f'train_{prefix}fold{{fold}}.csv'
    val_csv = SPLITS / f'val_{prefix}fold{{fold}}.csv'
    if not train_csv.exists() or not val_csv.exists():
        print(f'WARN: fold{{fold}} CSVs missing, skipping check')
        continue
    
    # PC-GITA has patient_id, others have subject_id (usually)
    # Check column names dynamically
    t_df = pd.read_csv(train_csv)
    v_df = pd.read_csv(val_csv)
    subj_col = 'patient_id' if 'patient_id' in t_df.columns else ('subject_id' if 'subject_id' in t_df.columns else None)
    
    if not subj_col:
        print(f'WARN: no subject tracking column found for fold{{fold}}. Cannot verify leakage.')
        continue

    t_subj = set(t_df[subj_col])
    v_subj = set(v_df[subj_col])
    overlap = t_subj & v_subj
    if overlap:
        print(f'CRITICAL: fold{{fold}} has {{len(overlap)}} overlapping subjects: {{overlap}}')
        leakage_found = True
    else:
        print(f'fold{{fold}}: OK (train={{len(t_subj)}} subjects, val={{len(v_subj)}} subjects, overlap=0)')

if leakage_found:
    raise RuntimeError('DATA LEAKAGE DETECTED — aborting. Delete and re-upload the Kaggle dataset.')
print('All folds passed leakage check.')
"""

def generate_cleanup_cell(dataset):
    return f"""import shutil
runs_root = ROOT / "product/artifacts/runs"
cleaned = 0
PHASE6_TAGS = ["_resnet50_ca_lstm_", "_dual_cnn_sa_lstm_"]
for ds in ["{dataset}"]:
    ds_dir = runs_root / ds
    if not ds_dir.exists():
        continue
    for run_dir in list(ds_dir.iterdir()):
        name = run_dir.name
        if any(tag in f"_{{name}}_" for tag in PHASE6_TAGS):
            shutil.rmtree(run_dir)
            cleaned += 1
print(f"Cleaned {{cleaned}} stale Phase 6 run directories.")
"""

SEEDS = [42, 123, 999]
DATASETS = ["italian_pd", "pcgita", "esc50", "emodb"]

EXPERIMENTS = [
    {
        "name": "resnet50_ca_lstm",
        "epochs": 15,
        "lr": 1e-4,
        "wd": 1e-2,
        "label_smoothing": 0.05,
    },
    {
        "name": "dual_cnn_sa_lstm",
        "epochs": 15,
        "lr": 1e-4,
        "wd": 1e-2,
        "label_smoothing": 0.05,
    }
]

def generate_experiment_cell(dataset, exp):
    total = len(SEEDS) * 5
    actual_epochs = 25 if dataset == "esc50" else exp['epochs']
    cell = f"""# ============================================================
# Phase 6 Stage B: {dataset}, 3 seeds, 5 folds
# Model: {exp['name']} (15 runs total)
# Label smoothing parameter: {exp['label_smoothing']}, Epochs: {actual_epochs}
# ============================================================
import sys
import subprocess
from pathlib import Path

ROOT = Path('/kaggle/working/PROJECT')
total = {total}
done = 0
print(f"Phase 6 Stage B: {{total}} runs ( {dataset} | {exp['name']} | 3 seeds × 5 folds)")

"""
    for seed in SEEDS:
        for fold in range(5):
            run_name = f"{dataset}_{exp['name']}_s{seed}_fold{fold}"
            cell += f"""
run_name = "{run_name}"
summary_path = ROOT / f"product/artifacts/runs/{dataset}/{{run_name}}/summary.json"

if summary_path.exists():
    print(f"SKIP {{run_name}} (already done)")
    done += 1
else:
    print(f"\\n>>> RUNNING {{run_name}}")
    cmd = [
        sys.executable, "-u", "product/training/train_unified.py",
        "--dataset", "{dataset}",
        "--model_type", "{exp['name']}",
        "--fold", "{fold}",
        "--epochs", "{actual_epochs}",
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
    
    # [NEW] Incremental Zip - so we don't lose data if Kaggle times out!
    import zipfile
    output_zip = f'/kaggle/working/phase6_{dataset}_{exp["name"]}_results.zip'
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        runs_dir = ROOT / 'product/artifacts/runs'
        for f in runs_dir.glob('**/*'):
            if f.is_file() and f.suffix != '.pt':
                zipf.write(f, f.relative_to(runs_dir.parent.parent))
    print(f"Incremental zip updated: {{output_zip}}")
"""
    return cell


def generate_aggregate_cell(dataset, exp):
    return f"""# Aggregate Stage B results into a summary table
import json as _json
import numpy as np
from pathlib import Path as _P

print("\\n=== Phase 6 Stage B Results ({dataset} | {exp['name']}) ===\\n")
print(f"{{'Model':<25}} {{'Fold':<6}} {{'Best F1':>8}} {{'Final F1':>9}} {{'AUC':>7}}")
print("-" * 60)

runs_root = ROOT / "product/artifacts/runs/{dataset}"
model_type = "{exp['name']}"
SEEDS = {SEEDS}

for seed in SEEDS:
    fold_f1s = []
    print(f"SEED {{seed}}:")
    for fold in range(5):
        run_name = f"{dataset}_{{model_type}}_s{{seed}}_fold{{fold}}"
        summary_p = runs_root / run_name / "summary.json"
        if summary_p.exists():
            s = _json.loads(summary_p.read_text())
            f1  = s.get("best_macro_f1", 0.0)
            ff1 = s.get("final_macro_f1", 0.0)
            auc = s.get("final_auc", 0.0)
            fold_f1s.append(f1)
            print(f"{{model_type:<25}} {{fold:<6}} {{f1:>8.4f}} {{ff1:>9.4f}} {{auc:>7.4f}}")
        else:
            print(f"{{model_type:<25}} {{fold:<6}} {{'MISSING':>8}}")
    if fold_f1s:
        print(f"  → mean ± std: {{np.mean(fold_f1s):.4f}} ± {{np.std(fold_f1s):.4f}}")
    print()
"""

def generate_zip_cell(dataset, exp):
    return f"""# Zip all results for download
output_zip = f'/kaggle/working/phase6_{dataset}_{exp['name']}_results.zip'
print(f'Zipping results...')

with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
    runs_dir = ROOT / 'product/artifacts/runs'
    for f in runs_dir.glob('**/*'):
        if f.is_file() and f.suffix != '.pt':  # skip large model weights
            zipf.write(f, f.relative_to(runs_dir.parent.parent))

print(f'Done! Download {{output_zip}} from the Output tab.')
print(f'Stage B complete: 15 runs for {dataset} / {exp['name']}.')
"""

def make_code_cell(source, cell_id):
    lines = [l + "\n" for l in source.strip().splitlines()]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
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
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": lines,
    }

def build_notebook(dataset, exp):
    title_md = f"""\
# Phase 6: SOTA Push — Stage B
**Dataset**: {dataset}
**Model**: `{exp['name']}`
**Configuration**: 3 seeds (42, 123, 999) | 5 folds | Epochs: {exp['epochs']} | Label Smoothing: {exp['label_smoothing']}

### Goal
Statistically validate the Stage B baseline and self-attention models across 4 datasets for cross-lingual validation. 
*Note: This workload has been dynamically sharded by dataset and model to avoid the 12-hour Kaggle execution limit.*
"""
    return {
        "cells": [
            make_md_cell(title_md, "title"),
            make_code_cell(generate_setup_cell(dataset), "setup"),
            make_code_cell(generate_path_fix_cell(dataset), "path_fix"),
            make_code_cell(generate_cleanup_cell(dataset), "cleanup"),
            make_code_cell(generate_experiment_cell(dataset, exp), "experiments"),
            make_code_cell(generate_aggregate_cell(dataset, exp), "aggregate"),
            make_code_cell(generate_zip_cell(dataset, exp), "zip_results"),
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

if __name__ == "__main__":
    out_dir = Path(__file__).parent.parent / "notebooks"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    total_nb = 0
    for ds in DATASETS:
        for exp in EXPERIMENTS:
            nb = build_notebook(ds, exp)
            nb_name = f"phase6_stageB_{ds}_{exp['name']}.ipynb"
            nb_path = out_dir / nb_name
            nb_path.write_text(json.dumps(nb, indent=4, ensure_ascii=False), encoding="utf-8")
            print(f"Written: {nb_path.name}  ({nb_path.stat().st_size:,} bytes)")
            total_nb += 1
            
    print(f"SUCCESS: Generated {total_nb} notebooks for Stage B!")
