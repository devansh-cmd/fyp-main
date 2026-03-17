 """
Pack the Phase 5 Kaggle upload zip.
Includes:
  - All source code (models, training, datasets)
  - Split CSVs for Pitt, Italian PD, PhysioNet, PC-GITA
  - Spectrograms for all 4 datasets
  - The Phase 5 notebook

Does NOT include:
  - Raw WAV files or raw datasets
  - Existing run results (not needed on Kaggle)
  - Large zip files
"""
import zipfile
import os
from pathlib import Path

PROJECT_ROOT = Path(r'c:\FYP\PROJECT')
OUTPUT_ZIP = PROJECT_ROOT / 'phase5_upload.zip'

INCLUDE_DIRS = [
    # Source code
    'product/models',
    'product/training',
    'product/datasets',
    # Split CSVs for the 4 target datasets
    'product/artifacts/splits',
    # Spectrograms for the 4 target datasets
    'product/audio_preprocessing/outputs/spectrograms_pitt',
    'product/audio_preprocessing/outputs/spectrograms_italian',
    'product/audio_preprocessing/outputs/spectrograms_physionet',
    'product/audio_preprocessing/outputs/spectrograms_pcgita',
    # Notebook
    'product/notebooks',
]

EXCLUDE_EXTS = {'.zip', '.pt', '.pyc'}
EXCLUDE_DIRS = {'__pycache__', '.git', 'tmp_kaggle_results'}

# Only include these split CSVs (the 4 target datasets × 5 folds × train/val)
SPLIT_PREFIXES = ('train_pitt_fold', 'val_pitt_fold',
                  'train_italian_fold', 'val_italian_fold',
                  'train_physionet_fold', 'val_physionet_fold',
                  'train_pcgita_fold', 'val_pcgita_fold')

count = 0
with zipfile.ZipFile(OUTPUT_ZIP, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    for rel_dir in INCLUDE_DIRS:
        abs_dir = PROJECT_ROOT / rel_dir
        if not abs_dir.exists():
            print(f'  SKIP (not found): {rel_dir}')
            continue
        for fpath in abs_dir.rglob('*'):
            if fpath.is_dir():
                continue
            if any(ex in fpath.parts for ex in EXCLUDE_DIRS):
                continue
            if fpath.suffix in EXCLUDE_EXTS:
                continue
            # Filter split CSVs to only the target datasets
            if 'splits' in str(fpath):
                if not any(fpath.name.startswith(p) for p in SPLIT_PREFIXES):
                    continue
            arc_name = fpath.relative_to(PROJECT_ROOT)
            zf.write(fpath, arc_name)
            count += 1
            if count % 100 == 0:
                print(f'  Added {count} files...')

size_mb = OUTPUT_ZIP.stat().st_size / 1024 / 1024
print(f'\nDone! {count} files -> {OUTPUT_ZIP.name} ({size_mb:.1f} MB)')
