"""
Pack the Phase 6 Kaggle upload zip (Stage A).
Includes:
  - All source code (models, training, datasets)
  - Split CSVs for Italian PD only (per Stage A strategy)
  - Spectrograms for Italian PD only
  - The Phase 6 notebook
"""
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(r'c:\FYP\PROJECT')
OUTPUT_ZIP = PROJECT_ROOT / 'phase6_upload.zip'

INCLUDE_DIRS = [
    'product/models',
    'product/training',
    'product/datasets',
    'product/artifacts/splits',
    'product/audio_preprocessing/outputs/spectrograms_italian',
    'product/notebooks',
]

EXCLUDE_EXTS = {'.zip', '.pt', '.pyc'}
EXCLUDE_DIRS = {'__pycache__', '.git', 'tmp_kaggle_results', 'logs'}

# Stage A: Only pack Italian PD splits to save upload time and Kaggle space
SPLIT_PREFIXES = ('train_italian', 'val_italian')

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
                
            # Filter splits
            if 'splits' in str(fpath):
                if not any(fpath.name.startswith(p) for p in SPLIT_PREFIXES):
                    continue
                    
            # Only include phase 6 notebook to avoid confusion
            if 'notebooks' in str(fpath):
                if fpath.name != 'phase6_sota_push_kaggle.ipynb':
                    continue
                    
            arc_name = fpath.relative_to(PROJECT_ROOT)
            zf.write(fpath, arc_name)
            count += 1
            if count % 100 == 0:
                print(f'  Added {count} files...')

size_mb = OUTPUT_ZIP.stat().st_size / 1024 / 1024
print(f'\nDone! {count} files -> {OUTPUT_ZIP.name} ({size_mb:.1f} MB)')
