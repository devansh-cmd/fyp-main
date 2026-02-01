
import pandas as pd
import sys
import re
from pathlib import Path

def clip_id(p):
    # Extracts the base ID from paths like '.../1-100038-A-14_orig.png' -> '1-100038-A-14'
    name = Path(p).stem
    # Handle both Windows and Linux separators if present in string
    name = re.sub(r'_(orig|noisy|pitchUp\d+|stretch\d+\.\d+)$', '', name)
    return name

def check_leakage(train_path, val_path):
    if not Path(train_path).exists() or not Path(val_path).exists():
        print(f"Skipping {train_path} vs {val_path} (Files not found)")
        return 0
    
    t = pd.read_csv(train_path)
    v = pd.read_csv(val_path)
    
    ti = set(t['filepath'].apply(clip_id))
    vi = set(v['filepath'].apply(clip_id))
    
    overlap = ti & vi
    print(f"Leakage Check [{Path(train_path).name} vs {Path(val_path).name}]: {len(overlap)} overlaps found.")
    if overlap:
        print(f"Overlap IDs: {list(overlap)[:5]}...")
    return len(overlap)

if __name__ == "__main__":
    total_errors = 0
    # Check ESC-50
    total_errors += check_leakage('product/artifacts/splits/train_no_aug.csv', 'product/artifacts/splits/val_no_aug.csv')
    # Check EmoDB
    total_errors += check_leakage('product/artifacts/splits/train_emodb_no_aug.csv', 'product/artifacts/splits/val_emodb_no_aug.csv')
    
    if total_errors > 0:
        print("CRITICAL: Data leakage detected between training and validation sets!")
        sys.exit(1)
    else:
        print("SUCCESS: No data leakage detected.")
        sys.exit(0)
