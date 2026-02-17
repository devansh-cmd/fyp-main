import pandas as pd
import sys
from pathlib import Path

def check_leakage(train_path, val_path, dataset_name, subject_col=None):
    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    
    # Path overlap check (fundamental for all)
    train_paths = set(train['path'] if 'path' in train.columns else train['filepath'])
    val_paths = set(val['path'] if 'path' in val.columns else val['filepath'])
    
    path_overlap = train_paths.intersection(val_paths)
    
    print(f"--- Leakage Audit: {dataset_name} ---")
    print(f"Train Size: {len(train)} | Val Size: {len(val)}")
    print(f"Path Overlap: {len(path_overlap)}")
    
    if len(path_overlap) > 0:
        print(f"[CRITICAL] Found {len(path_overlap)} overlapping file paths!")
        
    # Subject overlap check (critical for medical/pathology)
    if subject_col and subject_col in train.columns and subject_col in val.columns:
        train_subjects = set(train[subject_col])
        val_subjects = set(val[subject_col])
        subject_overlap = train_subjects.intersection(val_subjects)
        print(f"Unique Subjects (Train): {len(train_subjects)} | Unique Subjects (Val): {len(val_subjects)}")
        print(f"Subject Overlap: {len(subject_overlap)}")
        if len(subject_overlap) > 0:
            print(f"[CRITICAL] Found {len(subject_overlap)} overlapping subjects!")
            print(f"Overlapping Subjects: {list(subject_overlap)[:10]}")
    else:
        if subject_col:
            print(f"[WARNING] Subject column '{subject_col}' not found in {dataset_name} CSVs.")

    print("")

def main():
    splits_dir = Path("product/artifacts/splits")
    
    # Audit Matrix
    audits = [
        ("pitt_segments", "train_pitt_segments.csv", "val_pitt_segments.csv", "subject_id"),
        ("italian_pd", "train_italian_png.csv", "val_italian_png.csv", "subject_id"),
        ("esc50", "train.csv", "val.csv", None),
        ("emodb", "train_emodb.csv", "val_emodb.csv", None),
        ("physionet", "train_physionet_png.csv", "val_physionet_png.csv", None),
    ]
    
    for name, train_f, val_f, sub_col in audits:
        check_leakage(splits_dir / train_f, splits_dir / val_f, name, sub_col)

if __name__ == "__main__":
    main()
