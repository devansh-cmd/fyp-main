import pandas as pd
import numpy as np
import os
from pathlib import Path
import random
import re
from sklearn.model_selection import StratifiedGroupKFold


def _build_italian_dataframe(data_dir):
    """Shared helper: scan the Italian PD directory and return a DataFrame."""
    data_dir = Path(data_dir)
    records = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith('.wav'):
                file_path = Path(root) / file
                label = None
                if "28 People with Parkinson's disease" in str(file_path):
                    label = "PD"
                elif "Healthy Control" in str(file_path):
                    label = "HC"
                if not label:
                    continue
                subject_id = file_path.parent.name
                match = re.match(r'^([a-zA-Z]+[0-9]?)', file)
                task = match.group(1) if match else "unknown"
                records.append({
                    "filename": file,
                    "path": str(file_path.relative_to(data_dir.parent.parent.parent)),
                    "subject_id": subject_id,
                    "label": label,
                    "task": task
                })
    return pd.DataFrame(records)


def make_kfold_split_italian(data_dir, output_dir, n_folds=5, seed=42):
    """Generate K-Fold splits using StratifiedGroupKFold (subject-independent)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = _build_italian_dataframe(data_dir)
    print(f"Total files found: {len(df)}")

    subjects = df[["subject_id", "label"]].drop_duplicates()
    print(f"PD Subjects: {len(subjects[subjects['label'] == 'PD'])}")
    print(f"HC Subjects: {len(subjects[subjects['label'] == 'HC'])}")

    # Use StratifiedGroupKFold: stratify by label, group by subject_id
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    labels = df["label"].values
    groups = df["subject_id"].values

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(df, labels, groups)):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        # Leakage verification
        train_subs = set(train_df["subject_id"])
        val_subs = set(val_df["subject_id"])
        assert len(train_subs & val_subs) == 0, f"Fold {fold}: Subject leakage detected!"

        train_df.to_csv(output_dir / f"train_italian_fold{fold}.csv", index=False)
        val_df.to_csv(output_dir / f"val_italian_fold{fold}.csv", index=False)
        print(f"  Fold {fold}: Train={len(train_df)} ({len(train_subs)} subjects) | Val={len(val_df)} ({len(val_subs)} subjects)")

    print(f"Saved {n_folds} fold splits to {output_dir}")


def make_split_italian(data_dir, output_dir, val_ratio=0.2, seed=42):
    random.seed(seed)
    
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    records = []
    
    # Dataset structure:
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith('.wav'):
                file_path = Path(root) / file
                
                # Determine label
                # Correct logic: check for cohort specific folder names
                if "28 People with Parkinson's disease" in str(file_path):
                    label = "PD"
                elif "Healthy Control" in str(file_path):
                    label = "HC"
                
                if not label:
                    continue
                
                # Determine subject_id (folder name)
                subject_id = file_path.parent.name
                
                # Determine task (prefix)
                # Prefixes: B1, B2, VA1, VA2, VE1, VE2, VI1, VI2, VO1, VO2, VU1, VU2, PR1, D1, D2, FB1
                match = re.match(r'^([a-zA-Z]+[0-9]?)', file)
                task = match.group(1) if match else "unknown"
                
                records.append({
                    "filename": file,
                    "path": str(file_path.relative_to(data_dir.parent.parent.parent)), # Relative to product/
                    "subject_id": subject_id,
                    "label": label,
                    "task": task
                })
    
    df = pd.DataFrame(records)
    print(f"Total files found: {len(df)}")
    
    # Group subjects for independent split
    subjects = df[["subject_id", "label"]].drop_duplicates()
    
    # Debug: Check for subjects appearing in multiple folders (sessions)
    subject_counts = df.groupby("subject_id")["path"].apply(lambda x: len(set([Path(p).parent.parent.name for p in x])))
    duplicates = subject_counts[subject_counts > 1]
    if not duplicates.empty:
        print("\nSubjects detected in multiple session folders (will be grouped together):")
        for sub, count in duplicates.items():
            print(f" - {sub}: in {count} folders")
    
    pd_subjects = subjects[subjects["label"] == "PD"]["subject_id"].tolist()
    hc_subjects = subjects[subjects["label"] == "HC"]["subject_id"].tolist()
    
    print(f"PD Subjects: {len(pd_subjects)}")
    print(f"HC Subjects: {len(hc_subjects)}")
    
    random.shuffle(pd_subjects)
    random.shuffle(hc_subjects)
    
    n_pd_val = int(len(pd_subjects) * val_ratio)
    n_hc_val = int(len(hc_subjects) * val_ratio)
    
    val_subjects = pd_subjects[:n_pd_val] + hc_subjects[:n_hc_val]
    train_subjects = pd_subjects[n_pd_val:] + hc_subjects[n_hc_val:]
    
    train_df = df[df["subject_id"].isin(train_subjects)]
    val_df = df[df["subject_id"].isin(val_subjects)]
    
    print(f"Train files: {len(train_df)} (Subjects: {len(train_subjects)})")
    print(f"Val files: {len(val_df)} (Subjects: {len(val_subjects)})")
    
    train_df.to_csv(output_dir / "train_italian.csv", index=False)
    val_df.to_csv(output_dir / "val_italian.csv", index=False)
    print(f"Saved splits to {output_dir}")

if __name__ == "__main__":
    BASE_DIR = r"c:\FYP\PROJECT"
    DATA_PATH = os.path.join(BASE_DIR, "product", "audio_preprocessing", "data", "Italian Parkinson's Voice and speech")
    SPLIT_PATH = os.path.join(BASE_DIR, "product", "artifacts", "splits")
    
    make_split_italian(DATA_PATH, SPLIT_PATH)
