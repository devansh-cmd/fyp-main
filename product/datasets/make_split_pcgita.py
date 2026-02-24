"""
PC-GITA DDK Split Generation
Creates speaker-independent Stratified 5-Fold Cross-Validation splits.
Subject IDs are extracted from filenames (e.g., AVPEPUDEAC0001 = subject).
Labels come from the hc/pd subdirectory structure.
"""
import pandas as pd
import os
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold


def _build_pcgita_dataframe(data_dir):
    """Scan pcgita_ddk directory and return a DataFrame with path, label, subject_id."""
    data_dir = Path(data_dir)
    records = []
    for label_dir in ["hc", "pd"]:
        label = label_dir.upper()  # HC or PD
        wav_dir = data_dir / label_dir
        if not wav_dir.exists():
            continue
        for wav_file in wav_dir.glob("*.wav"):
            # Subject ID is the part before the task suffix
            # e.g., AVPEPUDEAC0001_ka.wav -> AVPEPUDEAC0001
            # e.g., AVPEPUDEA0005_pataka.wav -> AVPEPUDEA0005
            parts = wav_file.stem.split("_")
            subject_id = parts[0]
            task = parts[1] if len(parts) > 1 else "unknown"
            records.append({
                "filename": wav_file.name,
                "path": str(wav_file.relative_to(data_dir.parent.parent.parent.parent)),
                "subject_id": subject_id,
                "label": label,
                "task": task,
            })
    return pd.DataFrame(records)


def make_kfold_split_pcgita(data_dir, output_dir, n_folds=5, seed=42):
    """Generate K-Fold splits using StratifiedGroupKFold (speaker-independent)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = _build_pcgita_dataframe(data_dir)
    print(f"Total DDK files: {len(df)}")

    subjects = df[["subject_id", "label"]].drop_duplicates()
    n_hc = len(subjects[subjects["label"] == "HC"])
    n_pd = len(subjects[subjects["label"] == "PD"])
    print(f"HC Subjects: {n_hc}")
    print(f"PD Subjects: {n_pd}")

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    labels = df["label"].values
    groups = df["subject_id"].values

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(df, labels, groups)):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        # Zero-leakage assertion
        train_subs = set(train_df["subject_id"])
        val_subs = set(val_df["subject_id"])
        assert len(train_subs & val_subs) == 0, f"Fold {fold}: Subject leakage detected!"

        train_df.to_csv(output_dir / f"train_pcgita_fold{fold}.csv", index=False)
        val_df.to_csv(output_dir / f"val_pcgita_fold{fold}.csv", index=False)
        print(f"  Fold {fold}: Train={len(train_df)} ({len(train_subs)} subj) | Val={len(val_df)} ({len(val_subs)} subj)")

    print(f"Saved {n_folds} fold splits to {output_dir}")


if __name__ == "__main__":
    BASE_DIR = r"c:\FYP\PROJECT"
    DATA_PATH = os.path.join(BASE_DIR, "product", "datasets", "raw", "pcgita_ddk")
    SPLIT_PATH = os.path.join(BASE_DIR, "product", "artifacts", "splits")
    make_kfold_split_pcgita(DATA_PATH, SPLIT_PATH)
