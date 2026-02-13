import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold


def _build_physionet_dataframe(project_root: Path):
    """Shared helper: index PhysioNet 2016 records and return a DataFrame."""
    base_data_dir = (
        project_root / "product" / "audio_preprocessing" / "data" / "physionet_2016"
    )
    annotations_dir = base_data_dir / "annotations" / "updated"
    sources = ["a", "b", "c", "d", "e", "f"]
    all_records = []

    for src in sources:
        src_folder_name = f"training-{src}"
        wav_dir = base_data_dir / src_folder_name
        ref_csv = annotations_dir / src_folder_name / "REFERENCE_withSQI.csv"

        if not ref_csv.exists():
            continue

        df_ref = pd.read_csv(ref_csv, header=None, names=["record_id", "label", "sqi"])

        for _, row in df_ref.iterrows():
            record_id = row["record_id"]
            label_raw = row["label"]
            sqi = row["sqi"]
            label = 1 if label_raw == 1 else 0
            wav_path = wav_dir / f"{record_id}.wav"
            if not wav_path.exists():
                continue
            all_records.append({
                "filename": f"{record_id}.wav",
                "path": str(wav_path.relative_to(project_root)),
                "label": label,
                "sqi": sqi,
                "source": src,
            })
    return pd.DataFrame(all_records)


def make_kfold_split_physionet(project_root: Path, n_folds: int = 5, seed: int = 42):
    """Generate K-Fold splits using StratifiedKFold with source+label stratification."""
    project_root = Path(project_root)
    split_out_dir = project_root / "product" / "artifacts" / "splits"
    split_out_dir.mkdir(parents=True, exist_ok=True)

    master_df = _build_physionet_dataframe(project_root)
    if master_df.empty:
        print("Error: No records found!")
        return

    print(f"Total records indexed: {len(master_df)}")

    # Stratify by source+label composite
    master_df["stratify_col"] = master_df["source"] + "_" + master_df["label"].astype(str)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for fold, (train_idx, val_idx) in enumerate(skf.split(master_df, master_df["stratify_col"])):
        train_df = master_df.iloc[train_idx].drop(columns=["stratify_col"])
        val_df = master_df.iloc[val_idx].drop(columns=["stratify_col"])

        train_df.to_csv(split_out_dir / f"train_physionet_fold{fold}.csv", index=False)
        val_df.to_csv(split_out_dir / f"val_physionet_fold{fold}.csv", index=False)
        print(f"  Fold {fold}: Train={len(train_df)} | Val={len(val_df)}")

    print(f"Saved {n_folds} fold splits to {split_out_dir}")



def make_split_physionet(project_root: Path, val_ratio: float = 0.2, seed: int = 42):
    """
    Creates a master index and stratified splits for the PhysioNet 2016 dataset.
    """
    base_data_dir = (
        project_root / "product" / "audio_preprocessing" / "data" / "physionet_2016"
    )
    annotations_dir = base_data_dir / "annotations" / "updated"
    split_out_dir = project_root / "product" / "artifacts" / "splits"
    split_out_dir.mkdir(parents=True, exist_ok=True)

    sources = ["a", "b", "c", "d", "e", "f"]
    all_records = []

    print("Indexing PhysioNet 2016 sources...")

    for src in sources:
        src_folder_name = f"training-{src}"
        wav_dir = base_data_dir / src_folder_name
        ref_csv = annotations_dir / src_folder_name / "REFERENCE_withSQI.csv"

        if not ref_csv.exists():
            print(f"Warning: Reference CSV not found for {src_folder_name}: {ref_csv}")
            continue

        # Load reference metadata
        # Format: filename (without extension), label (1/-1), sqi (1/0)
        df_ref = pd.read_csv(ref_csv, header=None, names=["record_id", "label", "sqi"])

        for _, row in df_ref.iterrows():
            record_id = row["record_id"]
            label_raw = row["label"]
            sqi = row["sqi"]

            # Map labels: 1 (Abnormal) -> 1, -1 (Normal) -> 0
            label = 1 if label_raw == 1 else 0

            wav_path = wav_dir / f"{record_id}.wav"

            if not wav_path.exists():
                # Some files might be missing or have different extensions in raw folders
                # Checking if it exists
                continue

            all_records.append(
                {
                    "filename": f"{record_id}.wav",
                    "path": str(wav_path.relative_to(project_root)),
                    "label": label,
                    "sqi": sqi,
                    "source": src,
                }
            )

    master_df = pd.DataFrame(all_records)

    if master_df.empty:
        print("Error: No records found!")
        return

    print(f"Total records indexed: {len(master_df)}")
    print(master_df["label"].value_counts(normalize=True))

    # Save Master Index
    master_df.to_csv(split_out_dir / "master_index_physionet.csv", index=False)
    print(f"Master index saved to: {split_out_dir / 'master_index_physionet.csv'}")

    # Stratified Split
    # We stratify by source AND label to ensure consistent domain distribution
    master_df["stratify_col"] = (
        master_df["source"] + "_" + master_df["label"].astype(str)
    )

    train_df, val_df = train_test_split(
        master_df,
        test_size=val_ratio,
        random_state=seed,
        stratify=master_df["stratify_col"],
    )

    # Clean up stratify column
    train_df = train_df.drop(columns=["stratify_col"])
    val_df = val_df.drop(columns=["stratify_col"])

    # Save Splits
    train_df.to_csv(split_out_dir / "train_physionet.csv", index=False)
    val_df.to_csv(split_out_dir / "val_physionet.csv", index=False)

    print(f"Splits saved to {split_out_dir}")
    print(f"Train size: {len(train_df)} | Val size: {len(val_df)}")
    print(
        f"Train label distribution:\n{train_df['label'].value_counts(normalize=True)}"
    )
    print(f"Val label distribution:\n{val_df['label'].value_counts(normalize=True)}")


if __name__ == "__main__":
    PROJ_ROOT = Path(__file__).resolve().parent.parent.parent
    make_split_physionet(PROJ_ROOT)
