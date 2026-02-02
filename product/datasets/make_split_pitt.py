import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import re
import hashlib
import json


def extract_subject_id(filename):
    """
    Extracts the 3-digit subject ID from names like '002-0c-n.wav'
    """
    match = re.search(r"^(\d{3})", filename)
    return match.group(1) if match else None


def generate_file_id(filepath: Path, project_root: Path):
    """
    Generates a stable, tamper-proof ID for a file based on its relative path.
    """
    rel_path = str(filepath.relative_to(project_root)).replace("\\", "/")
    return hashlib.sha256(rel_path.encode()).hexdigest()[:12]


def get_file_hash(filepath: Path):
    """Compute SHA-256 of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def make_split_pitt(project_root: Path, val_ratio: float = 0.2, seed: int = 42):
    """
    Creates stratified, subject-independent splits for the Pitt Corpus.
    Follows Rule 1 (Diagnosis-only) and Rule 2 (Subject-level integrity).
    Includes stable IDs and Immutability Manifest.
    """
    base_dir = (
        project_root
        / "product"
        / "audio_preprocessing"
        / "data"
        / "English Pitt Corpus"
    )
    split_out_dir = project_root / "product" / "artifacts" / "splits"
    split_out_dir.mkdir(parents=True, exist_ok=True)

    # Categories based on Rule 3: Folder determines label
    categories = {"cookie_control": 0, "dementia_control": 1}

    all_records = []

    for folder_name, label in categories.items():
        folder_path = base_dir / folder_name
        if not folder_path.exists():
            print(f"Warning: Folder {folder_name} not found at {folder_path}")
            continue

        for ext in ["*.wav", "*.mp3"]:
            for wav_path in folder_path.glob(ext):
                subject_id = extract_subject_id(wav_path.name)
                if not subject_id:
                    print(f"Warning: Could not extract subject_id from {wav_path.name}")
                    continue

                # Rule 4: One row per audio file
                # Stable record identifier
                file_id = generate_file_id(wav_path, project_root)

                all_records.append(
                    {
                        "file_id": file_id,
                        "filepath": str(wav_path.resolve()),
                        "filename": wav_path.name,
                        "subject_id": subject_id,
                        "label": label,
                    }
                )

    df = pd.DataFrame(all_records)
    if df.empty:
        print("Error: No records found!")
        return

    print(f"Total files found: {len(df)}")

    # Rule 2: Subject-level integrity
    unique_subjects = df.groupby("subject_id")["label"].max().reset_index()

    print(f"Diagnosis distribution at subject level (N={len(unique_subjects)}):")
    print(unique_subjects["label"].value_counts(normalize=True))

    # Stratified split at subject level
    train_subs, val_subs = train_test_split(
        unique_subjects["subject_id"],
        test_size=val_ratio,
        random_state=seed,
        stratify=unique_subjects["label"],
    )

    train_subs = set(train_subs)
    val_subs = set(val_subs)

    # Assign all variants of a subject to the same split
    df_train = df[df["subject_id"].isin(train_subs)]
    df_val = df[df["subject_id"].isin(val_subs)]

    # Save CSVs
    train_csv = split_out_dir / "train_pitt.csv"
    val_csv = split_out_dir / "val_pitt.csv"

    df_train.to_csv(train_csv, index=False)
    df_val.to_csv(val_csv, index=False)

    # --- Generate Splits Manifest (Immutability Checksum) ---
    manifest = {
        "dataset": "pitt_corpus",
        "seed": seed,
        "files": {
            "train_pitt.csv": get_file_hash(train_csv),
            "val_pitt.csv": get_file_hash(val_csv),
        },
    }

    manifest_path = split_out_dir / "splits_manifest_pitt.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)

    print(f"\n[MUTABLE] Splits saved to {split_out_dir}")
    print(f"[IMMUTABLE] Manifest saved to {manifest_path}")
    print(f"Train: {len(df_train)} files ({len(train_subs)} subjects)")
    print(f"Val:   {len(df_val)} files ({len(val_subs)} subjects)")

    # Rule 2 Verification: Check for leakage
    leakage = set(df_train["subject_id"]).intersection(set(df_val["subject_id"]))
    if leakage:
        print(f"CRITICAL ERROR: Leakage detected for subjects: {leakage}")
    else:
        print("Verification: Zero subject-level leakage detected.")


if __name__ == "__main__":
    PROJ_ROOT = Path(__file__).resolve().parent.parent.parent
    make_split_pitt(PROJ_ROOT)
