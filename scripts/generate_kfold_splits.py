"""
generate_kfold_splits.py
========================
Master orchestration script that generates Stratified Grouped K-Fold
split CSVs for ALL 5 datasets in one command.

Strategy:
  - Clinical datasets (Italian PD, Pitt, PhysioNet): Generate subject/record-level
    fold assignments, then join with existing spectrogram PNG data to produce
    training-ready CSVs pointing to .png files.
  - Acoustic datasets (ESC-50, EmoDB): Scan spectrogram directories directly
    (already generates PNG-based CSVs).

Usage:
    python scripts/generate_kfold_splits.py --n_folds 5 --seed 42
"""
import argparse
import sys
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "product" / "datasets"))

from make_split_italian import _build_italian_dataframe  # noqa: E402
from make_split_pitt import _build_pitt_dataframe  # noqa: E402
from make_split_physionet import _build_physionet_dataframe  # noqa: E402
from make_split import make_kfold_split_esc50  # noqa: E402
from make_split_emodb import make_kfold_split_emodb  # noqa: E402

from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description="Generate K-Fold splits for all datasets")
    ap.add_argument("--n_folds", type=int, default=5, help="Number of folds (default: 5)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    return ap.parse_args()


# ── Clinical Dataset Helpers ──────────────────────────────────────────────

def _remap_italian_to_png(df):
    """Remap Italian PD WAV paths to spectrogram PNG paths."""
    spec_dir = "product/audio_preprocessing/outputs/spectrograms_italian"
    df = df.copy()
    df["path"] = df["filename"].apply(
        lambda f: f"{spec_dir}/{Path(f).stem}_orig.png"
    )
    # Verify at least some PNGs exist
    sample = PROJECT_ROOT / df["path"].iloc[0]
    if not sample.exists():
        print(f"  [WARN] Sample PNG not found: {sample}")
    return df


def _remap_physionet_to_png(df):
    """Remap PhysioNet WAV paths to spectrogram PNG paths."""
    spec_dir = "product/audio_preprocessing/outputs/spectrograms_physionet"
    df = df.copy()
    df["path"] = df["filename"].apply(
        lambda f: f"{spec_dir}/{Path(f).stem}_orig.png"
    )
    sample = PROJECT_ROOT / df["path"].iloc[0]
    if not sample.exists():
        print(f"  [WARN] Sample PNG not found: {sample}")
    return df


def _remap_pitt_to_png(df, split_out_dir):
    """
    Remap Pitt WAV-level fold assignments to spectrogram segment PNGs.
    Uses the existing segments data (which maps subjects to their spectrogram files).
    """
    # Load the full segments data (all spectrograms with subject_id)
    # Try loading from existing train+val segments CSVs
    segments_files = [
        split_out_dir / "train_pitt_segments.csv",
        split_out_dir / "val_pitt_segments.csv",
    ]
    
    segments_dfs = []
    for sf in segments_files:
        if sf.exists():
            segments_dfs.append(pd.read_csv(sf))
    
    if not segments_dfs:
        # Fallback: scan the spectrograms directory directly
        spec_dir = PROJECT_ROOT / "product" / "audio_preprocessing" / "outputs" / "spectrograms_pitt"
        if not spec_dir.exists():
            print("  [ERROR] No Pitt segments data found! Cannot remap to PNGs.")
            return df
        
        records = []
        for png in spec_dir.rglob("*.png"):
            # Pattern: SUBJECT_FILEID_SEGMENT_augtype.png  e.g. 107_a8cc225a8896_000_orig.png
            parts = png.stem.split("_")
            if len(parts) >= 2:
                subject_id = parts[0]
                records.append({
                    "filepath": f"audio_preprocessing/outputs/spectrograms_pitt/{png.name}",
                    "subject_id": subject_id,
                })
        all_segments = pd.DataFrame(records)
    else:
        all_segments = pd.concat(segments_dfs, ignore_index=True)
    
    return all_segments


def generate_italian_kfold(n_folds, seed, split_out_dir):
    """Italian PD: StratifiedGroupKFold on subjects, remap to PNG."""
    italian_data = PROJECT_ROOT / "product" / "audio_preprocessing" / "data" / "Italian Parkinson's Voice and speech"
    df = _build_italian_dataframe(italian_data)
    print(f"Total WAV files found: {len(df)}")

    subjects = df[["subject_id", "label"]].drop_duplicates()
    print(f"PD Subjects: {len(subjects[subjects['label'] == 'PD'])}")
    print(f"HC Subjects: {len(subjects[subjects['label'] == 'HC'])}")

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(df, df["label"], df["subject_id"])):
        train_df = _remap_italian_to_png(df.iloc[train_idx])
        val_df = _remap_italian_to_png(df.iloc[val_idx])

        # Leakage check
        assert len(set(train_df["subject_id"]) & set(val_df["subject_id"])) == 0

        train_df.to_csv(split_out_dir / f"train_italian_fold{fold}.csv", index=False)
        val_df.to_csv(split_out_dir / f"val_italian_fold{fold}.csv", index=False)
        print(f"  Fold {fold}: Train={len(train_df)} ({train_df['subject_id'].nunique()} subj) | Val={len(val_df)} ({val_df['subject_id'].nunique()} subj)")

    print(f"Saved {n_folds} fold splits to {split_out_dir}")


def generate_pitt_kfold(n_folds, seed, split_out_dir):
    """Pitt: StratifiedGroupKFold on subjects, join with segment PNGs."""
    df_wav = _build_pitt_dataframe(PROJECT_ROOT)
    if df_wav.empty:
        print("Error: No WAV records found!")
        return

    # Get the full segments PNG data
    all_segments = _remap_pitt_to_png(df_wav, split_out_dir)

    # We need subject-level label for stratification
    if "label" not in all_segments.columns:
        # Merge label from WAV-level data
        subj_labels = df_wav[["subject_id", "label"]].drop_duplicates()
        all_segments = all_segments.merge(subj_labels, on="subject_id", how="left")

    print(f"Total spectrogram segments: {len(all_segments)}")
    unique_subjects = all_segments[["subject_id", "label"]].drop_duplicates()
    print(f"Unique subjects: {len(unique_subjects)}")

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    labels = all_segments["label"].values
    groups = all_segments["subject_id"].values

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(all_segments, labels, groups)):
        train_df = all_segments.iloc[train_idx]
        val_df = all_segments.iloc[val_idx]

        train_subs = set(train_df["subject_id"])
        val_subs = set(val_df["subject_id"])
        assert len(train_subs & val_subs) == 0, f"Fold {fold}: Subject leakage!"

        train_df.to_csv(split_out_dir / f"train_pitt_fold{fold}.csv", index=False)
        val_df.to_csv(split_out_dir / f"val_pitt_fold{fold}.csv", index=False)
        print(f"  Fold {fold}: Train={len(train_df)} ({len(train_subs)} subj) | Val={len(val_df)} ({len(val_subs)} subj)")

    print(f"Saved {n_folds} fold splits to {split_out_dir}")


def generate_physionet_kfold(n_folds, seed, split_out_dir):
    """PhysioNet: StratifiedKFold with source+label, remap to PNG."""
    master_df = _build_physionet_dataframe(PROJECT_ROOT)
    if master_df.empty:
        print("Error: No records found!")
        return

    print(f"Total records indexed: {len(master_df)}")

    master_df["stratify_col"] = master_df["source"] + "_" + master_df["label"].astype(str)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for fold, (train_idx, val_idx) in enumerate(skf.split(master_df, master_df["stratify_col"])):
        train_df = _remap_physionet_to_png(master_df.iloc[train_idx].drop(columns=["stratify_col"]))
        val_df = _remap_physionet_to_png(master_df.iloc[val_idx].drop(columns=["stratify_col"]))

        train_df.to_csv(split_out_dir / f"train_physionet_fold{fold}.csv", index=False)
        val_df.to_csv(split_out_dir / f"val_physionet_fold{fold}.csv", index=False)
        print(f"  Fold {fold}: Train={len(train_df)} | Val={len(val_df)}")

    print(f"Saved {n_folds} fold splits to {split_out_dir}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    n_folds = args.n_folds
    seed = args.seed
    split_out_dir = PROJECT_ROOT / "product" / "artifacts" / "splits"
    split_out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Generating {n_folds}-Fold Splits for ALL Datasets")
    print(f"  Seed: {seed} | Output: {split_out_dir}")
    print(f"  All CSVs point to .png spectrogram files")
    print(f"{'='*60}\n")

    # 1. Italian PD
    print(f"\n--- [1/5] Italian PD ---")
    generate_italian_kfold(n_folds, seed, split_out_dir)

    # 2. Pitt Corpus
    print(f"\n--- [2/5] Pitt Corpus ---")
    generate_pitt_kfold(n_folds, seed, split_out_dir)

    # 3. PhysioNet
    print(f"\n--- [3/5] PhysioNet ---")
    generate_physionet_kfold(n_folds, seed, split_out_dir)

    # 4. ESC-50 (already scans PNGs directly)
    print(f"\n--- [4/5] ESC-50 ---")
    esc50_spec_dir = PROJECT_ROOT / "product" / "audio_preprocessing" / "outputs" / "spectrograms"
    esc50_csv = PROJECT_ROOT / "product" / "audio_preprocessing" / "data" / "ESC-50" / "meta" / "esc50.csv"
    make_kfold_split_esc50(esc50_spec_dir, esc50_csv, split_out_dir, n_folds=n_folds, seed=seed)

    # 5. EmoDB (already scans PNGs directly)
    print(f"\n--- [5/5] EmoDB ---")
    emodb_spec_dir = PROJECT_ROOT / "product" / "audio_preprocessing" / "outputs" / "spectrograms_emodb"
    make_kfold_split_emodb(emodb_spec_dir, split_out_dir, n_folds=n_folds, seed=seed)

    # Summary
    print(f"\n{'='*60}")
    print(f"  COMPLETE: Generated {n_folds * 5 * 2} CSV files")
    print(f"  ({n_folds} folds x 5 datasets x 2 files)")
    print(f"  All CSVs contain .png spectrogram paths")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
