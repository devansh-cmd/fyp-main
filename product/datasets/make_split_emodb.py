# product/datasets/make_split_emodb.py
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import argparse
import re

# EmoDB emotion mapping
EMOTION_MAP = {
    "W": "anger",
    "L": "boredom",
    "E": "disgust",
    "A": "fear",
    "F": "happiness",
    "T": "sadness",
    "N": "neutral",
}

# Augmentation suffix pattern
AUG_SUFFIX = re.compile(
    r"_(orig|noise|stretch|pitch|gain|reverb)$", re.IGNORECASE
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--spec_dir", required=True, help="Root dir containing EmoDB PNG spectrograms"
    )
    ap.add_argument("--out_dir", required=True, help="Where to write train/val CSVs")
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def clip_base_from_png_path(png_path: Path) -> str:
    """
    Turn '.../03a01Fa_noise.png' -> '03a01Fa'
    """
    stem = png_path.stem
    stem = AUG_SUFFIX.sub("", stem)  # remove augmentation suffix
    return stem


def parse_emotion_from_filename(base: str) -> str:
    """
    Extract emotion from EmoDB filename.
    Example: '03a01Fa' -> position 5 is 'F' -> 'happiness'
    """
    if len(base) < 6:
        return "unknown"
    emotion_code = base[5]  # 6th character (0-indexed position 5)
    return EMOTION_MAP.get(emotion_code, "unknown")


def main():
    args = parse_args()
    spec_dir = Path(args.spec_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Scan PNGs; collect (filepath, base, label)
    files, bases, labels = [], [], []
    for p in spec_dir.rglob("*.png"):
        base = clip_base_from_png_path(p)
        label = parse_emotion_from_filename(base)
        if label == "unknown":
            print(f"[WARN] Skipping {p.name} - couldn't parse emotion")
            continue
        files.append(str(p.resolve()))
        bases.append(base)
        labels.append(label)

    print(f"Found {len(files)} PNG files")
    
    # --- Build DataFrame
    df = pd.DataFrame({"filepath": files, "clip_id": bases, "label": labels})
    
    # --- Get unique clips (for stratified splitting at clip level)
    unique_clips = df[["clip_id", "label"]].drop_duplicates()
    print(f"Unique clips: {len(unique_clips)}")
    print(f"Emotion distribution:\n{unique_clips['label'].value_counts()}")

    # --- Stratified split at clip level
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=args.val_ratio, random_state=args.seed
    )
    train_idx, val_idx = next(
        splitter.split(unique_clips["clip_id"], unique_clips["label"])
    )

    train_clips = set(unique_clips.iloc[train_idx]["clip_id"])
    val_clips = set(unique_clips.iloc[val_idx]["clip_id"])

    # --- Assign all augmented variants to same split
    df_train = df[df["clip_id"].isin(train_clips)]
    df_val = df[df["clip_id"].isin(val_clips)]

    print(f"\nTrain: {len(df_train)} images ({len(train_clips)} clips)")
    print(f"Val:   {len(df_val)} images ({len(val_clips)} clips)")

    # --- Save CSVs
    train_csv = out_dir / "train_emodb.csv"
    val_csv = out_dir / "val_emodb.csv"

    df_train[["filepath", "label"]].to_csv(train_csv, index=False)
    df_val[["filepath", "label"]].to_csv(val_csv, index=False)

    print("\nSaved:")
    print(f"  {train_csv}")
    print(f"  {val_csv}")
    print("\nDone!")


if __name__ == "__main__":
    main()
