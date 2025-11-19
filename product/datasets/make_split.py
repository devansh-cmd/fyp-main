# product/datasets/make_split.py
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import argparse
import re

AUG_SUFFIX = re.compile(
    r"_(orig|noisy|pitchUp\d+|stretch\d+(?:\.\d+)?)$", re.IGNORECASE
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--spec_dir", required=True, help="Root dir containing PNG spectrograms"
    )
    ap.add_argument(
        "--esc50_csv",
        required=True,
        help="ESC-50 metadata CSV (has filename, category)",
    )
    ap.add_argument("--out_dir", required=True, help="Where to write train/val CSVs")
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--outfile_prefix", default="", help="Optional prefix like 'clean_'"
    )
    return ap.parse_args()


def clip_base_from_png_path(png_path: Path) -> str:
    """
    Turn '.../1-100032-A-0_pitchUp2.png' -> '1-100032-A-0'
    """
    stem = png_path.stem
    stem = AUG_SUFFIX.sub("", stem)
    # also handle any accidental double underscores etc.
    return stem.split("_")[0]


def main():
    args = parse_args()
    spec_dir = Path(args.spec_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load ESC-50 label mapping (filename -> category)
    esc50 = pd.read_csv(args.esc50_csv)
    esc50["base"] = esc50["filename"].str.replace(".wav", "", regex=False)
    base_to_label = dict(zip(esc50["base"], esc50["category"]))

    # --- Scan PNGs; collect (filepath, base, label)
    files, bases, labels = [], [], []
    for p in spec_dir.rglob("*.png"):
        base = clip_base_from_png_path(p)
        label = base_to_label.get(base)
        if label is None:
            continue
        files.append(str(p.resolve()))
        bases.append(base)
        labels.append(label)

    if not files:
        raise SystemExit(f"No PNGs found under {spec_dir}. Check path/names.")

    df = pd.DataFrame(
        {"filepath": files, "clip_id": bases, "label": labels}
    ).drop_duplicates()
    # ---- Build a clip-level table for stratification
    clip_tbl = (
        df.groupby("clip_id")
        .agg(label=("label", "first"), n_png=("filepath", "count"))
        .reset_index()
    )
    # Sanity: each clip_id should have a single label
    assert (
        clip_tbl.groupby("clip_id")["label"].nunique().max() == 1
    ), "Mixed labels within a clip_id."

    # ---- Stratified split at CLIP level
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=args.val_ratio, random_state=args.seed
    )
    (train_idx, val_idx) = next(sss.split(clip_tbl["clip_id"], clip_tbl["label"]))
    train_clips = set(clip_tbl.iloc[train_idx]["clip_id"].tolist())
    val_clips = set(clip_tbl.iloc[val_idx]["clip_id"].tolist())

    # ---- Expand back to rows (all PNGs from the chosen clips)
    train_df = df[df["clip_id"].isin(train_clips)].copy()
    val_df = df[df["clip_id"].isin(val_clips)].copy()

    # ---- Final leakage check at clip level
    overlap = set(train_df["clip_id"]) & set(val_df["clip_id"])
    assert (
        len(overlap) == 0
    ), f"Leak persists for clip_ids: {list(sorted(overlap))[:10]}"

    # ---- Select output columns and write
    keep = ["filepath", "label"]
    prefix = args.outfile_prefix
    (out_dir / f"{prefix}train.csv").write_text(
        train_df[keep].sort_values("filepath").to_csv(index=False)
    )
    (out_dir / f"{prefix}val.csv").write_text(
        val_df[keep].sort_values("filepath").to_csv(index=False)
    )

    # ---- Report
    print(f"Total PNGs: {len(df)} | Clips: {len(clip_tbl)}")
    print(
        f"Train PNGs: {len(train_df)} | Clips: {len(train_clips)} | Classes: {train_df['label'].nunique()}"
    )
    print(
        f"Val   PNGs: {len(val_df)} | Clips: {len(val_clips)} | Classes: {val_df['label'].nunique()}"
    )
    print("Wrote:", out_dir / f"{prefix}train.csv", "and", out_dir / f"{prefix}val.csv")


if __name__ == "__main__":
    main()
