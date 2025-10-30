from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import argparse

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec_dir", required=True)
    ap.add_argument("--esc50_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    return ap.parse_args()

def main():
    args = parse_args()
    spec_dir = Path(args.spec_dir)
    esc50 = pd.read_csv(args.esc50_csv)

    # map base wav id (e.g., '1-100032-A-0') -> category
    esc50["base"] = esc50["filename"].str.replace(".wav", "", regex=False)
    base_to_label = dict(zip(esc50["base"], esc50["category"]))

    files, labels = [], []
    for p in spec_dir.rglob("*.png"):
        base = p.stem.split("_")[0]          # '1-100032-A-0_orig' -> '1-100032-A-0'
        label = base_to_label.get(base)
        if label is None:
            continue
        files.append(str(p.resolve()))
        labels.append(label)

    if not files:
        raise SystemExit(f"No PNGs found under {spec_dir}. Check the path and filenames.")

    df = pd.DataFrame({"filepath": files, "label": labels}).sample(frac=1.0, random_state=42)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.val_ratio, random_state=42)
    train_idx, val_idx = next(sss.split(df["filepath"], df["label"]))
    train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out / "train.csv", index=False)
    val_df.to_csv(out / "val.csv", index=False)

    print(f"Total: {len(df)} | Train: {len(train_df)} | Val: {len(val_df)}")
    print(f"Wrote: {out/'train.csv'} and {out/'val.csv'}")

if __name__ == "__main__":
    main()
