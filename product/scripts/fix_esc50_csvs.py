import os
import pandas as pd
import glob

splits_dir = r"c:\FYP\PROJECT\product\artifacts\splits"
csv_files = glob.glob(os.path.join(splits_dir, "*fold*.csv"))
esc50_csvs = [f for f in csv_files if "train_fold" in f or "val_fold" in f]

for fpath in esc50_csvs:
    df = pd.read_csv(fpath)
    if 'filepath' in df.columns:
        # replace C:\FYP\PROJECT\ with empty string to make relative
        df['filepath'] = df['filepath'].str.replace(r"C:\\FYP\\PROJECT\\", "", regex=False)
        # normalize slashes just in case
        df['filepath'] = df['filepath'].str.replace("\\", "/")
        df.to_csv(fpath, index=False)
        print(f"Fixed paths in {os.path.basename(fpath)}")
print("Done fixing ESC-50 CSVs.")
