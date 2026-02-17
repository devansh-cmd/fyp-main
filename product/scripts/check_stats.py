import pandas as pd
from pathlib import Path

def get_stats(csv_path):
    if not Path(csv_path).exists():
        return None
    df = pd.read_csv(csv_path)
    stats = {
        "rows": len(df),
    }
    if "subject_id" in df.columns:
        stats["subjects"] = df["subject_id"].nunique()
    elif "clip_id" in df.columns:
        stats["clips"] = df["clip_id"].nunique()
    return stats

splits_dir = Path("c:/FYP/PROJECT/product/artifacts/splits")
files = [
    "train_italian.csv", "val_italian.csv",
    "train_pitt.csv", "val_pitt.csv",
    "train_emodb.csv", "val_emodb.csv",
    "train_physionet.csv", "val_physionet.csv",
    "train.csv", "val.csv"
]

results = {}
for f in files:
    results[f] = get_stats(splits_dir / f)

import json
print(json.dumps(results, indent=2))
