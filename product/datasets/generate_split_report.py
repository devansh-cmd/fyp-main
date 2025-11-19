from pathlib import Path
import pandas as pd
import json
import re


def clip_id(p):
    stem = Path(p).stem
    stem = re.sub(
        r"_(orig|noisy|pitchUp\d+|stretch\d+(?:\.\d+)?)$", "", stem, flags=re.IGNORECASE
    )
    return stem


def main(
    train_csv=r"C:/FYP/PROJECT/product/artifacts/splits/clean_train.csv",
    val_csv=r"C:/FYP/PROJECT/product/artifacts/splits/clean_val.csv",
    out_path=r"C:/FYP/PROJECT/product/artifacts/splits/split_report.json",
):
    train = pd.read_csv(train_csv)
    val = pd.read_csv(val_csv)

    train_ids = set(train["filepath"].apply(clip_id))
    val_ids = set(val["filepath"].apply(clip_id))

    overlap = train_ids & val_ids
    assert len(overlap) == 0, f"Leak persists: {len(overlap)} overlapping clip IDs."

    report = {
        "train_png_count": len(train),
        "val_png_count": len(val),
        "train_clip_count": len(train_ids),
        "val_clip_count": len(val_ids),
        "train_class_count": train["label"].nunique(),
        "val_class_count": val["label"].nunique(),
        "overlap_clips": len(overlap),
    }

    Path(out_path).write_text(json.dumps(report, indent=2))
    print("✅ Saved split_report.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
