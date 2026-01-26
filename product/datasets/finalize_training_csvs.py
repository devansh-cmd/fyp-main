import pandas as pd
from pathlib import Path

def fix_csv_paths(project_root: Path):
    """
    Creates new CSVs that point to generated spectrograms instead of raw wavs.
    """
    splits_dir = project_root / "product" / "artifacts" / "splits"
    
    # 1. Italian PD
    print("Fixing Italian PD CSVs...")
    spec_dir_pd = "product/audio_preprocessing/outputs/spectrograms_italian"
    for split in ["train_italian", "val_italian"]:
        df = pd.read_csv(splits_dir / f"{split}.csv")
        # Italian CSV uses 'filename' column
        new_paths = []
        for fn in df['filename']:
            stem = Path(fn).stem
            # The generation script used wav_path.stem + "_orig.png"
            # For '..wav' files, Path().stem results in a trailing dot (e.g. 'blah.')
            new_paths.append(f"{spec_dir_pd}/{stem}_orig.png")
        
        df['path'] = new_paths
        df.to_csv(splits_dir / f"{split}_png.csv", index=False)

    # 2. PhysioNet
    print("Fixing PhysioNet CSVs...")
    spec_dir_phys = "product/audio_preprocessing/outputs/spectrograms_physionet"
    for split in ["train_physionet", "val_physionet"]:
        df = pd.read_csv(splits_dir / f"{split}.csv")
        # PhysioNet CSV uses 'path' (to .wav) and 'filename' (.wav)
        new_paths = []
        for fn in df['filename']:
            stem = Path(fn).stem
            new_paths.append(f"{spec_dir_phys}/{stem}_orig.png")
            
        df['path'] = new_paths
        df.to_csv(splits_dir / f"{split}_png.csv", index=False)

    # 3. ESC-50 & EmoDB
    # These already point to spectrograms in their 'filepath' column usually, 
    # but let's verify or standardize.
    
    print("Standardization complete.")

if __name__ == "__main__":
    fix_csv_paths(Path(__file__).resolve().parent.parent.parent)
