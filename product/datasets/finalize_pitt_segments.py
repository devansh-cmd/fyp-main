import pandas as pd
from pathlib import Path

def finalize_pitt_segments(project_root: Path):
    """
    Creates the final segment-level CSVs for training.
    Maps generated PNGs to labels and subjects.
    """
    spec_dir = project_root / "product" / "audio_preprocessing" / "outputs" / "spectrograms_pitt"
    split_dir = project_root / "product" / "artifacts" / "splits"
    
    # Load original file-level metadata to get labels
    train_meta = pd.read_csv(split_dir / "train_pitt.csv")
    val_meta = pd.read_csv(split_dir / "val_pitt.csv")
    
    # Create lookup for file_id -> label
    id_to_label = pd.concat([train_meta, val_meta]).set_index('file_id')['label'].to_dict()
    id_to_subject = pd.concat([train_meta, val_meta]).set_index('file_id')['subject_id'].to_dict()
    
    # List all generated segments
    all_segments = list(spec_dir.glob("*.png"))
    
    train_segments = []
    val_segments = []
    
    # Identify which split IDs belong to
    train_ids = set(train_meta['file_id'])
    val_ids = set(val_meta['file_id'])
    
    for seg_path in all_segments:
        # Filename: [subject]_[file_id]_[start]_[type].png
        parts = seg_path.stem.split("_")
        if len(parts) < 4:
            continue
            
        file_id = parts[1]
        label = id_to_label.get(file_id)
        subject_id = id_to_subject.get(file_id)
        
        rel_path = f"audio_preprocessing/outputs/spectrograms_pitt/{seg_path.name}"
        
        record = {
            "filepath": rel_path,
            "subject_id": subject_id,
            "label": label,
            "file_id": file_id
        }
        
        if file_id in train_ids:
            train_segments.append(record)
        elif file_id in val_ids:
            val_segments.append(record)
            
    df_train_seg = pd.DataFrame(train_segments)
    df_val_seg = pd.DataFrame(val_segments)
    
    df_train_seg.to_csv(split_dir / "train_pitt_segments.csv", index=False)
    df_val_seg.to_csv(split_dir / "val_pitt_segments.csv", index=False)
    
    print("Finalized Segments:")
    print(f"  Train: {len(df_train_seg)} images")
    print(f"  Val:   {len(df_val_seg)} images")
    print(f"CSVs saved to {split_dir}")

if __name__ == "__main__":
    PROJ_ROOT = Path(__file__).resolve().parent.parent.parent
    finalize_pitt_segments(PROJ_ROOT)
