import pandas as pd
from pathlib import Path

def filter_csv(input_path, output_path):
    print(f"Filtering {input_path} -> {output_path}")
    df = pd.read_csv(input_path)
    
    # Determine path column
    path_col = 'filepath' if 'filepath' in df.columns else 'path'
    
    # Keep only original spectrograms
    filtered_df = df[df[path_col].str.endswith('_orig.png')]
    
    filtered_df.to_csv(output_path, index=False)
    print(f"  Rows before: {len(df)}, Rows after: {len(filtered_df)}")

def main():
    splits_dir = Path("c:/FYP/PROJECT/product/artifacts/splits")
    
    files_to_filter = [
        ("train.csv", "train_no_aug.csv"),
        ("val.csv", "val_no_aug.csv"),
        ("train_emodb.csv", "train_emodb_no_aug.csv"),
        ("val_emodb.csv", "val_emodb_no_aug.csv")
    ]
    
    for input_name, output_name in files_to_filter:
        input_file = splits_dir / input_name
        output_file = splits_dir / output_name
        
        if input_file.exists():
            filter_csv(input_file, output_file)
        else:
            print(f"Skipping {input_name} (not found)")

if __name__ == "__main__":
    main()
