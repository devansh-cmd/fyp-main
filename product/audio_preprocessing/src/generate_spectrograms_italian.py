import os
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Disable JIT for numba to avoid issues in some environments
os.environ["NUMBA_DISABLE_JIT"] = "1"

# Add src to path for imports
sys.path.append(str(Path(__file__).resolve().parent))

from audio_utils import (
    load_audio,
    compute_logmel,
    save_logmel_png,
)

def process_file(wav_path: Path, spec_out: Path, target_sr: int = 16000) -> bool:
    """
    Load wav at target_sr (16kHz), compute log-mel, and save.
    Returns True if successful.
    """
    try:
        y, sr = load_audio(wav_path, target_sr=target_sr)
        
        # Consistent with unified pipeline: 2048 n_fft, 512 hop, 128 mels
        # This yields ~224x224 for typical speech lengths (5-10s)
        S_db = compute_logmel(y, sr, n_fft=2048, hop_length=512, n_mels=128)
        
        out_name = wav_path.stem + "_orig.png"
        out_path = spec_out / out_name
        
        save_logmel_png(S_db, sr, out_path)
        return True
    except Exception as e:
        print(f"\nError processing {wav_path.name}: {e}")
        return False

def main():
    # Path configuration
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "product" / "audio_preprocessing" / "data" / "Italian Parkinson's Voice and speech"
    SPEC_OUT = PROJECT_ROOT / "product" / "audio_preprocessing" / "outputs" / "spectrograms_italian"
    SPLIT_DIR = PROJECT_ROOT / "product" / "artifacts" / "splits"
    
    train_csv = SPLIT_DIR / "train_italian.csv"
    val_csv = SPLIT_DIR / "val_italian.csv"
    
    if not train_csv.exists() or not val_csv.exists():
        print(f"Error: Split CSVs not found in {SPLIT_DIR}")
        print("Please run product/datasets/make_split_italian.py first.")
        return

    # Load file lists from splits
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    all_files = pd.concat([df_train, df_val])
    
    print("--- Italian PD Spectrogram Generation ---")
    print(f"Source: {DATA_DIR}")
    print(f"Output: {SPEC_OUT}")
    print(f"Total files to process: {len(all_files)}")
    
    SPEC_OUT.mkdir(parents=True, exist_ok=True)
    
    processed_count = 0
    success_count = 0
    
    # Progress bar for better visibility
    for _, row in tqdm(all_files.iterrows(), total=len(all_files), desc="Generating Spectrograms"):
        # Path in CSV is relative to product/
        wav_path = PROJECT_ROOT / "product" / row['path']
        
        if not wav_path.exists():
            print(f"Warning: File not found: {wav_path}")
            continue
            
        if process_file(wav_path, SPEC_OUT, target_sr=16000):
            success_count += 1
        processed_count += 1

    print("\n--- Processing Complete ---")
    print(f"Total Attempted: {processed_count}")
    print(f"Successfully Generated: {success_count}")
    print(f"Spectrograms saved to: {SPEC_OUT}")

if __name__ == "__main__":
    main()
