import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys

# Import custom audio utils
# Assuming script is in product/audio_preprocessing/src/
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.append(str(SCRIPT_DIR))

from audio_utils import load_audio, compute_logmel, save_logmel_png

def process_file(wav_path: Path, spec_out: Path, target_sr: int = 2000):
    """
    Process a single heart sound file into a log-mel spectrogram.
    Optimized for heart sounds (2kHz sampling).
    """
    try:
        # Load and resample to 2kHz
        y, sr = load_audio(wav_path, target_sr=target_sr)
        
        # Compute Log-Mel Spectrogram
        # Heart sounds are low freq, so n_fft=512 is plenty at 2kHz (approx 250ms window)
        S_db = compute_logmel(y, sr, n_fft=512, hop_length=128, n_mels=128)
        
        # Save as PNG (normalized for CNN)
        out_name = wav_path.stem + "_orig.png"
        out_path = spec_out / out_name
        
        # We use a custom save function that handles grayscale/RGB mapping
        save_logmel_png(S_db, sr, out_path)
        return True
    except Exception as e:
        print(f"\nError processing {wav_path.name}: {e}")
        return False

def main():
    SPLIT_DIR = PROJECT_ROOT / "product" / "artifacts" / "splits"
    TRAIN_CSV = SPLIT_DIR / "train_physionet.csv"
    VAL_CSV = SPLIT_DIR / "val_physionet.csv"
    
    SPEC_OUT = PROJECT_ROOT / "product" / "audio_preprocessing" / "outputs" / "spectrograms_physionet"
    SPEC_OUT.mkdir(parents=True, exist_ok=True)
    
    if not TRAIN_CSV.exists() or not VAL_CSV.exists():
        print("Error: Train/Val split CSVs not found. Run make_split_physionet.py first.")
        return

    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    all_files = pd.concat([train_df, val_df])
    
    print(f"Total PhysioNet files to process: {len(all_files)}")
    
    success_count = 0
    
    for _, row in tqdm(all_files.iterrows(), total=len(all_files), desc="Generating Heart Spectrograms"):
        wav_path = PROJECT_ROOT / row['path']
        
        if not wav_path.exists():
            print(f"Warning: File not found: {wav_path}")
            continue
            
        if process_file(wav_path, SPEC_OUT, target_sr=2000):
            success_count += 1

    print(f"\n--- PhysioNet Processing Complete ---")
    print(f"Successfully Generated: {success_count} / {len(all_files)}")
    print(f"Spectrograms saved to: {SPEC_OUT}")

if __name__ == "__main__":
    main()
