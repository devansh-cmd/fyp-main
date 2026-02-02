import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

# Disable JIT for numba to avoid issues in some environments
os.environ["NUMBA_DISABLE_JIT"] = "1"

# Add src to path for imports
sys.path.append(str(Path(__file__).resolve().parent))

from audio_utils import load_audio, compute_logmel, save_logmel_png, augment_add_noise


def process_file_pitt_rigor(
    row,
    spec_out: Path,
    target_sr: int = 16000,
    window_sec: int = 20,
    overlap_sec: int = 5,
    is_train: bool = True,
) -> int:
    """
    Highly rigorous sliding window segmentation.
    Policy:
    - Train: Exactly 2 outputs (orig, noisy15db)
    - Val: Exactly 1 output (orig)
    Naming: [subject_id]_[file_id]_[start_sec]_[type].png
    """
    wav_path = Path(row["filepath"])
    subject_id = row["subject_id"]
    file_id = row["file_id"]

    try:
        y, sr = load_audio(wav_path, target_sr=target_sr)

        # Audio length in samples
        total_samples = len(y)
        window_samples = int(window_sec * sr)
        step_samples = int((window_sec - overlap_sec) * sr)

        # Determine segment windows
        if total_samples < window_samples:
            windows = [np.pad(y, (0, window_samples - total_samples), mode="constant")]
            starts = [0]
        else:
            windows = []
            starts = []
            for start in range(0, total_samples - window_samples + 1, step_samples):
                windows.append(y[start : start + window_samples])
                starts.append(start / sr)

        generated_count = 0

        for win, start_sec in zip(windows, starts):
            # Stable ID encoding: [subject]_[file_hash]_[start_sec]
            base_filename = f"{subject_id}_{file_id}_{int(start_sec):03d}"

            # --- Output 1: Clean (Always) ---
            S_db_clean = compute_logmel(win, sr, n_fft=2048, hop_length=512, n_mels=128)
            out_path_clean = spec_out / f"{base_filename}_orig.png"
            save_logmel_png(S_db_clean, sr, out_path_clean)
            generated_count += 1

            # --- Output 2: Noisy (Train Only - DETERMINISTIC) ---
            if is_train:
                # Rule: Exactly 2 images for Train (orig and noisy15db)
                y_noisy = augment_add_noise(win, snr_db=15.0)
                S_db_noisy = compute_logmel(
                    y_noisy, sr, n_fft=2048, hop_length=512, n_mels=128
                )
                out_path_noisy = spec_out / f"{base_filename}_noisy15db.png"
                save_logmel_png(S_db_noisy, sr, out_path_noisy)
                generated_count += 1

        return generated_count
    except Exception as e:
        print(f"\nError processing {wav_path.name}: {e}")
        return 0


def main():
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
    SPEC_OUT = (
        PROJECT_ROOT
        / "product"
        / "audio_preprocessing"
        / "outputs"
        / "spectrograms_pitt"
    )
    SPLIT_DIR = PROJECT_ROOT / "product" / "artifacts" / "splits"

    train_csv = SPLIT_DIR / "train_pitt.csv"
    val_csv = SPLIT_DIR / "val_pitt.csv"

    if not train_csv.exists() or not val_csv.exists():
        print(f"Error: Split CSVs not found in {SPLIT_DIR}")
        return

    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)

    print("--- Pitt Corpus Rigorous Processing ---")
    print("Policy: Train=2[orig,noisy15db] | Val=1[orig]")
    print("Naming: [subject]_[file_id]_[start]_[type].png")

    SPEC_OUT.mkdir(parents=True, exist_ok=True)

    total_segments = 0

    # Process Train Set (Deterministic 2-output)
    for _, row in tqdm(
        df_train.iterrows(), total=len(df_train), desc="Generating Train Spectrograms"
    ):
        total_segments += process_file_pitt_rigor(row, SPEC_OUT, is_train=True)

    # Process Val Set (Deterministic 1-output)
    for _, row in tqdm(
        df_val.iterrows(), total=len(df_val), desc="Generating Val Spectrograms"
    ):
        total_segments += process_file_pitt_rigor(row, SPEC_OUT, is_train=False)

    print(f"\nProcessing Complete. Total spectrograms generated: {total_segments}")
    print(f"Output Directory: {SPEC_OUT}")


if __name__ == "__main__":
    main()
