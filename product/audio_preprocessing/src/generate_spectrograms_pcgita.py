"""
PC-GITA DDK Spectrogram Generation
Generates Log-Mel spectrograms from all DDK WAV files.
No augmentation — clean originals only (same protocol as Italian PD).
"""
import os
import sys
from pathlib import Path
from tqdm import tqdm

os.environ["NUMBA_DISABLE_JIT"] = "1"
sys.path.append(str(Path(__file__).resolve().parent))

from audio_utils import load_audio, compute_logmel, save_logmel_png


def process_file(wav_path: Path, spec_out: Path, target_sr: int = 16000) -> bool:
    """Load wav, resample to 16kHz, compute log-mel, save as PNG."""
    try:
        y, sr = load_audio(wav_path, target_sr=target_sr)
        S_db = compute_logmel(y, sr, n_fft=2048, hop_length=512, n_mels=128)
        out_path = spec_out / (wav_path.stem + "_orig.png")
        save_logmel_png(S_db, sr, out_path)
        return True
    except Exception as e:
        print(f"\nError processing {wav_path.name}: {e}")
        return False


def main():
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "product" / "datasets" / "raw" / "pcgita_ddk"
    SPEC_OUT = PROJECT_ROOT / "product" / "audio_preprocessing" / "outputs" / "spectrograms_pcgita"

    SPEC_OUT.mkdir(parents=True, exist_ok=True)

    # Gather all WAVs from hc/ and pd/ subdirectories
    wavs = list(DATA_DIR.glob("**/*.wav"))
    print(f"--- PC-GITA DDK Spectrogram Generation ---")
    print(f"Source: {DATA_DIR}")
    print(f"Output: {SPEC_OUT}")
    print(f"Total WAV files: {len(wavs)}")

    success = 0
    for wav in tqdm(wavs, desc="Generating Spectrograms"):
        if process_file(wav, SPEC_OUT, target_sr=16000):
            success += 1

    print(f"\n--- Complete ---")
    print(f"Generated: {success}/{len(wavs)} spectrograms")
    print(f"Saved to: {SPEC_OUT}")


if __name__ == "__main__":
    main()
