import os

os.environ["NUMBA_DISABLE_JIT"] = "1"  # Fixes the LLVM/Numba overflow bug
from pathlib import Path
from typing import List

from audio_utils import (
    get_project_paths,
    load_audio,
    compute_logmel,
    save_logmel_png,
    augment_add_noise,
    augment_pitch_up,
    augment_time_stretch,
)


def process_single_file(wav_path: Path, spec_out: Path) -> List[Path]:
    """
    Load a single wav, compute log-mel for original + 3 augmentations,
    save each as PNG into spec_out. Return list of saved PNG paths.
    """
    saved_paths: List[Path] = []

    # load audio
    y, sr = load_audio(wav_path)

    base = wav_path.stem

    # original
    S_db = compute_logmel(y, sr)
    out_orig = spec_out / f"{base}_orig.png"
    save_logmel_png(S_db, sr, out_orig, title=f"{base} (orig)")
    print("Saved spectrogram →", out_orig)
    saved_paths.append(out_orig)

    # noisy
    try:
        y_noisy = augment_add_noise(y, sr)
        S_db_noisy = compute_logmel(y_noisy, sr)
        out_noisy = spec_out / f"{base}_noisy.png"
        save_logmel_png(S_db_noisy, sr, out_noisy, title=f"{base} (noisy)")
        saved_paths.append(out_noisy)
    except Exception as e:
        print(f"Noise augmentation failed for {wav_path.name}: {e}")

    # pitch up 2 semitones
    y_pitch = augment_pitch_up(y, sr, n_steps=2)
    S_db_pitch = compute_logmel(y_pitch, sr)
    out_pitch = spec_out / f"{base}_pitchUp2.png"
    save_logmel_png(S_db_pitch, sr, out_pitch, title=f"{base} (pitchUp2)")
    print("Saved spectrogram →", out_pitch)
    saved_paths.append(out_pitch)

    # time stretch 0.9
    y_stretch = augment_time_stretch(y, rate=0.9)
    S_db_stretch = compute_logmel(y_stretch, sr)
    out_stretch = spec_out / f"{base}_stretch0.9.png"
    save_logmel_png(S_db_stretch, sr, out_stretch, title=f"{base} (stretch0.9)")
    print("Saved spectrogram →", out_stretch)
    saved_paths.append(out_stretch)

    return saved_paths


def main() -> None:
    paths = get_project_paths()
    AUDIO_DIR = paths["AUDIO_DIR"]
    SPEC_OUT = paths["SPEC_OUT"]
    SPEC_OUT.mkdir(parents=True, exist_ok=True)

    wav_paths = sorted(AUDIO_DIR.glob("*.wav"))
    total_wavs = len(wav_paths)
    if total_wavs == 0:
        print("No .wav files found in", AUDIO_DIR)
        return

    processed = 0
    total_pngs = 0

    for idx, wav_path in enumerate(wav_paths, start=1):
        try:
            saved = process_single_file(wav_path, SPEC_OUT)
            processed += 1
            total_pngs += len(saved)
        except Exception as exc:
            print(f"Error processing {wav_path.name}: {exc}")
            continue

        if idx % 50 == 0 or idx == total_wavs:
            print(f"[{idx}/{total_wavs}] processed {wav_path.name} (saved 4 images)")

    print(f"Processed {processed} wav files")
    print(f"Generated {total_pngs} spectrogram PNGs total")


if __name__ == "__main__":
    main()
