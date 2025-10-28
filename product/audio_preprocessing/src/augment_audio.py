import os
os.environ["NUMBA_DISABLE_JIT"] = "1"  # avoids the LLVM/Numba issue on some systems
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

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

def make_noisy_spec(y, sr, base_name: str, out_dir: Path) -> Path | None:
    """
    Add gaussian noise -> mel -> save PNG as *_noisy.png
    Returns saved path or None if it fails.
    """
    try:
        y_noisy = augment_add_noise(y, snr_db=25.0)
        S_db_noisy = compute_logmel(y_noisy, sr)
        out_noisy = out_dir / f"{base_name}_noisy.png"
        save_logmel_png(S_db_noisy, sr, out_noisy, title=f"{base_name} (noisy)")
        return out_noisy
    except Exception as e:
        print(f"[WARN] noisy aug failed for {base_name}: {e}")
        return None


def make_pitch_spec(y, sr, base_name: str, out_dir: Path, n_steps: float = 2.0) -> Path | None:
    """
    Pitch shift up by n_steps semitones -> mel -> save *_pitchUp2.png
    Returns saved path or None if it fails.
    """
    try:
        y_pitch = augment_pitch_up(y, sr, n_steps=n_steps)
        S_db_pitch = compute_logmel(y_pitch, sr)
        out_pitch = out_dir / f"{base_name}_pitchUp2.png"
        save_logmel_png(S_db_pitch, sr, out_pitch, title=f"{base_name} (pitchUp2)")
        return out_pitch
    except Exception as e:
        print(f"[WARN] pitch aug failed for {base_name}: {e}")
        return None


def make_stretch_spec(y, sr, base_name: str, out_dir: Path, rate: float = 0.9) -> Path | None:
    """
    Time-stretch by 'rate' (e.g. 0.9 = slightly slower) -> mel -> save *_stretch0.9.png
    Returns saved path or None if it fails.
    """
    try:
        y_stretch = augment_time_stretch(y, rate=rate)
        S_db_stretch = compute_logmel(y_stretch, sr)
        out_stretch = out_dir / f"{base_name}_stretch0.9.png"
        save_logmel_png(S_db_stretch, sr, out_stretch, title=f"{base_name} (stretch0.9)")
        return out_stretch
    except Exception as e:
        print(f"[WARN] stretch aug failed for {base_name}: {e}")
        return None


def process_single_wav(wav_path: Path, spec_out: Path) -> List[Path]:
    """
    For one wav file:
    - load audio
    - generate ONLY the augmented spectrograms
    Return list of successfully saved PNG paths.
    """
    saved_paths: List[Path] = []

    # load original audio
    y, sr = load_audio(wav_path)
    base_name = wav_path.stem 

    # augmentation 1: noise
    noisy_path = make_noisy_spec(y, sr, base_name, spec_out)
    if noisy_path is not None:
        saved_paths.append(noisy_path)

    # augmentation 2: pitch up
    pitch_path = make_pitch_spec(y, sr, base_name, spec_out, n_steps=2.0)
    if pitch_path is not None:
        saved_paths.append(pitch_path)

    # augmentation 3: stretch
    stretch_path = make_stretch_spec(y, sr, base_name, spec_out, rate=0.9)
    if stretch_path is not None:
        saved_paths.append(stretch_path)

    return saved_paths


def main() -> None:
    """
    Walk ESC-50 audio dir.
    For each *.wav:
      - if *_noisy / *_pitchUp2 / *_stretch0.9 already exist, skip that one
      - else generate the missing ones
    Keep stats and print summary at end.
    """
    paths = get_project_paths()
    AUDIO_DIR = paths["AUDIO_DIR"]
    SPEC_OUT = paths["SPEC_OUT"]
    SPEC_OUT.mkdir(parents=True, exist_ok=True)

    wav_paths = sorted(AUDIO_DIR.glob("*.wav"))
    total_wavs = len(wav_paths)
    if total_wavs == 0:
        print("No .wav files found in", AUDIO_DIR)
        return

    augmented_wavs = 0
    new_pngs = 0

    for idx, wav_path in enumerate(wav_paths, start=1):
        base_name = wav_path.stem

        # quick existence check so we don't re-do work every run
        noisy_file   = SPEC_OUT / f"{base_name}_noisy.png"
        pitch_file   = SPEC_OUT / f"{base_name}_pitchUp2.png"
        stretch_file = SPEC_OUT / f"{base_name}_stretch0.9.png"

        if noisy_file.exists() and pitch_file.exists() and stretch_file.exists():
            if idx % 100 == 0 or idx == total_wavs:
                print(f"[SKIP {idx}/{total_wavs}] {base_name} already augmented")
            continue

        try:
            saved_list = process_single_wav(wav_path, SPEC_OUT)
            if len(saved_list) > 0:
                augmented_wavs += 1
                new_pngs += len(saved_list)

            if idx % 50 == 0 or idx == total_wavs:
                print(f"[{idx}/{total_wavs}] augmented {base_name} -> {len(saved_list)} imgs")

        except Exception as e:
            print(f"[ERROR] could not augment {wav_path.name}: {e}")
            # keep going, don't crash whole run
            continue

    print("====================================")
    print(f"Total WAV files found:        {total_wavs}")
    print(f"WAVs that got augmentation:   {augmented_wavs}")
    print(f"New augmented PNGs generated: {new_pngs}")
    print("Done.")


if __name__ == "__main__":
    main()
