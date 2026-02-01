import os

os.environ["NUMBA_DISABLE_JIT"] = "1"  # Fixes the LLVM/Numba overflow bug
from pathlib import Path
from typing import List
from audio_utils import (
    load_audio,
    compute_logmel,
    save_logmel_png,
    augment_noise_random_snr,
    augment_stretch_random,
    augment_pitch_random,
    augment_random_gain,
    augment_reverb,
)

# EmoDB emotion mapping (official)
EMOTION_MAP = {
    "W": "anger",
    "L": "boredom",
    "E": "disgust",
    "A": "fear",
    "F": "happiness",
    "T": "sadness",
    "N": "neutral",
}


def parse_emodb_filename(filename: str) -> dict:
    """
    Parse EmoDB filename to extract metadata.
    Format: <speaker><sentence><repetition><emotion><version>.wav
    Example: 03a01Fa.wav
        - Speaker: 03
        - Sentence: a
        - Repetition: 01
        - Emotion: F (happiness)
        - Version: a
    """
    stem = Path(filename).stem
    if len(stem) < 7:
        raise ValueError(f"Invalid EmoDB filename format: {filename}")
    
    speaker_id = stem[0:2]
    sentence = stem[2]
    repetition = stem[3:5]
    emotion_code = stem[5]
    version = stem[6] if len(stem) > 6 else ""
    
    emotion = EMOTION_MAP.get(emotion_code, "unknown")
    
    return {
        "speaker_id": speaker_id,
        "sentence": sentence,
        "repetition": repetition,
        "emotion_code": emotion_code,
        "emotion": emotion,
        "version": version,
        "base_name": stem,
    }


def process_single_file(wav_path: Path, spec_out: Path) -> List[Path]:
    """
    Load a single wav, compute log-mel for original + 5 augmentations,
    save each as PNG into spec_out. Return list of saved PNG paths.
    
    Augmentations:
    1. Original (no augmentation)
    2. Additive noise (random SNR 5-15 dB)
    3. Time stretch (random rate 0.9-1.1)
    4. Pitch shift (random ±1-2 semitones)
    5. Random gain (0.8x-1.2x)
    6. Room reverb (mild)
    """
    saved_paths: List[Path] = []

    # Parse filename to get emotion label
    try:
        metadata = parse_emodb_filename(wav_path.name)
        base = metadata["base_name"]
        emotion = metadata["emotion"]
    except Exception as e:
        print(f"Error parsing filename {wav_path.name}: {e}")
        return saved_paths

    # Load audio
    y, sr = load_audio(wav_path)

    # 1. Original
    S_db = compute_logmel(y, sr)
    out_orig = spec_out / f"{base}_orig.png"
    save_logmel_png(S_db, sr, out_orig, title=f"{base} (orig, {emotion})")
    saved_paths.append(out_orig)

    # 2. Additive noise (random SNR 5-15 dB)
    try:
        y_noise = augment_noise_random_snr(y, snr_range=(5.0, 15.0))
        S_db_noise = compute_logmel(y_noise, sr)
        out_noise = spec_out / f"{base}_noise.png"
        save_logmel_png(S_db_noise, sr, out_noise, title=f"{base} (noise)")
        saved_paths.append(out_noise)
    except Exception as e:
        print(f"Noise augmentation failed for {wav_path.name}: {e}")

    # 3. Time stretch (random 0.9-1.1)
    try:
        y_stretch = augment_stretch_random(y, rate_range=(0.9, 1.1))
        S_db_stretch = compute_logmel(y_stretch, sr)
        out_stretch = spec_out / f"{base}_stretch.png"
        save_logmel_png(S_db_stretch, sr, out_stretch, title=f"{base} (stretch)")
        saved_paths.append(out_stretch)
    except Exception as e:
        print(f"Stretch augmentation failed for {wav_path.name}: {e}")

    # 4. Pitch shift (random ±1-2 semitones)
    try:
        y_pitch = augment_pitch_random(y, sr, semitone_range=(-2, 2))
        S_db_pitch = compute_logmel(y_pitch, sr)
        out_pitch = spec_out / f"{base}_pitch.png"
        save_logmel_png(S_db_pitch, sr, out_pitch, title=f"{base} (pitch)")
        saved_paths.append(out_pitch)
    except Exception as e:
        print(f"Pitch augmentation failed for {wav_path.name}: {e}")

    # 5. Random gain (0.8x-1.2x)
    try:
        y_gain = augment_random_gain(y, gain_range=(0.8, 1.2))
        S_db_gain = compute_logmel(y_gain, sr)
        out_gain = spec_out / f"{base}_gain.png"
        save_logmel_png(S_db_gain, sr, out_gain, title=f"{base} (gain)")
        saved_paths.append(out_gain)
    except Exception as e:
        print(f"Gain augmentation failed for {wav_path.name}: {e}")

    # 6. Room reverb (mild)
    try:
        y_reverb = augment_reverb(y, sr, room_scale=0.3)
        S_db_reverb = compute_logmel(y_reverb, sr)
        out_reverb = spec_out / f"{base}_reverb.png"
        save_logmel_png(S_db_reverb, sr, out_reverb, title=f"{base} (reverb)")
        saved_paths.append(out_reverb)
    except Exception as e:
        print(f"Reverb augmentation failed for {wav_path.name}: {e}")

    return saved_paths


def main() -> None:
    # EmoDB-specific paths
    # Dynamically find project root (3 levels up from this script)
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
    AUDIO_DIR = PROJECT_ROOT / "product" / "audio_preprocessing" / "data" / "EmoDB-wav"
    SPEC_OUT = PROJECT_ROOT / "product" / "audio_preprocessing" / "outputs" / "spectrograms_emodb"
    SPEC_OUT.mkdir(parents=True, exist_ok=True)

    wav_paths = sorted(AUDIO_DIR.glob("*.wav"))
    total_wavs = len(wav_paths)
    if total_wavs == 0:
        print("No .wav files found in", AUDIO_DIR)
        return

    print(f"Found {total_wavs} WAV files in EmoDB dataset")
    print("Generating 6 spectrograms per file (1 orig + 5 augmentations)")
    print(f"Expected total: {total_wavs * 6} PNG files\n")

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
            print(f"[{idx}/{total_wavs}] processed {wav_path.name} → {len(saved)} images")

    print(f"\n{'='*50}")
    print(f"Processed {processed}/{total_wavs} wav files")
    print(f"Generated {total_pngs} spectrogram PNGs total")
    print(f"Output directory: {SPEC_OUT}")


if __name__ == "__main__":
    main()
