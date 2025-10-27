from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Spectrogram parameters used across functions
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 128


def get_project_paths() -> Dict[str, Path]:
    """
    Discover the project root by walking up from this file until a folder
    containing 'product/audio_preprocessing' is found. Fallback to Path.cwd()
    if not found. Returns a dict with PROJECT_ROOT, AUDIO_PREPROC_ROOT,
    AUDIO_DIR, SPEC_OUT, WAVEFORM_OUT. Also ensures SPEC_OUT and WAVEFORM_OUT
    directories exist.
    """
    this_file = Path(__file__).resolve()
    project_root: Path | None = None

    # Walk up parents to find a directory that contains product/audio_preprocessing
    for parent in [this_file] + list(this_file.parents):
        candidate = parent if parent.is_dir() else parent.parent
        if (candidate / "product" / "audio_preprocessing").exists():
            project_root = candidate
            break

    if project_root is None:
        project_root = Path.cwd()

    AUDIO_PREPROC_ROOT = project_root / "product" / "audio_preprocessing"
    AUDIO_DIR = AUDIO_PREPROC_ROOT / "data" / "ESC-50" / "audio"
    SPEC_OUT = AUDIO_PREPROC_ROOT / "outputs" / "spectrograms"
    WAVEFORM_OUT = AUDIO_PREPROC_ROOT / "outputs" / "waveforms"

    # Ensure output directories exist
    SPEC_OUT.mkdir(parents=True, exist_ok=True)
    WAVEFORM_OUT.mkdir(parents=True, exist_ok=True)

    return {
        "PROJECT_ROOT": project_root,
        "AUDIO_PREPROC_ROOT": AUDIO_PREPROC_ROOT,
        "AUDIO_DIR": AUDIO_DIR,
        "SPEC_OUT": SPEC_OUT,
        "WAVEFORM_OUT": WAVEFORM_OUT,
    }


def load_audio(wav_path: Path) -> Tuple[np.ndarray, int]:
    """
    Load an audio file using librosa with original sampling rate.
    Returns (y, sr).
    """
    y, sr = librosa.load(str(wav_path), sr=None)
    return y, sr


def save_waveform_png(y: np.ndarray, sr: int, out_path: Path, title: str):
    """
    Plot and save a waveform PNG using librosa.display.waveshow.
    Prints a short confirmation on save.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved waveform → {out_path}")


def compute_logmel(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Compute a log-mel spectrogram (in dB) from audio time series.
    Uses N_FFT=1024, HOP_LENGTH=512, N_MELS=128, fmin=0, fmax=sr/2.
    Returns the spectrogram in dB (S_db).
    """
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=0,
        fmax=sr / 2,
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db


def save_logmel_png(S_db: np.ndarray, sr: int, out_path: Path, title: str):
    """
    Save a log-mel spectrogram as a PNG using librosa.display.specshow.
    Prints a short confirmation on save.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        S_db,
        sr=sr,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="mel",
        cmap="magma",
    )
    plt.colorbar(format="%+2.0f dB")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved spectrogram → {out_path}")


def augment_add_noise(y: np.ndarray, snr_db: float = 25.0) -> np.ndarray:
    """
    Add light Gaussian noise to achieve a target SNR (in dB).
    Returns the noisy signal.
    """
    if y.size == 0:
        return y
    sig_power = np.mean(y.astype(np.float64) ** 2)
    # Avoid division by zero
    if sig_power == 0:
        return y
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = sig_power / snr_linear
    noise = np.random.normal(0.0, np.sqrt(noise_power), size=y.shape)
    return y + noise


def augment_pitch_up(y: np.ndarray, sr: int, n_steps: float = 2.0) -> np.ndarray:
    """
    Shift the pitch of the audio up by n_steps semitones.
    Uses librosa.effects.pitch_shift.
    """
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)


def augment_time_stretch(y: np.ndarray, rate: float = 0.9) -> np.ndarray:
    """
    Time-stretch the audio by a given rate using librosa.effects.time_stretch.
    If the audio is too short (<2 samples) returns the original audio unchanged.
    """
    if y.size < 2:
        return y
    # librosa.effects.time_stretch expects a 1D audio array
    try:
        y_stretched = librosa.effects.time_stretch(y, rate=rate)
    except Exception:
        # As a safe fallback, return original if time-stretching fails
        return y
    return y_stretched