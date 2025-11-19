import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import soundfile as sf
import scipy.signal

########################################
# PATH MANAGEMENT
########################################


def get_project_paths():
    """
    Centralised paths so every script agrees on dirs.
    """
    PROJECT_ROOT = Path(r"C:\FYP\PROJECT")
    AUDIO_DIR = (
        PROJECT_ROOT / "product" / "audio_preprocessing" / "data" / "ESC-50" / "audio"
    )
    META_CSV = (
        PROJECT_ROOT
        / "product"
        / "audio_preprocessing"
        / "data"
        / "ESC-50"
        / "meta"
        / "esc50.csv"
    )
    SPEC_OUT = (
        PROJECT_ROOT / "product" / "audio_preprocessing" / "outputs" / "spectrograms"
    )

    return {
        "PROJECT_ROOT": PROJECT_ROOT,
        "AUDIO_DIR": AUDIO_DIR,
        "META_CSV": META_CSV,
        "SPEC_OUT": SPEC_OUT,
    }


########################################
# AUDIO LOADING (no librosa)
########################################


def load_audio(path: Path, target_sr: int = 22050):
    """
    Load audio file as mono float32 at target_sr.
    Avoids librosa.load() so we don't trigger numba internals.
    """
    y, file_sr = sf.read(str(path), always_2d=False)

    # if stereo -> average to mono
    if isinstance(y, np.ndarray) and y.ndim > 1:
        y = np.mean(y, axis=1)

    # resample if needed using polyphase resampling (stable, no numba)
    if file_sr != target_sr:
        y = scipy.signal.resample_poly(y, target_sr, file_sr)

    # ensure float32
    y = np.asarray(y, dtype=np.float32)

    return y, target_sr


########################################
# MEL SPECTROGRAM (manual, no librosa.feature)
########################################


def _hz_to_mel(hz: np.ndarray | float):
    return 2595.0 * np.log10(1.0 + (np.asarray(hz) / 700.0))


def _mel_to_hz(mel: np.ndarray | float):
    mel = np.asarray(mel)
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(
    sr: int, n_fft: int, n_mels: int = 128, fmin: float = 0.0, fmax: float | None = None
):
    """
    Build a mel filterbank [n_mels x (1 + n_fft//2)].
    """
    if fmax is None:
        fmax = sr / 2

    # FFT bin freqs from 0..Nyquist
    fft_freqs = np.linspace(0, sr / 2, 1 + n_fft // 2)

    # mel-scaled points
    mels = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    hz_points = _mel_to_hz(mels)

    # which fft bins each mel point maps to
    bin_idx = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    fbanks = np.zeros((n_mels, 1 + n_fft // 2), dtype=np.float32)

    for m in range(1, n_mels + 1):
        left = bin_idx[m - 1]
        center = bin_idx[m]
        right = bin_idx[m + 1]

        # guard against degenerate bins
        if center == left:
            center += 1
        if right == center:
            right += 1

        # rising edge
        for k in range(left, center):
            if 0 <= k < fbanks.shape[1]:
                fbanks[m - 1, k] = (k - left) / (center - left)

        # falling edge
        for k in range(center, right):
            if 0 <= k < fbanks.shape[1]:
                fbanks[m - 1, k] = (right - k) / (right - center)

        # normalise area so filters roughly comparable energy
        denom = hz_points[m + 1] - hz_points[m - 1]
        if denom != 0:
            fbanks[m - 1, :] *= 2.0 / denom

    return fbanks


def compute_logmel(
    y: np.ndarray, sr: int, n_fft: int = 2048, hop_length: int = 512, n_mels: int = 128
):
    """
    Compute log-mel spectrogram WITHOUT librosa / numba.
    Steps:
    - STFT via scipy.signal.stft
    - power spectrum
    - mel filterbank multiply
    - log10 + ref to max
    Returns 2D np.array [n_mels, time_frames] in dB.
    """
    # STFT
    freqs, times, Zxx = scipy.signal.stft(
        y,
        fs=sr,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        nfft=n_fft,
        boundary=None,
        padded=False,
    )
    power_spec = (np.abs(Zxx) ** 2).astype(np.float32)  # [freq_bins, time_frames]

    # mel projection
    mel_fb = _mel_filterbank(sr, n_fft, n_mels=n_mels)  # [n_mels, freq_bins]
    mel_spec = mel_fb @ power_spec  # [n_mels, time_frames]

    # numerical floor so log10 doesn't explode
    mel_spec = np.maximum(mel_spec, 1e-10)

    # convert to dB and normalise
    mel_db = 10.0 * np.log10(mel_spec)
    mel_db -= np.max(mel_db)

    return mel_db.astype(np.float32)


########################################
# SAVE SPECTROGRAM IMAGE
########################################


def save_logmel_png(
    S_db: np.ndarray, sr: int, out_path: Path, title: str | None = None
):
    """
    Save mel spectrogram as a tight 224x-ish PNG (no axes).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(3, 3), dpi=224 / 3)  # ~224x224 pixels
    plt.axis("off")
    plt.imshow(
        S_db,
        origin="lower",
        aspect="auto",
        cmap="magma",
    )
    if title:
        plt.title(title)
    plt.tight_layout(pad=0)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()


########################################
# AUGMENTATIONS (pure numpy/scipy)
########################################


def augment_add_noise(y: np.ndarray, snr_db: float = 25.0):
    """
    Add Gaussian noise at a target signal-to-noise ratio.
    """
    rms_signal = np.sqrt(np.mean(y**2))
    if rms_signal == 0:
        return y.copy()

    rms_noise = rms_signal / (10.0 ** (snr_db / 20.0))
    noise = np.random.normal(0.0, rms_noise, size=y.shape).astype(np.float32)
    y_noisy = y + noise
    return y_noisy.astype(np.float32)


def augment_pitch_up(y: np.ndarray, sr: int, n_steps: float = 2.0):
    """
    Naive pitch shift:
    - speed up by rate = 2^(n_steps/12)
    - then resample back to original length so shape is consistent
    Doesn't call librosa.effects.pitch_shift (which triggers numba).
    """
    rate = 2.0 ** (n_steps / 12.0)  # semitone ratio

    # speed up / raise pitch
    y_fast = scipy.signal.resample_poly(y, up=int(1000 * rate), down=1000)

    # bring back to original length
    target_len = len(y)
    y_fast_fixed = scipy.signal.resample(y_fast, target_len)

    return y_fast_fixed.astype(np.float32)


def augment_time_stretch(y: np.ndarray, rate: float = 0.9):
    """
    Naive time stretch using resampling.
    rate < 1.0 => slower.
    After stretch, pad or crop back to original length.
    """
    stretched = scipy.signal.resample_poly(y, up=int(1000 * rate), down=1000)

    target_len = len(y)
    cur_len = len(stretched)

    if cur_len > target_len:
        # crop centre
        start = (cur_len - target_len) // 2
        stretched = stretched[start : start + target_len]
    elif cur_len < target_len:
        # zero-pad end
        pad = target_len - cur_len
        stretched = np.pad(stretched, (0, pad), mode="constant")

    return stretched.astype(np.float32)
