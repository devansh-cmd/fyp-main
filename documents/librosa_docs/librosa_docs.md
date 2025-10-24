This file summarises the most relevant Librosa functions used in **audio preprocessing** for the ESC-50 experiments.



## 🔹 1. Audio Loading
[`librosa.load()`](https://librosa.org/doc/latest/generated/librosa.load.html)  
Loads an audio file as a floating-point time series.

```python
y, sr = librosa.load(path, sr=22050, mono=True)

## 🔹 2. Waveform Visualization
librosa.display.waveshow(y, sr=sr)

## 🔹 3. Spectrogram Generation
-librosa.feature.melspectrogram()→ computes the energy of each Mel band over time.

-S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    y = audio waveform (a NumPy array)
    sr = sample rate (e.g., 22050 Hz)
    n_fft = window size for each FFT chunk
    hop_length = how much you slide the window each step (time resolution)
    n_mels = number of Mel bands (frequency resolution)


S_dB = librosa.power_to_db(S, ref=np.max)→ converts it to a log-scaled dB spectrogram, which makes energy differences more perceptually meaningful.
n_mels: number of Mel bands

fmax: maximum frequency displayed

Convert to dB for CNN input

## 🔹 4. Data Augmentation Utilities
librosa.effects
librosa.effects.time_stretch(y, rate)
librosa.effects.pitch_shift(y, sr, n_steps)
librosa.effects.trim(y) — removes leading/trailing silence

## 🔹 5. Saving and Exporting
librosa.output.write_wav()(deprecated)
Use soundfile.write()instead.
import soundfile as sf
sf.write("output.wav", y, sr)