import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np

# Load the audio file
file_path = '/mnt/data/09 Purple Rain.wav'  # Update this path to the correct file location if different
y, sr = librosa.load(file_path, sr=None)

# Compute the short-time Fourier transform (STFT)
D = librosa.stft(y)
S_DB = librosa.amplitude_to_db(np.abs(D), ref=np.max)

# Calculate the spectral centroid and bandwidth
spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]

# Convert frame counts to time for plotting spectral features
frames = range(len(spectral_centroids))
t = librosa.frames_to_time(frames, sr=sr)

# Plotting
fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Waveform plot
librosa.display.waveshow(y, sr=sr, ax=ax[0], alpha=0.5)
ax[0].set(title='Waveform', ylabel='Amplitude')

# Spectrogram plot
img = librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='log', ax=ax[1])
fig.colorbar(img, ax=ax[1], format="%+2.0f dB")
ax[1].set(title='Spectrogram')

# Spectral features plot
ax[2].plot(t, spectral_centroids, color='r', label='Spectral Centroid')
ax[2].fill_between(t, spectral_centroids - spectral_bandwidth / 2, 
                   spectral_centroids + spectral_bandwidth / 2, color='b', alpha=0.5, 
                   label='Spectral Bandwidth')
ax[2].legend(loc='upper right')
ax[2].set(title='Spectral Features', ylabel='Frequency (Hz)')

# Shared x-axis configurations
plt.xlabel('Time (s)')
plt.tight_layout()
plt.show()
