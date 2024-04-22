# Load the audio file
file_path = '01 Smells Like Teen Spirit.wav'
audio, sample_rate = librosa.load(file_path, sr=None)

# Perform a Short-Time Fourier Transform (STFT) to get the frequency content over time
stft = librosa.stft(audio)

# Convert the STFT to decibels, which is a more useful measure for human hearing
db_stft = librosa.amplitude_to_db(abs(stft))

# Plot the spectrogram
plt.figure(figsize=(14, 5))
librosa.display.specshow(db_stft, sr=sample_rate, x_axis='time', y_axis='log', cmap='coolwarm')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectral Analysis')
plt.show()










y, sr = librosa.load('01 Smells Like Teen Spirit.wav')

y, sr = librosa.load('09 Purple Rain.wav')

y, sr = librosa.load('14 Free Fallin.wav')



 # Calculate the spectral centroid and bandwidth
spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]



__# Convert frame counts to time
frames = range(len(spectral_centroids))
t = librosa.frames_to_time(frames, sr=sr)


# Load the audio file
file_path = '01 Smells Like Teen Spirit.wav'
audio, sample_rate = librosa.load(file_path, sr=None)

# Perform a Short-Time Fourier Transform (STFT) to get the frequency content over time
stft = librosa.stft(audio)

# Convert the STFT to decibels, which is a more useful measure for human hearing
db_stft = librosa.amplitude_to_db(abs(stft))

# Plot the spectrogram
plt.figure(figsize=(14, 5))
librosa.display.specshow(db_stft, sr=sample_rate, x_axis='time', y_axis='log', cmap='coolwarm')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectral Analysis')
plt.show()


 # Calculate the spectral centroid and bandwidth
spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]

# Convert frame counts to time
frames = range(len(spectral_centroids))
t = librosa.frames_to_time(frames, sr=sr)


# Mel-Frequency Cepstral Coefficients (MCFFs):
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.show()


# Spectral Contrast:
spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
librosa.display.specshow(spec_contrast, x_axis='time')
plt.colorbar()
plt.title('Spectral Contrast')
plt.show()




# Harmonic & Percussove Separation: 
y_harm, y_perc = librosa.effects.hpss(y)
plt.figure(figsize=(10, 3))
librosa.display.waveshow(y_harm, sr=sr, alpha=0.5, color='b')
librosa.display.waveshow(y_perc, sr=sr, alpha=0.5, color='r')
plt.title('Harmonic and Percussive Separation')
plt.show()




y_harm, y_perc = librosa.effects.hpss(y)
plt.figure(figsize=(10, 3))
librosa.display.waveshow(y_harm, sr=sr, alpha=0.5, color='yellow')
librosa.display.waveshow(y_perc, sr=sr, alpha=0.5, color='green')
plt.title('Harmonic and Percussive Separation')
plt.show()


# Chroma Feature Extraction: 
chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
plt.colorbar()
plt.title('Chroma')
plt.show()




# Plotting the Spectral Centroid along the waveform
plt.figure(figsize=(15, 5))
librosa.display.waveshow(y, sr=sr, alpha=0.4, color="purple")
plt.plot(t, spectral_centroids, color='r') # Spectral centroid
plt.plot(t, spectral_centroids - spectral_bandwidth / 2, color='b', alpha=0.5) # Min range
plt.plot(t, spectral_centroids + spectral_bandwidth / 2, color='b', alpha=0.5) # Max range
plt.title('Spectral Centroid and Bandwidth')
plt.show()


# Calculate the RMS energy
rms_energy = librosa.feature.rms(y=y)[0]

# Plotting the RMS along the waveform
plt.figure(figsize=(15, 5))
librosa.display.waveshow(y, sr=sr, alpha=0.4, color="navy")
plt.plot(t, rms_energy, color='g') # RMS Energy
plt.title('RMS Energy Over Time')
plt.show()




# Detect onsets
onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
onset_times = librosa.frames_to_time(onset_frames, sr=sr)

# Plotting the onsets
plt.figure(figsize=(15, 5))
# librosa.display.waveshow(y, sr=sr, alpha=0.4)
# plt.vlines(onset_times, ymin=-1, ymax=1, color='r')
librosa.display.waveshow(y, sr=sr, alpha=0.4, color="orange")
plt.title('Onsets Over Time')
plt.show()





import librosa
import numpy as np
import matplotlib.pyplot as plt

# Load your audio file
y, sr = librosa.load('01 Smells Like Teen Spirit.wav')

# Generate time axis data
time = np.linspace(0, len(y) / sr, num=len(y))

# Plot the waveform
plt.figure(figsize=(14, 5))
plt.plot(time, y)
plt.title('Audio Waveform')
plt.ylabel('Amplitude')
plt.xlabel('Time (s)')
plt.show()





import librosa
import numpy as np
import matplotlib.pyplot as plt

# Load the audio file
y, sr = librosa.load('01 Smells Like Teen Spirit.wav')

# Time axis for the audio file
t = np.linspace(0, len(y) / sr, num=len(y))

# Plot the audio waveform
plt.figure(figsize=(15, 5))
plt.plot(t, y, alpha=0.4)
plt.title('Audio Waveform')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()

# Detect onsets
onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
onset_times = librosa.frames_to_time(onset_frames, sr=sr)

# Plotting the onsets over the waveform
plt.figure(figsize=(15, 5))
plt.plot(t, y, alpha=0.4)
plt.vlines(onset_times, ymin=min(y), ymax=max(y), color='r')
plt.title('Onsets Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()

# Calculate the RMS energy
rms_energy = librosa.feature.rms(y=y)[0]
rms_times = librosa.frames_to_time(range(len(rms_energy)), sr=sr)

# Plotting the RMS energy over time
plt.figure(figsize=(15, 5))
plt.plot(t, y, alpha=0.4)
plt.plot(rms_times, rms_energy, color='g')
plt.title('RMS Energy Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude / RMS Energy')
plt.show()

# Calculate the spectral centroid and bandwidth
spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
centroid_times = librosa.frames_to_time(range(len(spectral_centroids)), sr=sr)

# Plotting the Spectral Centroid and Bandwidth over the waveform
plt.figure(figsize=(15, 5))
plt.plot(t, y, alpha=0.4)
plt.plot(centroid_times, spectral_centroids, color='r')
plt.fill_between(centroid_times, spectral_centroids - spectral_bandwidth / 2, spectral_centroids + spectral_bandwidth / 2, color='b', alpha=0.5)
plt.title('Spectral Centroid and Bandwidth')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.show()



import librosa
import numpy as np
import matplotlib.pyplot as plt

# Load your audio file
y, sr = librosa.load('01 Smells Like Teen Spirit.wav')

# Generate time axis data
t = np.linspace(0, len(y) / sr, num=len(y))

# Calculate spectral centroid and bandwidth
spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
frames = range(len(spectral_centroids))
t_centroid = librosa.frames_to_time(frames, sr=sr)  # Time for centroid plots

# Plot the audio waveform
plt.figure(figsize=(15, 5))
plt.plot(t, y, alpha=0.4, color="green")  # Waveform in blue
plt.plot(t_centroid, spectral_centroids, color='r')  # Spectral centroid in red
plt.fill_between(t_centroid, spectral_centroids - spectral_bandwidth / 2, spectral_centroids + spectral_bandwidth / 2, color='g', alpha=0.5)  # Bandwidth range in blue
plt.title('Spectral Centroid and Bandwidth')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude / Frequency (Hz)')
plt.show()





zero_crossings = librosa.zero_crossings(y, pad=False)
print("Total Zero Crossings:", sum(zero_crossings))



import librosa
import numpy as np
import matplotlib.pyplot as plt

# Load the audio file
y, sr = librosa.load('01 Smells Like Teen Spirit.wav', sr=None)

# Extract features
spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)

# Convert to decibels
spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

# Setup for plotting
fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw={'projection': 'polar'})
ax.axis('off')

# Create the meshgrid for the spectrogram plot
# Ensure theta and r have one more point than the spectrogram_db dimensions
theta = np.linspace(0, 2*np.pi, spectrogram_db.shape[1] + 1)
r = np.linspace(0, 1, spectrogram_db.shape[0] + 1)
theta, r = np.meshgrid(theta, r)

# Plot the spectrogram
ax.pcolormesh(theta, r, spectrogram_db, shading='flat')

plt.show()















import librosa
import numpy as np
import matplotlib.pyplot as plt

# Load the audio file
y, sr = librosa.load('01 Smells Like Teen Spirit.wav', sr=None)

# Extract features
chromagram = librosa.feature.chroma_stft(y=y, sr=sr)

# Setup for plotting
fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw={'projection': 'polar'})
ax.axis('off')

# Create the meshgrid for the chromagram plot
# Ensure theta and r have one more point than the chromagram dimensions
chroma_theta = np.linspace(0, 2*np.pi, chromagram.shape[1] + 1)
chroma_r = np.linspace(1, 1.1, chromagram.shape[0] + 1)
chroma_theta, chroma_r = np.meshgrid(chroma_theta, chroma_r)

# Plot the chromagram
ax.pcolormesh(chroma_theta, chroma_r, chromagram, shading='flat')

plt.show()












import librosa
import numpy as np
import matplotlib.pyplot as plt

# Load the audio file
y, sr = librosa.load('01 Smells Like Teen Spirit.wav', sr=None)

# Extract features
spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
chromagram = librosa.feature.chroma_stft(y=y, sr=sr)

# Convert spectrogram to decibels
spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

# Setup for plotting
fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw={'projection': 'polar'})
ax.axis('off')

# Spectrogram grid
theta = np.linspace(0, 2 * np.pi, spectrogram_db.shape[1] + 1)
r = np.linspace(0, 1, spectrogram_db.shape[0] + 1)
theta_grid, r_grid = np.meshgrid(theta, r)

# Chromagram grid
chroma_theta = np.linspace(0, 2 * np.pi, chromagram.shape[1] + 1)
chroma_r = np.linspace(1.05, 1.15, chromagram.shape[0] + 1)  # Adjusted to sit just outside the spectrogram
chroma_theta_grid, chroma_r_grid = np.meshgrid(chroma_theta, chroma_r)

# Plot the spectrogram
ax.pcolormesh(theta_grid, r_grid, spectrogram_db, shading='flat')

# Plot the chromagram
ax.pcolormesh(chroma_theta_grid, chroma_r_grid, chromagram, shading='flat', cmap='viridis')  # Optionally set a different colormap

plt.show()






import numpy as np
import matplotlib.pyplot as plt

# Since we don't have the actual audio data to analyze and create an EQ curve for "Smells Like Teen Spirit",
# we will create a mock-up EQ curve similar to the previously provided example for "Purple Rain".

# Frequency bands in Hz (mock values)
frequencies = np.array([20, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 20000])
# Gain values for each band in dB (mock values)
gains = np.array([-3, -2, 2, -1, -2, 0, 3, -1, 2, 1, -2])

# Create the plot
plt.figure(figsize=(10, 6))

# Plotting the parametric EQ curve
plt.semilogx(frequencies, gains, marker='o', linestyle='-', color='black')

# Adding color coded areas for different filters
plt.fill_between(frequencies, gains, where=gains > 0, interpolate=True, color='grey', alpha=0.3)
plt.fill_between(frequencies, gains, where=gains <= 0, interpolate=True, color='grey', alpha=0.3)

# Highlight specific EQ bands with colors
colors = ['green', 'yellow', 'purple', 'fuchsia', 'blue', 'red']
filters = [(20, 100), (10000, 20000), (20, 400), (400, 1600), (1600, 6400), (6400, 20000)]
for (low_cut, high_cut), color in zip(filters, colors):
    plt.fill_between(frequencies, gains, where=(frequencies >= low_cut) & (frequencies <= high_cut), interpolate=True, color=color, alpha=0.7)

# Setting the x-axis limits to the audible range
plt.xlim(20, 20000)

# Setting the y-axis limits to the dB range of the EQ
plt.ylim(-18, 18)

# Labels and grid
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain (dB)')
plt.title('Parametric EQ Curve for "Smells Like Teen Spirit"')
plt.grid(True, which="both", ls="--", linewidth=0.5)

# Adding key frequency markers
for freq in frequencies:
    plt.axvline(x=freq, color='k', linestyle='--', linewidth=0.5, alpha=0.7)

# Show the plot
plt.show()

















import librosa
import numpy as np
import matplotlib.pyplot as plt

# Load the audio file
y, sr = librosa.load('01 Smells Like Teen Spirit.wav', sr=None)

# Extract features
spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
chromagram = librosa.feature.chroma_stft(y=y, sr=sr)

# Setup for plotting
fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw={'projection': 'polar'})
ax.axis('off')

# Setup mesh for spectrogram
theta = np.linspace(0, 2*np.pi, spectrogram_db.shape[1] + 1)
r = np.linspace(0, 1, spectrogram_db.shape[0] + 1)
theta, r = np.meshgrid(theta, r)
ax.pcolormesh(theta, r, spectrogram_db, shading='flat')

# Setup mesh for chromagram
chroma_theta = np.linspace(0, 2*np.pi, chromagram.shape[1] + 1)
chroma_r = np.linspace(1, 1.1, chromagram.shape[0] + 1)
chroma_theta, chroma_r = np.meshgrid(chroma_theta, chroma_r)
ax.pcolormesh(chroma_theta, chroma_r, chromagram, shading='flat', cmap='cool')

plt.show()











# Frequency bands in Hz (mock values)
frequencies = np.array([20, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 20000])
# Gain values for each band in dB (mock values)
gains = np.array([-3, -2, 2, -1, -2, 0, 3, -1, 2, 1, -2])

# Create the plot
plt.figure(figsize=(10, 6))

# Plotting the parametric EQ curve
plt.semilogx(frequencies, gains, marker='o', linestyle='-', color='black')

# Adding color coded areas for different filters
plt.fill_between(frequencies, gains, where=gains>0, interpolate=True, color='grey', alpha=0.3)
plt.fill_between(frequencies, gains, where=gains<=0, interpolate=True, color='grey', alpha=0.3)

# Highlight specific EQ bands with colors
colors = ['green', 'yellow', 'purple', 'fuchsia', 'blue', 'red']
filters = [(20, 100), (10000, 20000), (20, 400), (400, 1600), (1600, 6400), (6400, 20000)]
for (low_cut, high_cut), color in zip(filters, colors):
    plt.fill_between(frequencies, gains, where=(frequencies >= low_cut) & (frequencies <= high_cut), interpolate=True, color=color, alpha=0.7)

# Setting the x-axis limits to the audible range
plt.xlim(20, 20000)

# Setting the y-axis limits to the dB range of the EQ
plt.ylim(-18, 18)

# Labels and grid
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain (dB)')
plt.title('Parametric EQ Curve for "Purple Rain"')
plt.grid(True, which="both", ls="--", linewidth=0.5)

# Adding key frequency markers
for freq in frequencies:
    plt.axvline(x=freq, color='k', linestyle='--', linewidth=0.5, alpha=0.7)

# Show the plot
plt.show()







import librosa
import numpy as np
import matplotlib.pyplot as plt

# Load the audio file
y, sr = librosa.load('09 Purple Rain.wav', sr=22050)  # Reduced sampling rate

# Extract features
melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, hop_length=1024)
chromagram = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=1024)
spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=1024)
tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
harmonic, percussive = librosa.effects.hpss(y)

# Convert to decibels for the melspectrogram
melspectrogram_db = librosa.power_to_db(melspectrogram, ref=np.max)

# Normalize and prepare for padding
features = [melspectrogram_db, chromagram, spectral_contrast, tonnetz, harmonic]
features_padded = []

for f in features:
    # Check if the feature is 1D or 2D and apply appropriate padding
    if f.ndim == 1:
        # It's a 1D array
        padded = np.pad(f, (0, max(0, 500 - f.shape[0])), mode='constant', constant_values=0)
        padded = padded[np.newaxis, :]  # Make it 2D by adding an axis
    else:
        # It's a 2D array
        padded = np.pad(f, ((0, 0), (0, max(0, 500 - f.shape[1]))), mode='constant', constant_values=0)
    
    features_padded.append(padded)

# Visualizing a portion to manage memory
plt.figure(figsize=(10, 4))
plt.imshow(features_padded[0], aspect='auto', origin='lower')
plt.title('Reduced Melspectrogram')
plt.colorbar()
plt.show()









import librosa
import numpy as np
import matplotlib.pyplot as plt

# Load a segment of the audio file
y, sr = librosa.load('09 Purple Rain.wav', sr=22050, duration=60)  # Load only the first 60 seconds

# Extract features with a larger hop_length to reduce data size
melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=32, hop_length=2048)
chromagram = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12, hop_length=2048)
spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=2048)
tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
harmonic, percussive = librosa.effects.hpss(y)

# Convert to decibels for the melspectrogram
melspectrogram_db = librosa.power_to_db(melspectrogram, ref=np.max)

# Normalize and prepare for visualization
features = [melspectrogram_db, chromagram, spectral_contrast, tonnetz, harmonic]
feature_names = ['Mel Spectrogram', 'Chromagram', 'Spectral Contrast', 'Tonnetz', 'Harmonic']

# Create a figure with subplots for each feature
fig, axes = plt.subplots(len(features), 1, figsize=(8, 15), subplot_kw={'projection': 'polar'})

# Plot each feature in a separate subplot
for ax, feature, name in zip(axes, features, feature_names):
    if feature.ndim > 1:
        # Average the feature over its rows to reduce complexity
        feature = np.mean(feature, axis=0)
    
    theta = np.linspace(0, 2 * np.pi, num=feature.shape[0])
    r = feature  # Magnitude as radius
    
    ax.scatter(theta, r)
    ax.set_title(name)
    ax.set_ylim(0, np.max(r) + 0.1)  # Adjust the limits

plt.tight_layout()
plt.show()






import librosa
import numpy as np
import matplotlib.pyplot as plt

# Load a segment of the audio file
y, sr = librosa.load('09 Purple Rain.wav', sr=22050, duration=60)  # Load only the first 60 seconds

# Extract features with a larger hop_length to reduce data size
melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=32, hop_length=2048)
chromagram = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12, hop_length=2048)
spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=2048)
tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
harmonic, percussive = librosa.effects.hpss(y)

# Convert to decibels for the melspectrogram
melspectrogram_db = librosa.power_to_db(melspectrogram, ref=np.max)

# Normalize and prepare for visualization
features = [melspectrogram_db, chromagram, spectral_contrast, tonnetz, harmonic]
feature_names = ['Mel Spectrogram', 'Chromagram', 'Spectral Contrast', 'Tonnetz', 'Harmonic']
colors = ['red', 'green', 'blue', 'purple', 'orange']  # Assign a color to each feature

# Create a polar plot
fig, ax = plt.subplots(1, 1, figsize=(10, 8), subplot_kw={'projection': 'polar'})

# Plot each feature in the same subplot with different colors
for feature, name, color in zip(features, feature_names, colors):
    if feature.ndim > 1:
        # Average the feature over its rows to reduce complexity
        feature = np.mean(feature, axis=0)
    
    theta = np.linspace(0, 2 * np.pi, num=feature.shape[0])
    r = feature  # Magnitude as radius
    
    ax.scatter(theta, r, label=name, color=color, alpha=0.75, s=10)  # Use smaller dots with some transparency

ax.set_title('Audio Features Overlaid on Polar Plot')
ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))  # Adjust legend position outside the plot

plt.show()



















# Load a segment of the audio file
y, sr = librosa.load('09 Purple Rain.wav', sr=22050, duration=60)  # Load only the first 60 seconds

# Extract features with a larger hop_length to reduce data size
melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=32, hop_length=2048)
chromagram = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12, hop_length=2048)
spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=2048)
tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
harmonic, percussive = librosa.effects.hpss(y)

# Convert to decibels for the melspectrogram
melspectrogram_db = librosa.power_to_db(melspectrogram, ref=np.max)

# Normalize and prepare for visualization
features = [melspectrogram_db, chromagram, spectral_contrast, tonnetz, harmonic]
feature_names = ['Mel Spectrogram', 'Chromagram', 'Spectral Contrast', 'Tonnetz', 'Harmonic']
colors = ['red', 'green', 'blue', 'purple', 'orange']  # Assign a color to each feature

# Create a polar plot
fig, ax = plt.subplots(1, 1, figsize=(10, 8), subplot_kw={'projection': 'polar'})

# Plot each feature in the same subplot with different colors
for feature, name, color in zip(features, feature_names, colors):
    if feature.ndim > 1:
        # Average the feature over its rows to reduce complexity
        feature = np.mean(feature, axis=0)
    
    theta = np.linspace(0, 2 * np.pi, num=feature.shape[0])
    r = feature  # Magnitude as radius
    
    ax.scatter(theta, r, label=name, color=color, alpha=0.75, s=10)  # Use smaller dots with some transparency

ax.set_title('Audio Features Overlaid on Polar Plot')
ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))  # Adjust legend position outside the plot
# Remove the polar labels
ax.set_xticklabels([])

# Remove the radial labels
ax.set_yticklabels([])

# Remove the grid
ax.grid(False)

# Remove the outer circle (spine)
ax.spines['polar'].set_visible(False)

# Optionally, if you want to remove the radial ticks as well:
ax.yaxis.set_ticks([])

# Show the plot without the polar labels and grid
plt.show()







import librosa
import numpy as np
import matplotlib.pyplot as plt

# Load a segment of the audio file
y, sr = librosa.load('09 Purple Rain.wav', sr=22050, duration=60)  # Load only the first 60 seconds

# Extract features
melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=32, hop_length=2048)
chromagram = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=2048)
spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=2048)
tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
harmonic, percussive = librosa.effects.hpss(y)

# Normalize the features to a common scale
features = [harmonic, melspectrogram, chromagram, spectral_contrast, tonnetz]  # Place harmonic first
feature_names = ['Harmonic', 'Mel Spectrogram', 'Chromagram', 'Spectral Contrast', 'Tonnetz']
colors = ['orange', 'red', 'green', 'blue', 'purple']  # Corresponding colors, with harmonic's color first

normalized_features = []

for feature in features:
    # Scale features to be between 0 and 1
    min_val = np.min(feature)
    max_val = np.max(feature)
    scaled_feature = (feature - min_val) / (max_val - min_val)
    normalized_features.append(scaled_feature)

# Create a polar plot
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': 'polar'})

# Plot each normalized feature in the same subplot with different colors
for feature, name, color in zip(normalized_features, feature_names, colors):
    # Collapse feature dimensions if necessary
    if feature.ndim > 1:
        feature = np.mean(feature, axis=0)
    
    # Map to polar coordinates
    theta = np.linspace(0, 2 * np.pi, feature.size)
    r = feature
    
    ax.scatter(theta, r, alpha=0.75, s=10, label=name, color=color)  # Use smaller dots with some transparency

# Customize the plot - removing labels, ticks, and spines
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.grid(False)
ax.spines['polar'].set_visible(False)
ax.yaxis.set_ticks([])

# Add a legend
ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

# Show the plot
plt.show()





