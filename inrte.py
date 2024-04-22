import librosa
import numpy as np
import plotly as plt
import plotly.express as px
from plotly.graph_objs import Scatterpolargl
import soundfile

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

# Create a list to store plot data
data = []

for feature, name, color in zip(normalized_features, feature_names, colors):
  # Collapse feature dimensions if necessary (same as before)
  theta = np.linspace(0, 2 * np.pi, feature.size)
  r = feature
  data.append(
      Scatterpolargl(
          theta=theta,
          r=r,
          mode='markers',
          # Set marker size and transparency
          marker=dict(size=10, opacity=0.75, color=color),
          # Set hover text information
          text=name,
          hoverinfo="text"
      )
  )

layout = dict(
    title='Audio Feature Visualization (Polar)',
    # Remove unnecessary tick labels and gridlines from the plot
    showticklabels=False, 
    tickvals=[],
    gridlines=False
)

fig = dict(data=data, layout=layout)
# Use Plotly to display interactive plot
#plotly.offline.ipylaunch(fig)
 