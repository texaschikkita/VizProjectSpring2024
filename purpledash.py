import dash
from dash import html, dcc
import plotly.express as px
import librosa
import numpy as np

# Initialize the Dash app
app = dash.Dash(__name__)

# Load audio
y, sr = librosa.load('09 Purple Rain.wav', sr=None)
# Compute various features
spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)

# Spectrogram as before
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
time = np.linspace(0, len(y) / sr, num=D.shape[1])
freq = librosa.fft_frequencies(sr=sr, n_fft=2048)
spectrogram_fig = px.imshow(D, x=time, y=freq, aspect='auto', origin='lower')

# Setup the app layout
app.layout = html.Div([
    html.H1("Audio Analysis Dashboard"),
    dcc.Graph(id='spectrogram-graph', figure=spectrogram_fig),
    dcc.Graph(id='feature-graph', figure=px.line(x=time, y=spectral_centroids[0], labels={'x': 'Time', 'y': 'Spectral Centroid'})),
    html.Audio(src='path_to_your_audio_file.wav', controls=True)
])

if __name__ == '__main__':
    app.run_server(debug=True)
