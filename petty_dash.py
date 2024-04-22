import dash
from dash import html, dcc
import plotly.express as px
import librosa
import numpy as np

app = dash.Dash(__name__)

# Load an actual audio file
y, sr = librosa.load('14 Free Fallin.wav', sr=None)  
spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
time = np.linspace(0, len(y) / sr, num=D.shape[1])
freq = librosa.fft_frequencies(sr=sr, n_fft=2048)

spectrogram_fig = px.imshow(D, x=time, y=freq, labels={'x': 'Time', 'y': 'Frequency'}, aspect='auto', origin='lower', color_continuous_scale='Inferno')

app.layout = html.Div([
    html.H1("Audio Analysis Dashboard"),
    dcc.Graph(id='spectrogram-graph', figure=spectrogram_fig),
    dcc.Graph(id='feature-graph', figure=px.line(x=time, y=spectral_centroids[0], labels={'x': 'Time', 'y': 'Spectral Centroid'})),
    html.Audio(src='path_to_your_actual_audio_file.wav', controls=True)  # Update this path
])c

if __name__ == '__main__':
    app.run_server(debug=True)
