# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchaudio.transforms import Spectrogram

# %%
train_data_x = np.load("../data/vitaldb/valid_EEG.npy")
train_data_y = np.load("../data/vitaldb/valid_label.npy")

# %%
def plot_spectrogram(spectrogram, ax):
    # Convert to frequencies to log scale and transpose so that the time is
    # represented in the x-axis (columns).
    log_spec = np.log(spectrogram)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    Y = range(height)
    ax.pcolormesh(X, Y, log_spec)
# %%
preprocessing_layer = Spectrogram(n_fft=100)
a = preprocessing_layer.forward(waveform = torch.Tensor(train_data_x[1000]))
# %%
fig, ax = plt.subplots(1, figsize=(10, 10))
plot_spectrogram(a.numpy(), ax)
# %%
