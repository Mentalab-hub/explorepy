import copy

import numpy as np
import pandas as pd
from scipy import signal
import seaborn as sns
import matplotlib.pyplot as plt

fs = 2000
lf = .5
hf = 40
n_ch = 32
N = 2
zim = np.zeros((n_ch, N * 2))
print(np.shape(zim))

def notch_filter(exg, fs):
    f0 = 50.0  # Frequency to be removed from signal (Hz)
    Q = 30.0  # Quality factor
    # Design notch filter
    b, a = signal.iirnotch(f0, Q, fs)
    return signal.filtfilt(b, a, exg)


def bp_filter(exg, lf, hf, fs, type='bandpass'):
    lf = (2 * lf) / fs
    hf = (2 * hf) / fs
    b, a = signal.butter(N, [lf, hf], type)
    data = signal.lfilter(b, a, exg)
    return data


df = pd.read_csv('file_name0_ExG.csv', delimiter=',')
sig = df['ch1']
sig = sig - np.mean(sig)
#notched = notch_filter(sig, fs)
bandpassed = bp_filter(sig, lf, hf, fs)


fft_vals = np.fft.fft(bandpassed)
freqs = np.fft.fftfreq(len(bandpassed), 1/fs)
fft_magnitude = np.abs(fft_vals)


#fig, axes = plt.subplots(2, 1, figsize=(12, 5))
sns.lineplot(bandpassed)
#ax = sns.lineplot(x=freqs, y=fft_magnitude, ax=axes[1])
#ax.set_xlim(0, 40)
plt.show()
