import pandas as pd
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from analysis import CCAAnalysis

# Initialization
fs = 250
lf = .5
hf = 40

event_freq = [10, 7.5]
label_nontarget = 10
label_target = 11

t_min = 0
t_max = 6.

n_chan = 4
chan_list = ['ch' + str(i) for i in range(1, n_chan)]
exg_filename = 'Salman_23_08_SSVEP_ExG.csv'
marker_filename = 'Salman_23_08_SSVEP_Marker.csv'

# Import data
exg = pd.read_csv(exg_filename)
markers = pd.read_csv(marker_filename)

ts_sig = exg['TimeStamp'].to_numpy()
ts_markers = markers[markers.Code.isin([label_nontarget, label_target])]['TimeStamp'].to_numpy()
groundtruth = markers[markers.Code.isin([label_nontarget,
                                         label_target])]['Code'].to_numpy() - 10  # Subtract 10 to map to 0 and 1 labels

sig = exg[chan_list].to_numpy().T


def custom_filter(exg, lf, hf, fs, type):
    """

    Args:
        exg: EEG signal with the shape: (N_chan, N_sample)
        lf: Low cutoff frequency
        hf: High cutoff frequency
        fs: Sampling rate
        type: Filter type, 'bandstop' or 'bandpass'

    Returns:
        (numpy ndarray): Filtered signal (N_chan, N_sample)
    """
    N = 4
    b, a = signal.butter(N, [lf / fs, hf / fs], type)
    return signal.filtfilt(b, a, exg)


def extract_epochs(sig, sig_times, event_times, t_min, t_max, fs):
    """ Extracts epochs from signal

    Args:
        sig: EEG signal with the shape: (N_chan, N_sample)
        sig_times: Timestamp of the EEG samples with the shape (N_sample)
        event_times: Event marker times
        t_min: Starting time of the epoch relative to the event time
        t_max: End time of the epoch relative to the event time
        fs: Sampling rate

    Returns:

    """
    offset_st = int(t_min * fs)
    offset_end = int(t_max * fs)
    epoch_list = []
    for i, event_t in enumerate(event_times):
        idx = np.argmax(sig_times > event_t)
        epoch_list.append(sig[:, idx + offset_st:idx + offset_end])
    return np.array(epoch_list)


# Signal filtering
filt_sig = custom_filter(sig, 45, 55, fs, 'bandstop')
filt_sig = custom_filter(filt_sig, lf, hf, fs, 'bandpass')

epochs = extract_epochs(filt_sig, ts_sig, ts_markers, t_min, t_max, fs)

t_win = [.25, .5, .75, 1, 1.25, 1.5, 2, 2.5, 3, 4, 6]
predictions = {str(key): [] for key in t_win}
accuracies = []

for tmax in t_win:
    cca = CCAAnalysis(freqs=event_freq, win_len=tmax, s_rate=250, n_harmonics=2)
    for eeg_chunk in epochs:
        scores = cca.apply_cca(eeg_chunk[:, 0:int(tmax * fs)].T)
        predictions[str(tmax)].append(np.argmax(scores))
    accuracies.append(np.count_nonzero(np.array(predictions[str(tmax)]) == groundtruth) / len(predictions[str(tmax)]) * 100)

# print(accuracies)
fig, ax = plt.subplots()
ax.plot(t_win, accuracies, marker='*', markersize=5)
ax.set_ylabel('Accuracy (%)')
ax.set_xlabel('Time window (s)')
ax.set_xlim(0, 6.5)
ax.set_ylim(40, 100)
ax.grid(True)
plt.show()
print('s')
