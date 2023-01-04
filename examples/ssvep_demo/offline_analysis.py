import pandas as pd
from scipy import signal
import numpy as np
import mne
import matplotlib.pyplot as plt
import argparse
from analysis import CCAAnalysis


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
    b, a = signal.butter(N, [lf / (fs/2), hf / (fs/2)], type)
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


def main():
    parser = argparse.ArgumentParser(description="SSVEP offline analysis script")
    parser.add_argument("-f", "--filename", dest="filename", type=str, help="Recorded file name")
    args = parser.parse_args()

    # Initialization
    fs = 250
    lf = .5
    hf = 40

    event_freq = [10, 7.5]
    label_nontarget = 10
    label_target = 11

    t_min = -1
    t_max = 6.

    n_chan = 4
    chan_list = ['ch' + str(i) for i in range(1, n_chan + 1)]
    chan_name = ['O1', 'O2', 'POz', 'Oz']
    class_names = ['Left', 'Right']

    exg_filename = args.filename + '_ExG.csv'
    marker_filename = args.filename + '_Marker.csv'

    # Import data
    exg = pd.read_csv(exg_filename)
    markers = pd.read_csv(marker_filename)
    ts_sig = exg['TimeStamp'].to_numpy()
    ts_markers = markers[markers.Code.isin([label_nontarget, label_target])]['TimeStamp'].to_numpy()
    groundtruth = markers[markers.Code.isin([label_nontarget,
                                             label_target])]['Code'].to_numpy() - 10  # Subtract 10 to map to 0 and 1 labels
    sig = exg[chan_list].to_numpy().T

    # Signal filtering
    filt_sig = custom_filter(sig, 45, 55, fs, 'bandstop')
    filt_sig = custom_filter(filt_sig, lf, hf, fs, 'bandpass')

    epochs = extract_epochs(filt_sig, ts_sig, ts_markers, t_min, t_max, fs)

    t_win = [.25, .5, .75, 1, 1.25, 1.5, 2, 2.5, 3, 4, 6]
    predictions = {str(key): [] for key in t_win}
    accuracies = []
    t_offset = np.absolute(t_min)
    for tmax in t_win:
        cca = CCAAnalysis(freqs=event_freq, win_len=tmax, s_rate=250, n_harmonics=2)
        for eeg_chunk in epochs:
            scores = cca.apply_cca(eeg_chunk[:, int(t_offset*fs):int((tmax + t_offset) * fs)].T)
            predictions[str(tmax)].append(np.argmax(scores))
        accuracies.append(
            np.count_nonzero(np.array(predictions[str(tmax)]) == groundtruth) / len(predictions[str(tmax)]) * 100)

    # print(accuracies)
    fig, ax = plt.subplots()
    ax.plot(t_win, accuracies, marker='*', markersize=5)
    ax.set_ylabel('Accuracy (%)')
    ax.set_xlabel('Time window (s)')
    ax.set_xlim(0, 6.5)
    ax.set_ylim(40, 105)
    ax.grid(True)
    plt.show()

    # Time-frequency analysis
    freqs = np.arange(5., 15., .3)
    n_cycles = freqs

    times = np.linspace(t_min, t_max, epochs.shape[2])

    power_1 = mne.time_frequency.tfr_array_morlet(epochs[groundtruth == 0., :, :], sfreq=fs,
                                                  freqs=freqs, n_cycles=n_cycles,
                                                  output='avg_power')
    power_2 = mne.time_frequency.tfr_array_morlet(epochs[groundtruth == 1., :, :], sfreq=fs,
                                                  freqs=freqs, n_cycles=n_cycles,
                                                  output='avg_power')

    mne.baseline.rescale(power_1, times, (None, 0), mode='mean', copy=False)
    mne.baseline.rescale(power_2, times, (None, 0), mode='mean', copy=False)
    power = np.stack([power_1, power_2])
    x, y = mne.viz.centers_to_edges(times, freqs)

    fig, ax = plt.subplots(n_chan, 2)
    for j in range(2):
        for i in range(n_chan):
            mesh = ax[i, j].pcolormesh(x, y, power[j, i], cmap='RdBu_r', vmin=-3500, vmax=1500)
            ax[i, j].set_title('TFR - ' + chan_name[i] + ' - ' + class_names[j])
            ax[i, j].set(ylim=freqs[[0, -1]], xlabel='Time (s)', ylabel='Frequency (Hz)')
    plt.tight_layout()
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(mesh, cax=cbar_ax)
    plt.show()


if __name__ == '__main__':
    main()
