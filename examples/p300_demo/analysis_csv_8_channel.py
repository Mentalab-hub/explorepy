import argparse
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
import numpy as np

CH_LABELS = ['TP9', 'Cz', 'Pz', 'CP1', 'CP2', 'P3', 'P4', 'Oz']


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
        (numpy ndarray): EEG epochs
    """
    offset_st = int(t_min * fs)
    offset_end = int(t_max * fs)
    epoch_list = []
    for i, event_t in enumerate(event_times):
        idx = np.argmax(sig_times > event_t)
        epoch_list.append(sig[:, idx + offset_st:idx + offset_end])
    return np.array(epoch_list)


def reject_bad_epochs(epochs, p2p_max):
    """Rejects bad epochs based on a peak-to-peak amplitude criteria
    Args:
        epochs: Epochs of EEG signal
        p2p_max: maximum peak-to-peak amplitude

    Returns:
        (numpy ndarray):EEG epochs
    """
    temp = epochs.reshape((epochs.shape[0], -1))
    res = epochs[np.ptp(temp, axis=1) < p2p_max, :, :]
    print(f"{temp.shape[0] - res.shape[0]} epochs out of {temp.shape[0]} epochs have been rejected.")
    return res


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


def main():
    parser = argparse.ArgumentParser(description="P300 analysis script")
    parser.add_argument("-f", "--filename", dest="filename", type=str, help="Recorded file name")
    args = parser.parse_args()
    fs = 250
    lf = .5
    hf = 40

    label_target = 11
    label_nontarget = 10

    t_min = -.3
    t_max = 1.
    p2p_max = 500000  # rejection criteria, units in uV

    exg_filename = args.filename + '_ExG.csv'
    marker_filename = args.filename + '_Marker.csv'

    # Import data
    exg = pd.read_csv(exg_filename)
    markers = pd.read_csv(marker_filename)

    ts_sig = exg['TimeStamp'].to_numpy()
    ts_markers_nontarget = markers[markers.Code.isin([label_nontarget])]['TimeStamp'].to_numpy()
    ts_markers_target = markers[markers.Code.isin([label_target])]['TimeStamp'].to_numpy()
    sig = exg[['ch'+str(i) for i in range(1, 9)]].to_numpy().T
    sig -= (sig[0, :]/2)
    filt_sig = custom_filter(sig, 45, 55, fs, 'bandstop')
    filt_sig = custom_filter(filt_sig, lf, hf, fs, 'bandpass')

    target_epochs = extract_epochs(filt_sig, ts_sig, ts_markers_target, t_min, t_max, fs)
    nontarget_epochs = extract_epochs(filt_sig, ts_sig, ts_markers_nontarget, t_min, t_max, fs)

    target_epochs = reject_bad_epochs(target_epochs, p2p_max)
    nontarget_epochs = reject_bad_epochs(nontarget_epochs, p2p_max)

    erp_target = target_epochs.mean(axis=0)
    erp_nontarget = nontarget_epochs.mean(axis=0)

    t = np.linspace(t_min, t_max, erp_target.shape[1])
    fig, axes = plt.subplots(nrows=4, ncols=2)
    fig.tight_layout()
    for i, ax in enumerate(axes.flatten()):
        ax.plot(t, erp_nontarget[i, :], label='Non-target')
        ax.plot(t, erp_target[i, :], 'tab:orange', label='Target')
        ax.plot([0, 0], [-30, 30], linestyle='dotted', color='black')
        ax.set_ylabel('\u03BCV')
        ax.set_xlabel('Time (s)')
        ax.set_title(CH_LABELS[i])
        ax.set_ylim([-10, 20])
        ax.legend()
    plt.show()


if __name__ == '__main__':
    main()
