### Laura Hainke, 2022
### Niclas Brand, 2024
### niclas@mentalab.com

### Mentalab GmbH, Weinstraße 4, 80333 Munich, Germany

import argparse
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
import numpy as np

CH_LABELS = ['Cz', 'CP1', 'CP2', 'Pz', 'P3', 'P4', 'CPz', 'POz']

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
    epoch_length = offset_end - offset_st # fixed epoch length

    for i, event_t in enumerate(event_times):
        idx = np.argmax(sig_times > event_t)

        # Ensure there is enough data for the epoch
        if (idx + offset_st) >= 0 and (idx + offset_end) <= sig.shape[1]:
            epoch = sig[:, idx + offset_st:idx + offset_end]
            if epoch.shape[1] == epoch_length:
                epoch_list.append(epoch)
            else:
                print(f"Skipping epoch {i} due to size mismatch: {epoch.shape}")
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

    label_target = 'sw_11'
    label_nontarget = 'sw_10'

    t_min = -.3
    t_max = 1.
    p2p_max = 500000  # rejection criteria, units in uV

    exg_filename = args.filename + '_ExG.csv'
    marker_filename = args.filename + '_Marker.csv'

    # Import data
    exg = pd.read_csv(exg_filename)
    ch_nums = ['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8']
    exg = exg[['TimeStamp']+ch_nums]
    markers = pd.read_csv(marker_filename)

    ts_sig = exg['TimeStamp'].to_numpy()
    ts_markers_nontarget = markers[markers.Code.isin([label_nontarget])]['TimeStamp'].to_numpy()
    ts_markers_target = markers[markers.Code.isin([label_target])]['TimeStamp'].to_numpy()
    sig = exg[ch_nums].to_numpy().T
    # sig -= (sig[0, :]/2)
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
