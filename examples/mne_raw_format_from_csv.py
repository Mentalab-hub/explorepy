import pandas as pd
import mne
import matplotlib.pyplot as plt
from mne import export


def create_mne_raw_format(file_name, channel_number):
    file_name_root = file_name.split('.')[0][:-4]

    sampling_freq = pd.read_csv(file_name_root + '_Meta.csv', delimiter=',')['sr'][0]

    # Create some dummy metadata
    n_channels = channel_number
    ch_types = ["eeg"] * n_channels
    info = mne.create_info(len(ch_types), sfreq=sampling_freq, ch_types=ch_types)

    data_frame = pd.read_csv(file_name, delimiter=',')
    data_frame = data_frame.drop('TimeStamp', axis = 1)
    # convert to volt, because MNE expects data in Volt unit
    data_frame = data_frame.div(1e6)
    data_frame = data_frame.transpose()
    raw_data = mne.io.RawArray(data_frame, info).notch_filter(freqs=50)
    raw_data.plot(show_scrollbars=True, show_scalebars=True, remove_dc=True, highpass=1, lowpass=40)
    plt.show()
    return raw_data

def explort_eeglab_format(raw_input_df, output_file_name):
    export.export_raw(output_file_name, raw_input_df,
                      fmt='eeglab',
                      overwrite=True, physical_range=[-400000, 400000])


raw_df = create_mne_raw_format(file_name='chr180924_ExG.csv', channel_number=8)
explort_eeglab_format(output_file_name='output_edf.edf', raw_input_df=raw_df)
