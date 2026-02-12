# Note: This is a WIP
import random

import explorepy
from explorepy.stream_processor import TOPICS
from eegprep import clean_asr
import time
import numpy as np
import polars as pl

import matplotlib.pyplot as plt

n_ch = 32

assumed_sr = 250

calib_time = 30.
asr_window = 0.05
t = 180.

max_plot_rows = 4
max_plot_cols = 2
max_plot_indices = 1000

ts_buffer = []
data_buffer = [[] for _ in range(n_ch)]
ts_buffer_no_asr = []
data_buffer_no_asr = [[] for _ in range(n_ch)]

now = time.time()


def compare_cleaned_with_uncleaned():
    file_path_raw = "/Users/sonjastefani/Documents/dev/explore-desktop/test-data/32channel_semidry_artefacts_ExG_1ch_corrupted.csv"
    raw = pl.read_csv(file_path_raw)

    file_path_cleaned = f"./corrupted_asr_cleaned-data_calib-{calib_time}_window-{asr_window}_t-{t}.csv"
    comp_cleaned_df = pl.read_csv(file_path_cleaned)

    first_ts_cleaned = comp_cleaned_df["TimeStamp"][0]
    last_ts_cleaned = comp_cleaned_df["TimeStamp"][-1]

    file_path_filtered = f"corrupted_filtered_exg_calib-{calib_time}_window-{asr_window}_t-{t}.csv"
    filtered = pl.read_csv(file_path_filtered)
    matched_filtered = filtered.remove(pl.col("TimeStamp") < first_ts_cleaned)
    matched_filtered = matched_filtered.remove(pl.col("TimeStamp") > last_ts_cleaned)

    calib_file = f"asr_calibration-data_calib-{calib_time}.csv"
    calib_file_df = pl.read_csv(calib_file)

    as_np = calib_file_df.to_numpy()
    as_np = as_np.swapaxes(1, 0)

    cleaned_df = call_clean_from_eegprep(raw, comp_cleaned_df, as_np, sr=assumed_sr, cutoff=asr_window, n_chan=32)

    dataframes = [(matched_filtered, "r", "Uncleaned (filtered)"),
                  (comp_cleaned_df, "b", f"Cleaned with window = {asr_window}s"),
                  (cleaned_df, "g", "Cleaned with clean_asr")]

    plot_comp_from_dataframes(dataframes, [1, 2, 3, 4, 5, 6, 7, 8])


def add_dead_channels_to_dataframe(dataframe, count):
    n_channels = len(dataframe.columns) - 1
    drop_indices = random.sample(range(1, n_channels), count)

    for idx in drop_indices:
        column_name = dataframe.columns[idx]
        dataframe = dataframe.replace_column(idx, pl.Series(name=column_name,
                                                            values=np.full(dataframe.shape[0], -400000.05)))

    return dataframe


def plot_comp_from_dataframes(dataframes: list[tuple[pl.DataFrame, str, str]], channels: list[int]=None):
    """Plots comparison of any dataframes' first channel from a list of tuples

    Args:
        dataframes: list of tuples with the first element in the tuple being a polars DataFrame and the second element
        in the tuple being a colour string to use for plotting with matplotlib, i.e. "b" for blue etc.
        channels: list of ints that defines which channels to plot
    """
    if channels is None:
        print("No channels supplied for plotting, exiting...")
        return
    n_channels = len(channels)

    if n_channels > max_plot_cols * max_plot_rows:
        print(f"Too many channels to reasonably plot (got: {n_channels}, "
              f"max: {max_plot_cols * max_plot_rows}), exiting...")

    plot_cols = 1
    for i in range(max_plot_cols):
        if n_channels > (i+1) * max_plot_rows:
            plot_cols += 1
    plot_rows = min(n_channels, max_plot_rows)
    print(f"n_channels: {n_channels}, plot_cols: {plot_cols}, plot_rows: {plot_rows}")
    fig, axs = plt.subplots(plot_rows, plot_cols, layout="constrained")
    print(len(axs))
    print(axs)
    it = 0
    ax = None
    for idx_to_plot in channels:
        r = it%max_plot_rows
        c = it//max_plot_rows
        print(f"r: {r}, c: {c}, idx_to_plot: {idx_to_plot}")
        ax = axs[r, c]
        for tup in dataframes:
            assert type(tup[0]) is pl.DataFrame
            assert type(tup[1]) is str
            assert type(tup[2]) is str
            ax.plot(tup[0]["TimeStamp"][:max_plot_indices], tup[0][tup[0].columns[idx_to_plot+1]][:max_plot_indices],
                    tup[1], label=tup[2])
        ax.title.set_text(f"Channel {idx_to_plot}")
        it += 1
    if ax is not None:
        legend_handles, legend_labels = ax.get_legend_handles_labels()
        fig.legend(legend_handles, legend_labels, loc='upper center')
    plt.show()


def on_asr_received(packet):
    data = packet.get_data()
    ts = data[0]
    ts_buffer.extend(ts)
    for i in range(n_ch):
        data_buffer[i].extend(data[1][i, :])


def on_filtered_received(packet):
    data_filtered = packet.get_data()
    ts_filtered = data_filtered[0]
    ts_buffer_no_asr.extend(ts_filtered)
    for i in range(n_ch):
        data_buffer_no_asr[i].extend(data_filtered[1][i, :])


def call_clean_from_eegprep(raw_data, comp_cleaned, calib_data, sr, cutoff, n_chan):
    first_ts_cleaned = comp_cleaned["TimeStamp"][0]
    last_ts_cleaned = comp_cleaned["TimeStamp"][-1]

    matched_raw = raw_data.remove(pl.col("TimeStamp") < first_ts_cleaned)
    matched_raw = matched_raw.remove(pl.col("TimeStamp") > last_ts_cleaned)
    as_np_raw = matched_raw.to_numpy()
    as_np_raw = as_np_raw.swapaxes(1, 0)
    as_np_raw_ts = as_np_raw[0, :]
    as_np_raw = as_np_raw[1:, :]
    as_dict = {"data": as_np_raw, "srate": sr, "nbchan": n_chan}
    cleaned_dict = clean_asr(as_dict, cutoff=cutoff, ref_maxbadchannels=calib_data)
    cleaned_df = cleaned_dict["data"]
    cleaned_df = cleaned_df.swapaxes(1, 0)
    cleaned_df = pl.DataFrame(cleaned_df)
    s = pl.Series("TimeStamp", as_np_raw_ts)
    cleaned_df.insert_column(0, s)

    return cleaned_df


def set_up_explore_device(t_calib=30., dev_name="Explore_DABC", notch=50., bp=(1., 30.)):
    """Connects to a device, sets up filters and performs ASR calibration"""
    dev = explorepy.Explore()
    dev.connect(dev_name)

    dev.stream_processor.add_filter(cutoff_freq=notch, filter_type="notch")
    dev.stream_processor.add_filter(cutoff_freq=bp, filter_type="bandpass")

    time.sleep(5.0)

    dev.calibrate_asr(t_calib)
    dev.stream_processor.subscribe(on_filtered_received, topic=TOPICS.filtered_ExG)
    dev.stream_processor.subscribe(on_asr_received, topic=TOPICS.asr_ExG)

    time.sleep(t_calib + 5.)

    return dev


def write_calibration_data(t_calib=30., calib_file=None):
    dev = set_up_explore_device(t_calib=t_calib)
    if calib_file is None:
        calib_file = f"asr_calibration-data_calib-{t_calib}.csv"

    time.sleep(1.)

    calib_data = dev.stream_processor.asr_processor.calibration_data_input
    calib_df = pl.DataFrame(calib_data.swapaxes(1, 0))
    calib_df.write_csv(calib_file)
    dev.disconnect()
    time.sleep(1.)

    return calib_file


def clean_data_from_file(t_calib=30., t_window=0.05, rec_length=180., calib_file=None):
    dev = set_up_explore_device(t_calib=t_calib)

    if calib_file is not None:
        calib_file_df = pl.read_csv(calib_file)
        as_np = calib_file_df.to_numpy()
        as_np = as_np.swapaxes(1, 0)

        dev.stream_processor.asr_processor.calibration_data_input = as_np
        dev.stream_processor.asr_processor.set_state_from_calibration_data(calib_data=as_np)

    time.sleep(1.)

    dev.start_asr(window=t_window)

    time.sleep(rec_length)

    ts_buffer_np = np.array(ts_buffer)
    data_buffer_np = np.array(data_buffer)

    ts_buffer_no_asr_np = np.array(ts_buffer_no_asr)
    data_buffer_no_asr_np = np.array(data_buffer_no_asr)

    n = ["TimeStamp"]
    n.extend([f"ch{i + 1}" for i in range(n_ch)])
    ret = np.vstack((ts_buffer_np, data_buffer_np))
    df = pl.DataFrame(ret.swapaxes(1, 0), schema=n)
    f_name_cleaned = f"asr_cleaned-data_calib-{t_calib}_window-{t_window}_t-{rec_length}.csv"
    df.write_csv(f_name_cleaned)  # cleaned from filtered

    ret_two = np.vstack((ts_buffer_no_asr_np, data_buffer_no_asr_np))
    df_no_asr = pl.DataFrame(ret_two.swapaxes(1, 0), schema=n)
    f_name_uncleaned = f"filtered_exg_calib-{t_calib}_window-{t_window}_t-{rec_length}.csv"
    df_no_asr.write_csv(f_name_uncleaned)  # uncleaned but filtered

    dev.disconnect()
    time.sleep(1.)

    return f_name_cleaned, f_name_uncleaned


if __name__ == '__main__':
    t_windows_to_test = [0.01, 0.05]
    t_calib_to_test = [30.]
    rec_length_to_test = [20.]

    for t_calib in t_calib_to_test:
        calib_file_path = write_calibration_data(t_calib=t_calib)

        for t_window in t_windows_to_test:
            for rec_length in rec_length_to_test:
                cleaned_path, uncleaned_path = clean_data_from_file(t_calib=t_calib, t_window=t_window,
                                                                    rec_length=rec_length, calib_file=calib_file_path)

    # TODO gather files + metadata for comparison plots
    # TODO clear buffers between runs
    # TODO add channel dropping to matrix test

    # compare_cleaned_with_uncleaned()
