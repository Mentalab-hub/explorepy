# Note: This is a WIP

import explorepy
from explorepy.stream_processor import TOPICS
from eegprep import clean_asr
from eegprep.utils.asr import asr_calibrate
import time
import numpy as np
import polars as pl

import matplotlib.pyplot as plt

n_ch = 32

assumed_sr = 250

calib_time = 30.
asr_window = 0.05
t = 180.

ts_buffer = []
data_buffer = [[] for _ in range(n_ch)]
ts_buffer_no_asr = []
data_buffer_no_asr = [[] for _ in range(n_ch)]

now = time.time()

def compare_cleaned_with_uncleaned():
    file_path_raw = "/Users/sonjastefani/Documents/dev/explore-desktop/test-data/32channel_semidry_artefacts_ExG.csv"
    raw = pl.read_csv(file_path_raw)

    file_path_cleaned = f"./asr_cleaned-data_calib-{calib_time}_window-{asr_window}_t-{t}-extra.csv"
    comp_cleaned_df = pl.read_csv(file_path_cleaned)

    calib_file = f"asr_calibration-data_calib-{calib_time}.csv"
    calib_file_df = pl.read_csv(calib_file)

    as_np = calib_file_df.to_numpy()
    as_np = as_np.swapaxes(1, 0)

    cleaned_df = call_clean_from_eegprep(raw, comp_cleaned_df, as_np, sr=assumed_sr, cutoff=asr_window, n_chan=32)

    dataframes = [(comp_cleaned_df, "b"), (cleaned_df, "s")]
    
    plot_comp_from_dataframes(dataframes)


def plot_comp_from_dataframes(dataframes: list[tuple[pl.DataFrame, str]]):
    """Plots comparison of any dataframes' first channel from a list of tuples

    Args:
        dataframes: list of tuples with the first element in the tuple being a polars DataFrame and the second element
        in the tuple being a colour string to use for plotting with matplotlib, i.e. "b" for blue etc.
    """
    fig, ax = plt.subplots(1, 1)
    for tup in dataframes:
        assert type(tup[0]) is pl.DataFrame
        assert type(tup[1]) is str
        ax.plot(tup[0]["TimeStamp"][:], tup[0][tup[0].columns[1]][:], tup[1])
    plt.show()


def plot_comp():
    no_asr = pl.read_csv(f"filtered_exg_calib-30.0_window-0.05_t-120.0.csv")
    no_asr = no_asr.to_numpy().swapaxes(1, 0)
    asr_one = pl.read_csv(f"asr_cleaned-data_calib-30.0_window-0.05_t-120.0.csv")
    asr_one = asr_one.to_numpy().swapaxes(1, 0)
    asr_two = pl.read_csv(f"asr_cleaned-data_calib-30.0_window-1.0_t-120.0.csv")
    asr_two = asr_two.to_numpy().swapaxes(1, 0)
    fig, ax = plt.subplots(1, 1)
    #ax.plot(asr_one[0][:1000], asr_one[1][:1000], 'r')
    #ax.plot(asr_two[0][:1000], asr_two[1][:1000], 'b')
    ax.plot(no_asr[0][:], no_asr[1][:], 'g')
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


def clean_data_from_file():
    dev = explorepy.Explore()
    dev.connect("Explore_DABC")
    dev.stream_processor.add_filter(cutoff_freq=50., filter_type="notch")
    dev.stream_processor.add_filter(cutoff_freq=(1., 30.), filter_type="bandpass")

    time.sleep(5.0)

    dev.calibrate_asr(calib_time)
    dev.stream_processor.subscribe(on_filtered_received, topic=TOPICS.filtered_ExG)
    dev.stream_processor.subscribe(on_asr_received, topic=TOPICS.asr_ExG)

    time.sleep(calib_time + 5.)

    dev.stream_processor.asr_processor.calibration_data_input = as_np
    dev.stream_processor.asr_processor.set_state_from_calibration_data(calib_data=as_np)

    time.sleep(1.)

    # Uncomment to write calibration data to file after it has been "recorded"

    # calib_data = dev.stream_processor.asr_processor.calibration_data_input
    # calib_df = pl.DataFrame(calib_data.swapaxes(1, 0))
    # calib_df.write_csv(calib_file)

    dev.start_asr(window=asr_window)

    time.sleep(t)

    ts_buffer_np = np.array(ts_buffer)
    data_buffer_np = np.array(data_buffer)

    ts_buffer_no_asr_np = np.array(ts_buffer_no_asr)
    data_buffer_no_asr_np = np.array(data_buffer_no_asr)

    n = ["TimeStamp"]
    n.extend([f"ch{i + 1}" for i in range(n_ch)])
    ret = np.vstack((ts_buffer_np, data_buffer_np))
    df = pl.DataFrame(ret.swapaxes(1, 0), schema=n)
    df.write_csv(f"asr_cleaned-data_calib-{calib_time}_window-{asr_window}_t-{t}.csv")  # cleaned from filtered

    ret_two = np.vstack((ts_buffer_no_asr_np, data_buffer_no_asr_np))
    df_no_asr = pl.DataFrame(ret_two.swapaxes(1, 0), schema=n)
    df_no_asr.write_csv(f"filtered_exg_calib-{calib_time}_window-{asr_window}_t-{t}.csv")  # uncleaned but filtered


if __name__ == '__main__':
    compare_cleaned_with_uncleaned()
    # clean_data_from_file()
