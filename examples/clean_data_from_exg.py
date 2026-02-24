# Note: This is a WIP
import glob
import os
import random
from typing import Union

import explorepy
from explorepy.stream_processor import TOPICS
from eegprep import clean_asr
import time
import numpy as np
import polars as pl

import matplotlib.pyplot as plt

CALIB_LENGTH_STR = "calib"
WINDOW_STR = "window"
REC_TIME_STR = "t"

EEGPREP_STR = "eegprep"
CALIBRATION_DATA_STR = "calibdata"
FILTERED_STR = "filtered"
ASR_CLEANED_STR = "asrcleaned"

BLOCK_SEPARATOR = "_"
VALUE_SEPARATOR = "-"

calib_time = 30.
asr_window = 0.05
t = 180.

now = time.time()

class DataRecorder:
    def __init__(self, n_channels):
        self.dev = None
        self.n_channels = n_channels
        self.ts_buffer = []
        self.data_buffer = [[] for _ in range(self.n_channels)]
        self.ts_buffer_no_asr = []
        self.data_buffer_no_asr = [[] for _ in range(self.n_channels)]

    def clear_buffers(self):
        self.ts_buffer = []
        self.data_buffer = [[] for _ in range(self.n_channels)]
        self.ts_buffer_no_asr = []
        self.data_buffer_no_asr = [[] for _ in range(self.n_channels)]

    def on_asr_received(self, packet):
        data = packet.get_data()
        ts = data[0]
        self.ts_buffer.extend(ts)
        for i in range(self.n_channels):
            self.data_buffer[i].extend(data[1][i, :])

    def on_filtered_received(self, packet):
        data_filtered = packet.get_data()
        ts_filtered = data_filtered[0]
        self.ts_buffer_no_asr.extend(ts_filtered)
        for i in range(self.n_channels):
            self.data_buffer_no_asr[i].extend(data_filtered[1][i, :])

    def set_up_explore_device(self, t_calib=30., dev_name="Explore_DABC", notch=50., bp=(1., 30.), record_raw_data=False, file=None):
        """Connects to a device, sets up filters and performs ASR calibration"""
        self.dev = explorepy.Explore()
        self.dev.connect(dev_name, file_path=file)

        time.sleep(4.0)

        self.dev.stream_processor.add_filter(cutoff_freq=notch, filter_type="notch")
        self.dev.stream_processor.add_filter(cutoff_freq=bp, filter_type="bandpass")
        f_name = self.dev.stream_processor.parser.stream_interface.file_name
        print(f"Working on {f_name}")

        time.sleep(5.0)

        self.dev.calibrate_asr(t_calib)
        if record_raw_data:
            self.dev.stream_processor.subscribe(self.on_filtered_received, topic=TOPICS.filtered_ExG)
        self.dev.stream_processor.subscribe(self.on_asr_received, topic=TOPICS.asr_ExG)

        time.sleep(t_calib + 5.)

        return self.dev

    @staticmethod
    def get_calibration_filename(t_calib):
        return (f"{CALIBRATION_DATA_STR}"
                f"{BLOCK_SEPARATOR}"
                f"{CALIB_LENGTH_STR}{VALUE_SEPARATOR}{t_calib}.csv")

    @staticmethod
    def get_clean_filename(t_calib, t_window, rec_length):
        return (f"{ASR_CLEANED_STR}"
                f"{BLOCK_SEPARATOR}"
                f"{CALIB_LENGTH_STR}{VALUE_SEPARATOR}{t_calib}"
                f"{BLOCK_SEPARATOR}"
                f"{WINDOW_STR}{VALUE_SEPARATOR}{t_window}"
                f"{BLOCK_SEPARATOR}"
                f"{REC_TIME_STR}{VALUE_SEPARATOR}{rec_length}.csv")

    @staticmethod
    def get_filtered_filename(rec_length):
        return (f"{FILTERED_STR}"
                f"{BLOCK_SEPARATOR}"
                f"{CALIB_LENGTH_STR}{VALUE_SEPARATOR}{rec_length}.csv")

    @staticmethod
    def get_eegprep_filename(t_calib):
        return (f"{EEGPREP_STR}"
                f"{BLOCK_SEPARATOR}"
                f"{CALIB_LENGTH_STR}{VALUE_SEPARATOR}{t_calib}.csv")

    def write_calibration_data(self, t_calib=30., calib_file=None, file=None, root_folder=None, overwrite=False):
        if calib_file is None:
            calib_file = self.get_calibration_filename(t_calib)

        if root_folder is not None:
            calib_file = os.path.join(root_folder, calib_file)

        self.dev = self.set_up_explore_device(t_calib=t_calib, file=file)

        time.sleep(1.)

        calib_data = self.dev.stream_processor.asr_processor.calibration_data_input
        calib_df = pl.DataFrame(calib_data.swapaxes(1, 0))
        if not os.path.exists(calib_file) or overwrite:
            print(f"Writing calibration data to {calib_file}")
            calib_df.write_csv(calib_file)
        self.dev.disconnect()
        time.sleep(1.)

        self.clear_buffers()

        return calib_file

    def clean_data_from_file(self, cutoff=5.0, t_calib=30., t_window=0.05, rec_length=180., calib_file=None, record_raw_data=False, file=None, root_folder=None, overwrite=False):
        self.dev = self.set_up_explore_device(t_calib=t_calib, record_raw_data=record_raw_data, file=file)

        if calib_file is not None:
            calib_file_df = pl.read_csv(calib_file)
            as_np = calib_file_df.to_numpy()
            as_np = as_np.swapaxes(1, 0)

            print(f"Setting calib data from {calib_file}")

            self.dev.stream_processor.asr_processor.calibration_data_input = as_np
            self.dev.stream_processor.asr_processor.set_state_from_calibration_data(calib_data=as_np)

        time.sleep(1.)

        self.dev.stream_processor.asr_processor.cutoff = cutoff

        self.dev.start_asr(window=t_window)

        time.sleep(rec_length)

        ts_buffer_np = np.array(self.ts_buffer)
        data_buffer_np = np.array(self.data_buffer)

        n = ["TimeStamp"]
        n.extend([f"ch{i + 1}" for i in range(self.n_channels)])
        ret = np.vstack((ts_buffer_np, data_buffer_np))
        df = pl.DataFrame(ret.swapaxes(1, 0), schema=n)
        f_name_cleaned = self.get_clean_filename(t_calib, t_window, rec_length)
        if root_folder is not None:
            f_name_cleaned = os.path.join(root_folder, f_name_cleaned)
        if not os.path.exists(f_name_cleaned) or overwrite:
            print(f"Writing cleaned data to {f_name_cleaned}")
            df.write_csv(f_name_cleaned)  # cleaned from filtered

        f_name_uncleaned = None

        if record_raw_data:
            ts_buffer_no_asr_np = np.array(self.ts_buffer_no_asr)
            data_buffer_no_asr_np = np.array(self.data_buffer_no_asr)

            ret_two = np.vstack((ts_buffer_no_asr_np, data_buffer_no_asr_np))
            df_no_asr = pl.DataFrame(ret_two.swapaxes(1, 0), schema=n)
            f_name_uncleaned = self.get_filtered_filename(rec_length)
            if root_folder is not None:
                f_name_uncleaned = os.path.join(root_folder, f_name_uncleaned)
            if not os.path.exists(f_name_uncleaned) or overwrite:
                print(f"Writing uncleaned data to {f_name_uncleaned}")
                df_no_asr.write_csv(f_name_uncleaned)  # uncleaned but filtered

        self.dev.disconnect()
        time.sleep(1.)
        self.dev = None

        self.clear_buffers()

        return f_name_cleaned, f_name_uncleaned


class DataComparator:
    def __init__(self):
        self.dataframes = []
        self.max_plot_rows = 4
        self.max_plot_cols = 2
        self.max_plot_indices = 1000

    def get_first_and_last_ts(self):
        all_first_ts = []
        all_last_ts = []
        for tup in self.dataframes:
            all_first_ts.append(tup[0]["TimeStamp"][0])
            all_last_ts.append(tup[0]["TimeStamp"][-1])
        first_ts = max(all_first_ts)
        last_ts = min(all_last_ts)
        return first_ts, last_ts

    def cull_dataframes(self):
        culled_dataframes = []
        first_ts, last_ts = self.get_first_and_last_ts()
        for tup in self.dataframes:
            df = tup[0]
            df = df.remove(pl.col("TimeStamp") < first_ts)
            df = df.remove(pl.col("TimeStamp") > last_ts)
            culled_dataframes.append((df, tup[1], tup[2]))
        self.dataframes = culled_dataframes

    def add_dataframe_from_file(self, file_path, colour, label):
        dataframe = pl.read_csv(file_path)
        self.add_dataframe(dataframe, colour, label)

    def add_dataframe(self, dataframe, colour, label):
        self.dataframes.append((dataframe, colour, label))

    def clear_dataframes(self):
        self.dataframes = []

    def plot_comp_from_dataframes(self, channels: list[int] = None, max_samples : Union[str, int]="default"):
        """Plots comparison of any dataframes' first channel from a list of tuples

        Args:
            dataframes: list of tuples with the first element in the tuple being a polars DataFrame and the second element
            in the tuple being a colour string to use for plotting with matplotlib, i.e. "b" for blue etc.
            channels: list of ints that defines which channels to plot
        """
        n_samples_to_plot = self.max_plot_indices
        if max_samples:
            if type(max_samples) == int:
                n_samples_to_plot = max(max_samples, 1)
            elif type(max_samples) == str:
                if max_samples in ["all", "full"]:
                    n_samples_to_plot = None
        if channels is None:
            print("No channels supplied for plotting, exiting...")
            return
        n_channels = len(channels)

        if n_channels > self.max_plot_cols * self.max_plot_rows:
            print(f"Too many channels to reasonably plot (got: {n_channels}, "
                  f"max: {self.max_plot_cols * self.max_plot_rows}), exiting...")

        plot_cols = 1
        for i in range(self.max_plot_cols):
            if n_channels > (i + 1) * self.max_plot_rows:
                plot_cols += 1
        plot_rows = min(n_channels, self.max_plot_rows)
        fig, axs = plt.subplots(plot_rows, plot_cols, layout="constrained")
        it = 0
        ax = None
        for idx_to_plot in channels:
            r = it % self.max_plot_rows
            c = it // self.max_plot_rows
            ax = axs[r, c]
            for tup in self.dataframes:
                assert type(tup[0]) is pl.DataFrame
                assert type(tup[1]) is str
                assert type(tup[2]) is str
                ax.plot(tup[0]["TimeStamp"][:n_samples_to_plot],
                        tup[0][tup[0].columns[idx_to_plot + 1]][:n_samples_to_plot],
                        tup[1], label=tup[2])
            ax.title.set_text(f"Channel {idx_to_plot + 1}")  # start label for channels at 1
            it += 1
        if ax is not None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
            fig.legend(legend_handles, legend_labels, loc='upper center')
        plt.show()


def add_dead_channels_to_dataframe(dataframe, count):
    n_channels = len(dataframe.columns) - 1
    drop_indices = random.sample(range(1, n_channels), count)  # from 1 as col[0] is "TimeStamp"

    for idx in drop_indices:
        column_name = dataframe.columns[idx]
        dataframe = dataframe.replace_column(idx, pl.Series(name=column_name,
                                                            values=np.full(dataframe.shape[0], -400000.05)))

    return dataframe, drop_indices


def call_clean_from_eegprep(raw_data_file, calib_file, sr, cutoff, n_chan, first_ts, last_ts):
    calib_file_df = pl.read_csv(calib_file)
    calib_as_np = calib_file_df.to_numpy()
    calib_as_np = calib_as_np.swapaxes(1, 0)

    raw_data = pl.read_csv(raw_data_file)
    culled_raw_data = raw_data.remove(pl.col("TimeStamp") < first_ts)
    culled_raw_data = culled_raw_data.remove(pl.col("TimeStamp") > last_ts)
    culled_raw_data = culled_raw_data.to_numpy()
    culled_raw_data = culled_raw_data.swapaxes(1, 0)

    culled_raw_ts = culled_raw_data[0, :]
    culled_raw_data = culled_raw_data[1:, :]

    as_dict = {"data": culled_raw_data, "srate": sr, "nbchan": n_chan}

    cleaned_dict = clean_asr(as_dict, cutoff=cutoff, ref_maxbadchannels=calib_as_np)
    cleaned_df = cleaned_dict["data"]
    cleaned_df = cleaned_df.swapaxes(1, 0)
    cleaned_df = pl.DataFrame(cleaned_df)
    s = pl.Series("TimeStamp", culled_raw_ts)
    cleaned_df.insert_column(0, s)

    return cleaned_df


def perform_matrix_test(number_channels, cutoff, t_calib_length: float, t_window_list: list, rec_length_list: list,
                        input_file=None, plot_idx=None, overwrite_calib=False, overwrite_processor_cleaned=False,
                        overwrite_eegprep_cleaned=False):
    data_writer = DataRecorder(number_channels)
    data_comparator = DataComparator()

    root_folder = None
    if input_file is not None:
        root_folder = os.path.splitext(os.path.split(input_file)[1])[0]
        if not os.path.exists(root_folder):
            os.mkdir(root_folder)

    uncleaned_file = None
    cleaned_files = []
    expected_calib_path = os.path.join(root_folder if root_folder is not None else ".",
                                        data_writer.get_calibration_filename(t_calib_length))
    if overwrite_calib or not os.path.exists(expected_calib_path):
        calib_file_path = data_writer.write_calibration_data(t_calib=t_calib_length,
                                                             file=input_file,
                                                             root_folder=root_folder,
                                                             overwrite=True)
    else:
        print("\n--- Skipping write_calibration_data ---\n")
        calib_file_path = expected_calib_path
        print(f"Calibration file path: {calib_file_path}")

    print(f"Running cleaning with calibration file: {calib_file_path}")

    uncleaned_plot_colour = "gray"
    eegprep_plot_colour = "red"
    asr_processor_plot_colours = ["green", "skyblue", "violet", "gold", "aquamarine", "mediumpurple"]

    it = 0
    for t_window in t_window_list:
        for rec_length in rec_length_list:
            expected_clean_path = os.path.join(root_folder if root_folder is not None else ".",
                                               data_writer.get_clean_filename(t_calib_length, t_window, rec_length))
            expected_unclean_path = os.path.join(root_folder if root_folder is not None else ".",
                                                 data_writer.get_filtered_filename(rec_length))
            if (overwrite_processor_cleaned
                or not os.path.exists(expected_clean_path)
                or not os.path.exists(expected_unclean_path)):
                print(f"\n--- Cleaning data with t_window={t_window}, rec_length: {rec_length}, cutoff: {cutoff}---\n")
                cleaned_path, uncleaned_path = data_writer.clean_data_from_file(cutoff=cutoff,
                                                                                t_calib=t_calib_length,
                                                                                t_window=t_window,
                                                                                rec_length=rec_length,
                                                                                calib_file=calib_file_path,
                                                                                record_raw_data=uncleaned_file is None,
                                                                                file=input_file,
                                                                                root_folder=root_folder,
                                                                                overwrite=True)
            else:
                print(f"\n--- Skipping clean_data_from_file for t_window {t_window} and rec_length {rec_length} ---\n")
                cleaned_path = expected_clean_path if os.path.exists(expected_clean_path) else None
                uncleaned_path = expected_unclean_path if os.path.exists(expected_unclean_path) else None
                print(f"Cleaned path: {cleaned_path}\nUncleaned path: {uncleaned_path}")
            if uncleaned_path is not None and uncleaned_file is None:
                uncleaned_file = (uncleaned_path, uncleaned_plot_colour, "Uncleaned (filtered)")
            cleaned_files.append((cleaned_path,
                                  asr_processor_plot_colours[it%len(asr_processor_plot_colours)],
                                  f"Cleaned, cutoff={cutoff}, t_window={t_window}, calib={t_calib_length}, rec_length={rec_length}"))
            it += 1

    for cleaned_file in cleaned_files:
        data_comparator.add_dataframe_from_file(*cleaned_file)
    data_comparator.add_dataframe_from_file(*uncleaned_file)
    data_comparator.cull_dataframes()
    first_ts, last_ts = data_comparator.get_first_and_last_ts()
    eegprep_cleaned_path = os.path.join(root_folder if root_folder is not None else ".",
                                        data_writer.get_eegprep_filename(t_calib_length))
    if overwrite_eegprep_cleaned or not os.path.exists(eegprep_cleaned_path):
        eegprep_cleaned = call_clean_from_eegprep(raw_data_file=uncleaned_file[0], calib_file=calib_file_path, sr=250,
                                                  cutoff=5.0, n_chan=32, first_ts=first_ts, last_ts=last_ts)
        eegprep_cleaned.write_csv(eegprep_cleaned_path)
    else:
        print(f"\n--- Skipping call_clean_from_eegprep, loading instead from {eegprep_cleaned_path} ---\n")
        eegprep_cleaned = pl.read_csv(eegprep_cleaned_path)
    eeg_prep_tup = (eegprep_cleaned, eegprep_plot_colour, f"Cleaned (eegprep), cutoff={cutoff}")
    cleaned_files.append(eeg_prep_tup)
    data_comparator.add_dataframe(*eeg_prep_tup)
    if plot_idx is None or type(plot_idx) is not list:
        print("plot_idx is not give or the wrong time, using fall back index list...")
        plot_idx = [0, 1, 2, 3, 4, 5, 6, 7,]
    data_comparator.plot_comp_from_dataframes(channels=plot_idx)

def plot_recordings_from_folder(folder, idx=(0, 1, 2, 3, 4, 5, 6, 7)):
    data_comparator = DataComparator()
    files = glob.glob(os.path.join(folder, "*.csv"))
    files_for_comparator = []

    asr_processor_plot_colours = ["green", "skyblue", "violet", "gold", "aquamarine", "mediumpurple"]

    it = 0
    for f_name in files:
        if CALIBRATION_DATA_STR in f_name:
            pass
        else:
            if EEGPREP_STR in f_name:
                colour = "red"
                desc = "Eegprep cleaned"
            elif FILTERED_STR in f_name:
                colour = "gray"
                desc = "Filtered, uncleaned"
            else:
                blocks = os.path.splitext(os.path.basename(f_name))[0].split("_")
                desc = ", ".join(blocks[1:])
                desc = f"ASR cleaned - {desc}"
                colour = asr_processor_plot_colours[it % len(asr_processor_plot_colours)]
            files_for_comparator.append((f_name, colour, desc))
            it += 1

    for cleaned_file in files_for_comparator:
        data_comparator.add_dataframe_from_file(*cleaned_file)
    data_comparator.cull_dataframes()
    data_comparator.plot_comp_from_dataframes(channels=idx, max_samples="all")


if __name__ == '__main__':
    n_ch = 32
    cutoff = 5.0

    t_windows_to_test = [0.05, 0.1, 1.0, 5.0, 10.0]
    t_calib_to_test = 30.
    rec_length_to_test = [180.]  # calib_t + rec_t should be lower than length of test file


    # replace with valid path to test recording (csv only)
    test_file_name = "32channel_semidry_artefacts_ExG"  # t = 243.996
    test_file_root = "/Users/sonjastefani/Documents/dev/explore-desktop/test-data"

    input_file = os.path.join(test_file_root, f"{test_file_name}.csv")

    perform_matrix_test(n_ch, cutoff, t_calib_to_test, t_windows_to_test, rec_length_to_test, input_file=input_file)

    as_pl_df = pl.read_csv(input_file)
    as_pl_df_with_dead_channel, dropped_channels = add_dead_channels_to_dataframe(as_pl_df, 2)
    print(f"Dropped channels: {dropped_channels}")
    dropped_channels = [e-1 for e in dropped_channels]
    to_plot = np.array([dropped_channels[0]-1, dropped_channels[0], dropped_channels[0]+1,
                        dropped_channels[1]-1, dropped_channels[1], dropped_channels[1]+1])
    to_plot = to_plot[to_plot >= 0]
    to_plot = to_plot[to_plot < n_ch]
    to_plot = np.unique(to_plot)
    print(f"Indices to plot: {to_plot}")

    input_file_corrupted = os.path.join(test_file_root, f"{test_file_name}_ch-{"-".join(map(str, dropped_channels))}_corrupted.csv")

    as_pl_df_with_dead_channel.write_csv(input_file_corrupted)

    perform_matrix_test(n_ch, cutoff, t_calib_to_test, t_windows_to_test, rec_length_to_test, input_file=input_file_corrupted, plot_idx=list(to_plot))
