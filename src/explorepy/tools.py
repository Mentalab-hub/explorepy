# -*- coding: utf-8 -*-
from explorepy.parser import Parser
import os.path
import csv
import bluetooth
import numpy as np
from explorepy.filters import Filter
from scipy import signal
import pyedflib
import datetime


def bt_scan():
    """"Scan for bluetooth devices
    Scans for available explore devices.
    Prints out MAC address and name of each found device

    Args:

    Returns:

    """
    print("Searching for nearby devices...")
    nearby_devices = bluetooth.discover_devices(lookup_names=True)
    explore_devices = []
    for address, name in nearby_devices:
        if "Explore" in name:
            print("Device found: %s - %s" % (name, address))
            explore_devices.append((address, name))

    if not nearby_devices:
        print("No Devices found")

    return explore_devices


def bin2csv(bin_file, do_overwrite=False, out_dir=None):
    """Binary to CSV file converter.
    This function converts the given binary file to ExG and ORN csv files.

    Args:
        bin_file (str): Binary file full address
        out_dir (str): Output directory (if None, uses the same directory as binary file)
        do_overwrite (bool): Overwrite if files exist already

    """
    head_path, full_filename = os.path.split(bin_file)
    filename, extension = os.path.splitext(full_filename)
    assert os.path.isfile(bin_file), "Error: File does not exist!"
    assert extension == '.BIN', "File type error! File extension must be BIN."
    if out_dir is None:
        out_dir = head_path + '/'

    exg_out_file = out_dir + filename + '_exg'
    orn_out_file = out_dir + filename + '_orn'
    marker_out_file = out_dir + filename + '_marker'

    exg_ch = ['TimeStamp', 'ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8']
    exg_unit = ['s', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V']
    exg_recorder = FileRecorder(file_name=exg_out_file, ch_label=exg_ch, fs=250, ch_unit=exg_unit,
                                file_type='csv', do_overwrite=do_overwrite)

    orn_ch = ['TimeStamp', 'ax', 'ay', 'az', 'gx', 'gy', 'gz', 'mx', 'my', 'mz']
    orn_unit = ['s', 'mg', 'mg', 'mg', 'mdps', 'mdps', 'mdps', 'mgauss', 'mgauss', 'mgauss']
    orn_recorder = FileRecorder(file_name=orn_out_file, ch_label=orn_ch, fs=20,
                                ch_unit=orn_unit, file_type='csv', do_overwrite=do_overwrite)

    marker_ch = ['TimeStamp', 'Code']
    marker_unit = ['s', '-']
    marker_recorder = FileRecorder(file_name=marker_out_file, ch_label=marker_ch, fs=0, ch_unit=marker_unit,
                                   file_type='csv', do_overwrite=do_overwrite)
    with open(bin_file, "rb") as f_bin:
        parser = Parser(fid=f_bin)

        print("Converting...")
        while True:
            try:
                parser.parse_packet(mode='record', recorders=(exg_recorder, orn_recorder, marker_recorder))
            except ValueError:
                print("Binary file ended! Conversion finished!")
                break


def bin2edf(bin_file, do_overwrite=False, out_dir=None):
    """Binary to EDF file converter.
    This function converts the given binary file to ExG and ORN csv files.

    Args:
        bin_file (str): Binary file full address
        out_dir (str): Output directory (if None, uses the same directory as binary file)
        do_overwrite (bool): Overwrite if files exist already
    """
    head_path, full_filename = os.path.split(bin_file)
    filename, extension = os.path.splitext(full_filename)
    assert os.path.isfile(bin_file), "Error: File does not exist!"
    assert extension == '.BIN', "File type error! File extension must be BIN."
    if out_dir is None:
        out_dir = head_path + '/'

    exg_out_file = out_dir + filename + '_exg'
    orn_out_file = out_dir + filename + '_orn'

    with open(bin_file, "rb") as f_bin:
        parser = Parser(fid=f_bin)
        exg_ch = ['TimeStamp', 'ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8'][0:parser.n_chan + 1]
        exg_unit = ['s', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V'][0:parser.n_chan + 1]
        exg_max = [86400, .4, .4, .4, .4, .4, .4, .4, .4][0:parser.n_chan + 1]
        exg_min = [0, -.4, -.4, -.4, -.4, -.4, -.4, -.4, -.4][0:parser.n_chan + 1]
        exg_recorder = FileRecorder(file_name=exg_out_file, ch_label=exg_ch, fs=parser.fs, ch_unit=exg_unit,
                                    file_type='edf', do_overwrite=do_overwrite, ch_min=exg_min, ch_max=exg_max)

        orn_ch = ['TimeStamp', 'ax', 'ay', 'az', 'gx', 'gy', 'gz', 'mx', 'my', 'mz']
        orn_unit = ['s', 'mg', 'mg', 'mg', 'mdps', 'mdps', 'mdps', 'mgauss', 'mgauss', 'mgauss']
        orn_max = [86400, 2000, 2000, 2000, 250000, 250000, 250000, 50000, 50000, 50000]
        orn_min = [0, -2000, -2000, -2000, -250000, -250000, -250000, -50000, -50000, -50000]
        orn_recorder = FileRecorder(file_name=orn_out_file, ch_label=orn_ch, fs=20, ch_unit=orn_unit, file_type='edf',
                                    do_overwrite=do_overwrite, ch_max=orn_max, ch_min=orn_min)
        print("Converting...")
        while True:
            try:
                parser.parse_packet(mode='record', recorders=(exg_recorder, orn_recorder, exg_recorder))
            except ValueError:
                print("Binary file ended! Conversion finished!")
                break


class HeartRateEstimator:
    def __init__(self, fs=250, smoothing_win=20):
        """Real-time heart Rate Estimator class This class provides the tools for heart rate estimation. It basically detects
        R-peaks in ECG signal using the method explained in Hamilton 2002 [2].

        Args:
            fs (int): Sampling frequency
            smoothing_win (int): Length of smoothing window

        References:
            [1] Hamilton, P. S. (2002). Open source ECG analysis software documentation. Computers in cardiology, 2002.

            [2] Hamilton, P. S., & Tompkins, W. J. (1986). Quantitative investigation of QRS detection rules using the
            MIT/BIH arrhythmia database. IEEE transactions on biomedical engineering.
        """
        self.fs = fs
        self.threshold = .35  # Generally between 0.3125 and 0.475
        self.ns200ms = int(self.fs * .2)
        self.r_peaks_buffer = [(0., 0.)]
        self.noise_peaks_buffer = [(0., 0., 0.)]
        self.prev_samples = np.zeros(smoothing_win)
        self.prev_diff_samples = np.zeros(smoothing_win)
        self.prev_times = np.zeros(smoothing_win)
        self.prev_max_slope = 0

        self.bp_filter = Filter(l_freq=1, h_freq=30, order=3, sampling_freq=fs)
        self.hamming_window = signal.windows.hamming(smoothing_win, sym=True)
        self.hamming_window /= self.hamming_window.sum()

    @property
    def average_noise_peak(self):
        return np.mean([item[0] for item in self.noise_peaks_buffer])

    @property
    def average_qrs_peak(self):
        return np.mean([item[0] for item in self.r_peaks_buffer])

    @property
    def decision_threshold(self):
        return self.average_noise_peak + self.threshold * (self.average_qrs_peak - self.average_noise_peak)

    @property
    def average_rr_interval(self):
        if len(self.r_peaks_buffer) < 7:
            return 1.
        return np.mean(np.diff([item[1] for item in self.r_peaks_buffer]))

    @property
    def heart_rate(self):
        if len(self.r_peaks_buffer) < 7:
            print('Few peaks to get heart rate!')
            return 'NA'
        else:
            r_times = [item[1] for item in self.r_peaks_buffer]
            rr_intervals = np.diff(r_times, 1)
            if True in (rr_intervals > 3.):
                print('Missing peaks!')
                return 'NA'
            else:
                estimated_heart_rate = int(1. / np.mean(rr_intervals) * 60)
                if estimated_heart_rate > 140 or estimated_heart_rate < 40:
                    print('Estimated heart rate <40 or >140!')
                    estimated_heart_rate = 'NA'
                return estimated_heart_rate

    def push_r_peak(self, val, time):
        self.r_peaks_buffer.append((val, time))
        if len(self.r_peaks_buffer) > 8:
            self.r_peaks_buffer.pop(0)

    def push_noise_peak(self, val, peak_idx, peak_time):
        self.noise_peaks_buffer.append((val, peak_idx, peak_time))
        if len(self.noise_peaks_buffer) > 8:
            self.noise_peaks_buffer.pop(0)

    def estimate(self, ecg_sig, time_vector):
        """ Detection of R-peaks

        Args:
            time_vector (np.array): One-dimensional time vector
            ecg_sig (np.array): One-dimensional ECG signal

        Returns:
            List of detected peaks indices
        """
        assert len(ecg_sig.shape) == 1, "Signal must be a vector"

        # Preprocessing
        ecg_filtered = self.bp_filter.apply_bp_filter(ecg_sig).squeeze()
        ecg_sig = np.concatenate((self.prev_samples, ecg_sig))
        sig_diff = np.diff(ecg_filtered, 1)
        sig_abs_diff = np.abs(sig_diff)
        sig_smoothed = signal.convolve(np.concatenate((self.prev_diff_samples, sig_abs_diff)),
                                       self.hamming_window, mode='same', method='auto')[:len(ecg_filtered)]
        time_vector = np.concatenate((self.prev_times, time_vector))
        self.prev_samples = ecg_sig[-len(self.hamming_window):]
        self.prev_diff_samples = sig_abs_diff[-len(self.hamming_window):]
        self.prev_times = time_vector[-len(self.hamming_window):]
        peaks_idx_list, _ = signal.find_peaks(sig_smoothed)
        peaks_val_list = sig_smoothed[peaks_idx_list]
        peaks_time_list = time_vector[peaks_idx_list]
        detected_peaks_idx = []
        detected_peaks_time = []
        detected_peaks_val = []

        # Decision rules by Hamilton 2002 [1]
        for peak_idx, peak_val, peak_time in zip(peaks_idx_list, peaks_val_list, peaks_time_list):
            # 1- Ignore all peaks that precede or follow larger peaks by less than 200 ms.
            peaks_in_lim = [a and b and c for a, b, c in
                            zip(((peak_idx - self.ns200ms) < peaks_idx_list),
                                ((peak_idx + self.ns200ms) > peaks_idx_list),
                                (peak_idx != peaks_idx_list)
                                )
                            ]

            if True in (peak_val < peaks_val_list[peaks_in_lim]):
                continue

            # 2- If a peak occurs, check to see whether the ECG signal contained both positive and negative slopes.
            # TODO: Find a better way of checking this.
            # if peak_idx == 0:
            #     continue
            # elif peak_idx < 10:
            #     n_sample = peak_idx
            # else:
            #     n_sample = 10
            # The current n_sample leads to missing some R-peaks as it may have wider/thinner width.
            # slopes = np.diff(ecg_sig[peak_idx-n_sample:peak_idx])
            # if slopes[0] * slopes[-1] >= 0:
            #     continue

            # check missing peak
            self.check_missing_peak(peak_time, peak_idx, detected_peaks_idx, ecg_sig, time_vector)

            # 3- If the peak occurred within 360 ms of a previous detection and had a maximum slope less than half the
            # maximum slope of the previous detection assume it is a T-wave
            if (peak_time - self.r_peaks_buffer[-1][1]) < .36:
                if peak_idx < 15:
                    st_idx = 0
                else:
                    st_idx = peak_idx - 15
                if (peak_idx + 15) > (len(ecg_sig) - 1):
                    end_idx = len(ecg_sig) - 1
                else:
                    end_idx = peak_idx + 15

                curr_max_slope = np.abs(np.diff(ecg_sig[st_idx:end_idx])).max()
                if curr_max_slope < (.5 * self.prev_max_slope):
                    continue

            # 4- If the peak is larger than the detection threshold call it a QRS complex, otherwise call it noise.
            if peak_idx < 25:
                st_idx = 0
            else:
                st_idx = peak_idx - 25
            pval = peak_val  # ecg_sig[st_idx:peak_idx].max()

            if pval > self.decision_threshold:
                temp_idx = st_idx + np.argmax(ecg_sig[st_idx:peak_idx + 1])
                temp_time = time_vector[temp_idx]

                detected_peaks_idx.append(temp_idx)
                detected_peaks_val.append(ecg_sig[st_idx:peak_idx + 1].max())
                detected_peaks_time.append(temp_time)
                self.push_r_peak(pval, temp_time)

                if peak_idx < 25:
                    st_idx = 0
                else:
                    st_idx = peak_idx - 25
                self.prev_max_slope = np.abs(np.diff(ecg_sig[st_idx:peak_idx + 25])).max()
            else:
                self.push_noise_peak(pval, peak_idx, peak_time)

            # TODO: Check lead inversion!

        # Check for two close peaks
        occurrence_time = [item[1] for item in self.r_peaks_buffer]
        close_idx = (np.diff(np.array(occurrence_time), 1) < .05)
        if (True in close_idx) and len(detected_peaks_idx) > 0:
            del detected_peaks_time[0]
            del detected_peaks_val[0]

        return detected_peaks_time, detected_peaks_val

    def check_missing_peak(self, peak_time, peak_idx, detected_peaks_idx, ecg_sig, time_vector):
        # 5- If an interval equal to 1.5 times the average R-to-R interval has elapsed since the most recent
        # detection, within that interval there was a peak that was larger than half the detection threshold and
        # the peak followed the preceding detection by at least 360 ms, classify that peak as a QRS complex.
        if (peak_time - self.r_peaks_buffer[-1][1]) > (1.4 * self.average_rr_interval):
            last_noise_val, last_noise_idx, last_noise_time = self.noise_peaks_buffer[-1]
            if last_noise_val > (.5 * self.decision_threshold):
                if (last_noise_time - self.r_peaks_buffer[-1][1]) > .36:
                    self.noise_peaks_buffer.pop(-1)
                    if peak_idx > last_noise_idx:
                        if last_noise_idx < 20:
                            st_idx = 0
                        else:
                            st_idx = last_noise_idx - 20
                        detected_peaks_idx.append(st_idx + np.argmax(ecg_sig[st_idx:peak_idx]))
                        self.push_r_peak(last_noise_val, time_vector[detected_peaks_idx[-1]])
                        if peak_idx < 25:
                            st_idx = 0
                        else:
                            st_idx = peak_idx - 25
                        self.prev_max_slope = np.abs(np.diff(ecg_sig[st_idx:peak_idx + 25])).max()
                    else:
                        # The peak is in the previous chunk
                        # TODO: return a negative index for it!
                        pass


class FileRecorder:
    """Explorepy file recorder class.

    This class can write ExG, orientation and environment data into (separated) EDF+ files. It can write data while
    streaming from Explore device. The incoming data will be stored in a buffer and after it reached fs samples, it
    writes the buffer in EDF file.

    Attributes:

    """

    def __init__(self, file_name, ch_label, fs, ch_unit, ch_min=None, ch_max=None,
                 device_name='Explore', file_type='edf', do_overwrite=False):
        """

        Args:
            file_name (str): File name
            ch_label (list): List of channel labels.
            fs (int): Sampling rate (must be identical for all channels)
            ch_unit (list): List of channels unit (e.g. 'V', 'mG', 's', etc.)
            ch_min (list): List of minimum value of each channel. Only needed in edf mode (can be None in csv mode)
            ch_max (list): List of maximum value of each channel. Only needed in edf mode (can be None in csv mode)
            device_name (str): Recording device name
            file_type (str): File type. current options: 'edf' and 'csv'.
            do_overwrite (bool): Overwrite file if a file with the same name exists already.
        """

        # Check invalid characters
        if set(r'<>{}[]~`*%').intersection(file_name):
            raise ValueError("Invalid character in file name")

        self._file_obj = None
        self._file_type = file_type
        self._ch_label = ch_label
        self._ch_unit = ch_unit
        self._ch_max = ch_max
        self._ch_min = ch_min
        self._n_chan = len(ch_label)
        self._device_name = device_name
        self._fs = int(fs)

        if file_type == 'edf':
            if (len(ch_unit) != len(ch_label)) or (len(ch_label) != len(ch_min)) or (len(ch_label) != len(ch_max)):
                raise ValueError('ch_label, ch_unit, ch_min and ch_max must have the same length!')
            self._file_name = file_name + '.edf'
            self._create_edf(do_overwrite=do_overwrite)
            self._init_edf_channels()
            self._data = np.zeros((self._n_chan, 0))
        elif file_type == 'csv':
            self._file_name = file_name + '.csv'
            self._create_csv(do_overwrite=do_overwrite)

    @property
    def fs(self):
        return self._fs

    def _create_edf(self, do_overwrite):
        if (not do_overwrite) and os.path.isfile(self._file_name):
            raise FileExistsError(self._file_name + ' already exists!')
        assert self._file_obj is None, "Usage Error: File object has been created already."
        self._file_obj = pyedflib.EdfWriter(self._file_name, self._n_chan, file_type=pyedflib.FILETYPE_BDFPLUS)

    def _create_csv(self, do_overwrite):
        if (not do_overwrite) and os.path.isfile(self._file_name):
            raise FileExistsError(self._file_name + ' already exists!')
        assert self._file_obj is None, "Usage Error: File object has been created already."
        self._file_obj = open(self._file_name, 'w', newline='\n')
        self._csv_obj = csv.writer(self._file_obj, delimiter=",")
        self._csv_obj.writerow(self._ch_label)

    def stop(self):
        """Stop recording"""
        assert self._file_obj is not None, "Usage Error: File object has not been created yet."
        if self._file_type == 'edf':
            if self._data.shape[1] > 0:
                self._file_obj.writeSamples(list(self._data))
            self._file_obj.close()
            self._file_obj = None
        elif self._file_type == 'csv':
            self._file_obj.close()

    def _init_edf_channels(self):
        self._file_obj.setEquipment(self._device_name)
        self._file_obj.setStartdatetime(datetime.datetime.now())

        ch_info_list = []
        for ch in zip(self._ch_label, self._ch_unit, self._ch_max, self._ch_min):
            ch_info_list.append({'label':        ch[0],
                                 'dimension':    ch[1],
                                 'sample_rate':  self._fs,
                                 'physical_max': ch[2],
                                 'physical_min': ch[3],
                                 'digital_max':  8388607,
                                 'digital_min':  -8388608,
                                 'prefilter':    '',
                                 'transducer':   ''
                                 })
        for i, ch_info in enumerate(ch_info_list):
            self._file_obj.setSignalHeader(i, ch_info)

    def write_data(self, data):
        """writes data to the file

        Notes:
            If file type is set to EDF, this function writes each 1 seconds of data. If the input is less than 1 second,
            it will be buffered in the memory and it will be written in the file when enough data is in the buffer.

        Args:
            data (np.array): Array of data to be written in the file with dimension of n_chan x n_sample

        """
        if self._file_type == 'edf':
            if data.shape[0] != self._n_chan:
                raise ValueError('Input first dimension must be {}'.format(self._n_chan))
            self._data = np.concatenate((self._data, data), axis=1)

            if self._data.shape[1] > self._fs:
                self._file_obj.writeSamples(list(self._data[:, :self._fs]))
                self._data = self._data[:, self._fs:]
        elif self._file_type == 'csv':
            self._csv_obj.writerows(data.T.tolist())

    def set_marker(self, data):
        """Writes a marker event in the file

        Args:
            data (np.array): Array of marker data with size 2x1 ([[timestamp],[code]])

        """
        if self._file_type == 'csv':
            self.write_data(data)
        elif self._file_type == 'edf':
            self._file_obj.writeAnnotation(data[0, 0], 0.001, str(int(data[1, 0])))


if __name__ == '__main__':
    file_name = 'test_rec'
    labels = ['timestamp', 'ch01', 'ch02', 'ch_03', 'ch04']
    units = ['V', 'V', 'V', 'V', 's']
    mins = [-1, -1, -1, -1, 0]
    maxs = [1, 1, 1, 1, 86400]
    recorder = FileRecorder(file_name=file_name, fs=250, ch_label=labels, file_type='csv',
                            ch_unit=units, ch_max=maxs, ch_min=mins, do_overwrite=True)

    for i in range(1002):
        chunk = np.random.normal(0, 1, (5, 33))
        recorder.write_data(chunk)
    recorder.stop()
