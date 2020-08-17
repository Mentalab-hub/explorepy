# -*- coding: utf-8 -*-
"""Some useful tools such as file recorder, heart rate estimation, etc. used in explorepy"""
import datetime
import os.path
import csv
import copy
import numpy as np
from scipy import signal
import pyedflib
from pylsl import StreamInfo, StreamOutlet
import configparser
from appdirs import user_cache_dir, user_config_dir

import explorepy
from explorepy.filters import ExGFilter

EXG_CHANNELS = ['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8']
EXG_UNITS = ['uV' for ch in EXG_CHANNELS]
ORN_CHANNELS = ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'mx', 'my', 'mz']
ORN_UNITS = ['mg', 'mg', 'mg', 'mdps', 'mdps', 'mdps', 'mgauss', 'mgauss', 'mgauss']


def bt_scan():
    """"Scan for bluetooth devices
    Scans for available explore devices.
    Prints out MAC address and name of each found device

    Args:

    Returns:

    """
    print("Searching for nearby devices...")
    explore_devices = []
    if explorepy._bt_interface == 'sdk':
        device_manager = explorepy.exploresdk.ExploreSDK_Create()
        nearby_devices = device_manager.PerformDeviceSearch()
        for bt_device in nearby_devices:
            if "Explore" in bt_device.name:
                print("Device found: %s - %s" % (bt_device.name, bt_device.address))
                explore_devices.append((bt_device.name, bt_device.address))
    else:
        import bluetooth
        nearby_devices = bluetooth.discover_devices(lookup_names=True)
        for address, name in nearby_devices:
            if "Explore" in name:
                print("Device found: %s - %s" % (name, address))
                explore_devices.append((address, name))

    if not nearby_devices:
        print("No Devices found")

    return explore_devices


def create_exg_recorder(filename, file_type, adc_mask, fs, do_overwrite):
    exg_ch = ['TimeStamp'] + EXG_CHANNELS
    exg_ch = [exg_ch[0]] + [exg_ch[i+1] for i, flag in enumerate(reversed(adc_mask)) if flag == 1]
    exg_unit = ['s'] + EXG_UNITS
    exg_unit = [exg_unit[0]] + [exg_unit[i + 1] for i, flag in enumerate(reversed(adc_mask)) if flag == 1]
    exg_max = [86400.] + [4e5 for i in range(8)]
    exg_max = [exg_max[0]] + [exg_max[i + 1] for i, flag in enumerate(reversed(adc_mask)) if flag == 1]
    exg_min = [0.] + [-4e5 for i in range(8)]
    exg_min = [exg_min[0]] + [exg_min[i + 1] for i, flag in enumerate(reversed(adc_mask)) if flag == 1]
    return FileRecorder(filename=filename, ch_label=exg_ch, fs=fs, ch_unit=exg_unit,
                        file_type=file_type, do_overwrite=do_overwrite, ch_min=exg_min, ch_max=exg_max)


def create_orn_recorder(filename, file_type, do_overwrite):
    orn_ch = ['TimeStamp'] + ORN_CHANNELS
    orn_unit = ['s'] + ORN_UNITS
    orn_max = [86400, 2000, 2000, 2000, 250000, 250000, 250000, 50000, 50000, 50000]
    orn_min = [0, -2000, -2000, -2000, -250000, -250000, -250000, -50000, -50000, -50000]
    return FileRecorder(filename=filename, ch_label=orn_ch, fs=20, ch_unit=orn_unit, file_type=file_type,
                        do_overwrite=do_overwrite, ch_max=orn_max, ch_min=orn_min)


def create_marker_recorder(filename, do_overwrite):
    marker_ch = ['TimeStamp', 'Code']
    marker_unit = ['s', '-']
    return FileRecorder(filename=filename, ch_label=marker_ch, fs=0, ch_unit=marker_unit,
                        file_type='csv', do_overwrite=do_overwrite)


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

        self.bp_filter = ExGFilter(cutoff_freq=(1, 30), filter_type='bandpass', s_rate=fs, n_chan=1, order=3)
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
            print('Few peaks to get heart rate! Noisy signal!')
            return 'NA'
        else:
            r_times = [item[1] for item in self.r_peaks_buffer]
            rr_intervals = np.diff(r_times, 1)
            if True in (rr_intervals > 3.):
                print('Missing peaks! Noisy signal!')
                return 'NA'
            else:
                estimated_heart_rate = int(1. / np.mean(rr_intervals) * 60)
                if estimated_heart_rate > 140 or estimated_heart_rate < 40:
                    print('Estimated heart rate <40 or >140!')
                    estimated_heart_rate = 'NA'
                return estimated_heart_rate

    def _push_r_peak(self, val, time):
        self.r_peaks_buffer.append((val, time))
        if len(self.r_peaks_buffer) > 8:
            self.r_peaks_buffer.pop(0)

    def _push_noise_peak(self, val, peak_idx, peak_time):
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
        ecg_filtered = self.bp_filter.apply(ecg_sig).squeeze()
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
                self._push_r_peak(pval, temp_time)

                if peak_idx < 25:
                    st_idx = 0
                else:
                    st_idx = peak_idx - 25
                self.prev_max_slope = np.abs(np.diff(ecg_sig[st_idx:peak_idx + 25])).max()
            else:
                self._push_noise_peak(pval, peak_idx, peak_time)

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
                        self._push_r_peak(last_noise_val, time_vector[detected_peaks_idx[-1]])
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

    def __init__(self, filename, ch_label, fs, ch_unit, ch_min=None, ch_max=None,
                 device_name='Explore', file_type='edf', do_overwrite=False):
        """

        Args:
            filename (str): File name
            ch_label (list): List of channel labels.
            fs (int): Sampling rate (must be identical for all channels)
            ch_unit (list): List of channels unit (e.g. 'uV', 'mG', 's', etc.)
            ch_min (list): List of minimum value of each channel. Only needed in edf mode (can be None in csv mode)
            ch_max (list): List of maximum value of each channel. Only needed in edf mode (can be None in csv mode)
            device_name (str): Recording device name
            file_type (str): File type. current options: 'edf' and 'csv'.
            do_overwrite (bool): Overwrite file if a file with the same name exists already.
        """

        # Check invalid characters
        if set(r'<>{}[]~`*%').intersection(filename):
            raise ValueError("Invalid character in file name")

        self._file_obj = None
        self.file_type = file_type
        self._ch_label = ch_label
        self._ch_unit = ch_unit
        self._ch_max = ch_max
        self._ch_min = ch_min
        self._n_chan = len(ch_label)
        self._device_name = device_name
        self._fs = int(fs)
        self._rectime_offset = None

        if file_type == 'edf':
            if (len(ch_unit) != len(ch_label)) or (len(ch_label) != len(ch_min)) or (len(ch_label) != len(ch_max)):
                raise ValueError('ch_label, ch_unit, ch_min and ch_max must have the same length!')
            self._file_name = filename + '.edf'
            self._create_edf(do_overwrite=do_overwrite)
            self._init_edf_channels()
            self._data = np.zeros((self._n_chan, 0))
        elif file_type == 'csv':
            self._file_name = filename + '.csv'
            self._create_csv(do_overwrite=do_overwrite)

    @property
    def fs(self):
        """Sampling frequency"""
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
        if self.file_type == 'edf':
            if self._data.shape[1] > 0:
                self._file_obj.writeSamples(list(self._data))
            self._file_obj.close()
            self._file_obj = None
        elif self.file_type == 'csv':
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

    def write_data(self, packet):
        """writes data to the file

        Notes:
            If file type is set to EDF, this function writes each 1 seconds of data. If the input is less than 1 second,
            it will be buffered in the memory and it will be written in the file when enough data is in the buffer.

        Args:
            packet (explorepy.packet.Packet): ExG or Orientation packet

        """
        time_vector, signal = packet.get_data(self._fs)

        if len(time_vector) == 1:
            data = np.array(time_vector + signal)[:, np.newaxis]
            if self._rectime_offset is None:
                self._rectime_offset = time_vector
        else:
            if self._rectime_offset is None:
                self._rectime_offset = time_vector[0]
            data = np.concatenate((np.array(time_vector)[:, np.newaxis].T, np.array(signal)), axis=0)
        data = np.round(data, 4)

        if self.file_type == 'edf':
            if data.shape[0] != self._n_chan:
                raise ValueError('Input first dimension must be {}'.format(self._n_chan))
            self._data = np.concatenate((self._data, data), axis=1)

            if self._data.shape[1] > self._fs:
                self._file_obj.writeSamples(list(self._data[:, :self._fs]))
                self._data = self._data[:, self._fs:]
        elif self.file_type == 'csv':
            self._csv_obj.writerows(data.T.tolist())
            self._file_obj.flush()

    def set_marker(self, packet):
        """Writes a marker event in the file

        Args:
            packet (explorepy.packet.EventMarker): Event marker packet

        """
        if self.file_type == 'csv':
            self.write_data(packet=packet)
        elif self.file_type == 'edf':
            timestamp, code = packet.get_data()
            if self._rectime_offset is None:
                self._rectime_offset = timestamp
            timestamp = timestamp-self._rectime_offset
            self._file_obj.writeAnnotation(timestamp[0], 0.001, str(int(code[0])))


class LslServer:
    """Class for LabStreamingLayer integration"""
    def __init__(self, device_info):
        n_chan = device_info['adc_mask'].count(1)
        self.exg_fs = device_info['sampling_rate']
        orn_fs = 20

        info_exg = StreamInfo(device_info["device_name"]+"_ExG", 'ExG', n_chan, self.exg_fs, 'float32', 'ExG')

        info_exg.desc().append_child_value("manufacturer", "Mentalab")
        channels = info_exg.desc().append_child("channels")
        for i, mask in enumerate(device_info['adc_mask']):
            if mask == 1:
                channels.append_child("channel")\
                    .append_child_value("name", EXG_CHANNELS[i])\
                    .append_child_value("unit", EXG_UNITS[i])\
                    .append_child_value("type", "ExG")

        info_orn = StreamInfo(device_info["device_name"]+"_ORN", 'Orientation', 9, orn_fs, 'float32', 'ORN')
        info_orn.desc().append_child_value("manufacturer", "Mentalab")
        channels = info_exg.desc().append_child("channels")
        for chan, unit in zip(ORN_CHANNELS, ORN_UNITS):
            channels.append_child("channel") \
                .append_child_value("name", chan) \
                .append_child_value("unit", unit) \
                .append_child_value("type", "ORN")

        info_marker = StreamInfo(device_info["device_name"]+"_Marker", 'Markers', 1, 0, 'int32', 'Marker')

        self.orn_outlet = StreamOutlet(info_orn)
        self.exg_outlet = StreamOutlet(info_exg)
        self.marker_outlet = StreamOutlet(info_marker)

    def push_exg(self, packet):
        """Push data to ExG outlet

        Args:
            packet (explorepy.packet.EEG): ExG packet
        """
        _, exg_data = packet.get_data(self.exg_fs)
        for sample in exg_data.T:
            self.exg_outlet.push_sample(sample.tolist())

    def push_orn(self, packet):
        """Push data to orientation outlet

        Args:
            packet (explorepy.packet.Orientation): Orientation packet
        """
        _, orn_data = packet.get_data()
        self.orn_outlet.push_sample(orn_data)

    def push_marker(self, packet):
        """Push data to marker outlet

        Args:
            packet (explorepy.packet.EventMarker): Event marker packet
        """
        _, code = packet.get_data()
        self.marker_outlet.push_sample(code)


class ImpedanceMeasurement:
    """Impedance measurement class"""
    def __init__(self, device_info, calib_param, notch_freq):
        """
        Args:
            device_info (dict): Device information dictionary
            calib_param (dict): Calibration parameters dictionary
            notch_freq (int): Line frequency (for notch filter)
        """
        self._device_info = device_info
        self._calib_param = calib_param
        self._filters = {}
        self._notch_freq = notch_freq
        self._add_filters()

    def _add_filters(self):
        bp_freq = self._device_info['sampling_rate'] / 4 - 1.5, \
                  self._device_info['sampling_rate'] / 4 + 1.5
        noise_freq = self._device_info['sampling_rate'] / 4 + 2.5, \
                     self._device_info['sampling_rate'] / 4 + 5.5

        self._filters['notch'] = ExGFilter(cutoff_freq=self._notch_freq,
                                           filter_type='notch',
                                           s_rate=self._device_info['sampling_rate'],
                                           n_chan=self._device_info['adc_mask'].count(1))

        self._filters['demodulation'] = ExGFilter(cutoff_freq=bp_freq,
                                                  filter_type='bandpass',
                                                  s_rate=self._device_info['sampling_rate'],
                                                  n_chan=self._device_info['adc_mask'].count(1))

        self._filters['base_noise'] = ExGFilter(cutoff_freq=noise_freq,
                                                filter_type='bandpass',
                                                s_rate=self._device_info['sampling_rate'],
                                                n_chan=self._device_info['adc_mask'].count(1))

    def measure_imp(self, packet):
        """Compute electrode impedances

        Args:
            self:
            packet:

        Returns:
            packet:
        """
        temp_packet = self._filters['notch'].apply(input_data=packet, in_place=False)
        self._calib_param['noise_level'] = self._filters['base_noise'].\
            apply(input_data=temp_packet, in_place=False).get_ptp()
        self._filters['demodulation'].apply(input_data=temp_packet, in_place=True).calculate_impedance(self._calib_param)
        return temp_packet


class PhysicalOrientation:
    """
    Movement sensors modules
    """
    def __init__(self):
        self.ED_prv = None
        self.theta = 0.
        self.axis = np.array([0, 0, -1])
        self.matrix = np.identity(3)
        self.init_set = None
        self.calibre_set = None
        self.status = "NOT READY"

    def calculate(self, packet):
        packet = copy.deepcopy(packet)
        if self.init_set:
            self._map(packet)
        else:
            self._get_rest_orn(packet)
        return packet

    def _get_rest_orn(self, packet):
        D = packet.acc / (np.dot(packet.acc, packet.acc) ** 0.5)
        # [kx, ky, kz, mx_offset, my_offset, mz_offset] = self.calibre_set
        packet.mag[0] = self.calibre_set[0] * (packet.mag[0] - self.calibre_set[3])
        packet.mag[1] = self.calibre_set[1] * (packet.mag[1] - self.calibre_set[4])
        packet.mag[2] = self.calibre_set[2] * (packet.mag[2] - self.calibre_set[5])
        E = -1 * np.cross(D, packet.mag)
        E = E / (np.dot(E, E) ** 0.5)
        # here you can find an estimation of actual north from packet.mag, it is perpendicular to D and still
        # co-planar with D and mag, somehow reducing error
        N = -1 * np.cross(E, D)
        N = N / (np.dot(N, N) ** 0.5)
        T_init = np.column_stack((E, N, D))
        N_init = np.matmul(np.transpose(T_init), N)
        E_init = np.matmul(np.transpose(T_init), E)
        D_init = np.matmul(np.transpose(T_init), D)
        self.init_set = [T_init, N_init, E_init, D_init]
        self.ED_prv = [E, D]

    def read_calibre_data(self, device_name):
        config = configparser.ConfigParser()
        calibre_file = user_config_dir(appname="explorepy", appauthor="mentalab")+ "/conf.ini"
        if os.path.isfile(calibre_file) :
            config.read(calibre_file)
            try:
                calibre_coef = config[device_name]
                self.calibre_set = np.asarray([float(calibre_coef['kx']), float(calibre_coef['ky']),
                                               float(calibre_coef['kz']), float(calibre_coef['mx']),
                                               float(calibre_coef['my']), float(calibre_coef['mz'])])
                return True
            except KeyError:
                return False
        else:
            return False

    def _map(self, packet):
        acc = packet.acc
        acc = acc / (np.dot(acc, acc) ** 0.5)
        gyro = packet.gyro * 1.745329e-5  # radian per second
        packet.mag[0] = self.calibre_set[0] * (packet.mag[0] - self.calibre_set[3])
        packet.mag[1] = self.calibre_set[1] * (packet.mag[1] - self.calibre_set[4])
        packet.mag[2] = self.calibre_set[2] * (packet.mag[2] - self.calibre_set[5])
        mag = packet.mag
        D = acc
        dD = D - self.ED_prv[1]
        da = np.cross(self.ED_prv[1], dD)
        E = -1 * np.cross(D, mag)
        E = E / (np.dot(E, E) ** 0.5)
        dE = E - self.ED_prv[0]
        dm = np.cross(self.ED_prv[0], dE)
        dg = 0.05 * gyro
        dth = -0.95 * dg + 0.025 * da + 0.025 * dm
        D = self.ED_prv[1] + np.cross(dth, self.ED_prv[1])
        D = D / (np.dot(D, D) ** 0.5)
        Err = np.dot(D, E)
        D_tmp = D - 0.5 * Err * E
        E_tmp = E - 0.5 * Err * D
        D = D_tmp / (np.dot(D_tmp, D_tmp) ** 0.5)
        E = E_tmp / (np.dot(E_tmp, E_tmp) ** 0.5)
        N = -1 * np.cross(E, D)
        N = N / (np.dot(N, N) ** 0.5)
        '''
        If you comment this block it will give you the absolute orientation based on {East,North,Up} coordinate system.
        If you keep this block of code it will give you the relative orientation based on itial state of the device. so
        It is important to keep the device steady, so that the device can capture the initial direction properly.
        '''
        ##########################
        T = np.zeros((3, 3))
        [T_init, N_init, E_init, D_init] = self.init_set
        T = np.column_stack((E, N, D))
        T_test = np.matmul(T, T_init.transpose())
        N = np.matmul(T_test.transpose(), N_init)
        E = np.matmul(T_test.transpose(), E_init)
        D = np.matmul(T_test.transpose(), D_init)
        ##########################
        matrix = np.identity(3)
        matrix = np.column_stack((E, N, D))
        N = N / (np.dot(N, N) ** 0.5)
        E = E / (np.dot(E, E) ** 0.5)
        D = D / (np.dot(D, D) ** 0.5)
        self.ED_prv = [E, D]
        self.matrix = self.matrix * 0.9 + 0.1 * matrix
        [theta, rot_axis] = packet.compute_angle(matrix=self.matrix)
        self.theta = self.theta * 0.9 + 0.1 * theta
        packet.theta = self.theta
        self.axis = self.axis * 0.9 + 0.1 * rot_axis
        packet.rot_axis = self.axis

    @staticmethod
    def init_dir():
        if not (os.path.isfile(user_config_dir(appname="explorepy", appauthor="mentalab") + "/conf.ini")):
            os.makedirs(user_config_dir(appname="explorepy", appauthor="mentalab"), exist_ok=True) #create parent directory
            calibre_out_file = user_config_dir(appname="explorepy", appauthor="mentalab") + "/conf.ini"
            with open (calibre_out_file, "w") as f_coef:
                config = configparser.ConfigParser()
                config['DEFAULT'] = {'description': 'configuration data for Explore devices'}
                config.write(f_coef)
                f_coef.close()

        if not (os.path.isdir(user_cache_dir(appname="explorepy", appauthor="Mentalab"))):
            os.makedirs(user_cache_dir(appname="explorepy", appauthor="Mentalab"), exist_ok=True) #create parent directory

    @staticmethod
    def calibrate(cache_dir, device_name):
        calibre_out_file = user_config_dir(appname="explorepy", appauthor="mentalab") + "/conf.ini"
        parser = configparser.SafeConfigParser()
        parser.read(calibre_out_file)
        with open((cache_dir + "_ORN.csv"), "r") as f_set:
            csv_reader = csv.reader(f_set, delimiter=",")
            np_set = list(csv_reader)
            np_set = np.array(np_set[1:], dtype=np.float)
            mag_set_x = np.sort(np_set[:, -3])
            mag_set_y = np.sort(np_set[:, -2])
            mag_set_z = np.sort(np_set[:, -1])
            mx_offset = 0.5 * (mag_set_x[0] + mag_set_x[-1])
            my_offset = 0.5 * (mag_set_y[0] + mag_set_y[-1])
            mz_offset = 0.5 * (mag_set_z[0] + mag_set_z[-1])
            kx = 0.5 * (mag_set_x[-1] - mag_set_x[0])
            ky = 0.5 * (mag_set_y[-1] - mag_set_y[0])
            kz = 0.5 * (mag_set_z[-1] - mag_set_z[0])
            k = np.sort(np.array([kx, ky, kz]))
            kx = 1 / kx
            ky = 1 / ky
            kz = 1 / kz
            f_set.close()
        os.remove((cache_dir + "_ORN.csv"))
        os.remove((cache_dir + "_ExG.csv"))
        os.remove((cache_dir + "_Marker.csv"))
        if parser.has_section(device_name):
            parser = configparser.SafeConfigParser()
            parser.read(calibre_out_file)
            with open(calibre_out_file, "w") as f_coef:
                parser.set(device_name, 'kx', str(kx))
                parser.set(device_name, 'ky', str(ky))
                parser.set(device_name, 'kz', str(kz))
                parser.set(device_name, 'mx', str(mx_offset))
                parser.set(device_name, 'my', str(my_offset))
                parser.set(device_name, 'mz', str(mz_offset))
                parser.write(f_coef)
                f_coef.close()
        else:
            with open(calibre_out_file, "w") as f_coef:
                parser[device_name] = {'kx': str(kx),
                                       'ky': str(ky),
                                       'kz': str(kz),
                                       'mx': str(mx_offset),
                                       'my': str(mx_offset),
                                       'mz': str(mx_offset)}
                parser.write(f_coef)
                f_coef.close()

    @staticmethod
    def check_calibre_data(device_name):
        config = configparser.ConfigParser()
        calibre_file = user_config_dir(appname="explorepy", appauthor="mentalab") + "/conf.ini"
        if os.path.isfile(calibre_file):
            config.read(calibre_file)
            if config.has_section(device_name):
                return True
            else:
                return False
        else:
            return False


