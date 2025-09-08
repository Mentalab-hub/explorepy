# -*- coding: utf-8 -*-
"""Some useful tools such as file recorder, heart rate estimation, etc. used in explorepy"""
import csv
import datetime
import logging
import os.path
import socket
from contextlib import closing
from io import StringIO
from threading import Lock

import numpy as np
import pandas
import pyedflib
import serial
from mne import (
    Annotations,
    create_info,
    export,
    io
)
from pylsl import (
    StreamInfo,
    StreamOutlet,
    local_clock
)
from serial.tools import list_ports

import explorepy
from explorepy._exceptions import ExplorePyDeprecationError
from explorepy.filters import ExGFilter
from explorepy.packet import (
    EEG,
    BleImpedancePacket,
    Orientation,
    OrientationV1,
    OrientationV2
)
from explorepy.settings_manager import SettingsManager


logger = logging.getLogger(__name__)
lock = Lock()

TIMESTAMP_SCALE_BLE = 100000

MAX_CHANNELS = 32
EXG_CHANNELS = [f"ch{i}" for i in range(1, MAX_CHANNELS + 1)]
EXG_UNITS = ['uV' for ch in EXG_CHANNELS]
EXG_MAX_LIM = 400000
EXG_MIN_LIM = -400000
ORN_CHANNELS = ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'mx', 'my', 'mz', 'quat_w', 'quat_x', 'quat_y', 'quat_z']
ORN_UNITS = ['mg', 'mg', 'mg', 'mdps', 'mdps',
             'mdps', 'mgauss', 'mgauss', 'mgauss', '1', '1', '1', '1']


def get_local_time():
    """Local time in seconds with sub-ms accuracy (based on pylsl local_clock)

    Returns:
            float: local time in second
    """
    return local_clock()


# checks if the device is an explore pro device or not
def is_explore_pro_device():
    return explorepy.get_bt_interface() == 'ble' or explorepy.get_bt_interface() == 'usb'


def is_ble_mode():
    return explorepy.get_bt_interface() == 'ble'


def is_usb_mode():
    return explorepy.get_bt_interface() == 'usb'


def create_exg_recorder(filename, file_type, adc_mask, fs, do_overwrite, exg_ch=None, batch_mode=False):
    """ Create ExG recorder

    Args:
        filename (str): file name
        file_type (str): file type
        adc_mask (str): channel mask
        fs (int): sampling rate
        do_overwrite (bool): overwrite if the file already exists
        exg_ch (list): list of channel labels

    Returns:
        FileRecorder: file recorder object
    """
    if exg_ch is None:
        exg_ch = ['TimeStamp'] + EXG_CHANNELS
        exg_ch = [exg_ch[0]] + [exg_ch[i + 1]
                                for i, flag in enumerate(reversed(adc_mask)) if flag == 1]
    else:
        exg_ch = ['TimeStamp'] + exg_ch

    exg_unit = ['s'] + EXG_UNITS
    exg_unit = [exg_unit[0]] + [exg_unit[i + 1]
                                for i, flag in enumerate(reversed(adc_mask)) if flag == 1]
    exg_max = [21600.] + [EXG_MAX_LIM for i in range(MAX_CHANNELS)]
    exg_max = [exg_max[0]] + [exg_max[i + 1]
                              for i, flag in enumerate(reversed(adc_mask)) if flag == 1]
    exg_min = [0.] + [EXG_MIN_LIM for i in range(MAX_CHANNELS)]
    exg_min = [exg_min[0]] + [exg_min[i + 1]
                              for i, flag in enumerate(reversed(adc_mask)) if flag == 1]
    return FileRecorder(filename=filename, ch_label=exg_ch, fs=fs, ch_unit=exg_unit,
                        file_type=file_type, do_overwrite=do_overwrite, ch_min=exg_min, ch_max=exg_max,
                        adc_mask=adc_mask, batch_mode=batch_mode)  # noqa: E501


def create_orn_recorder(filename, file_type, do_overwrite, n_chan, batch_mode=False):
    """ Create orientation data recorder

    Args:
        filename (str): file name
        file_type (str): file type
        do_overwrite (bool): overwrite if the file already exists
        n_chan (int): number of orientation channels

    Returns:
        FileRecorder: file recorder object
    """
    orn_ch = ['TimeStamp'] + ORN_CHANNELS[:n_chan]
    orn_unit = ['s'] + ORN_UNITS[:n_chan]
    orn_max = [21600., 2000, 2000, 2000, 250000,
               250000, 250000, 50000, 50000, 50000, 1, 1, 1, 1][:n_chan + 1]
    orn_min = [0, -2000, -2000, -2000, -250000,
               -250000, -250000, -50000, -50000, -50000, 0, 0, 0, 0][:n_chan + 1]
    return FileRecorder(filename=filename, ch_label=orn_ch, fs=20, ch_unit=orn_unit, file_type=file_type,
                        do_overwrite=do_overwrite, ch_max=orn_max, ch_min=orn_min, batch_mode=batch_mode)


def create_marker_recorder(filename, do_overwrite, batch_mode=False):
    """ Create marker recorder

    Args:
        filename (str): file name
        do_overwrite (str): overwrite if the file already exists

    Returns:
        FileRecorder: file recorder object
    """
    marker_ch = ['TimeStamp', 'Code']
    marker_unit = ['s', '-']
    return FileRecorder(filename=filename, ch_label=marker_ch, fs=0, ch_unit=marker_unit,
                        file_type='csv', do_overwrite=do_overwrite, batch_mode=batch_mode)


def create_meta_recorder(filename, fs, adc_mask, device_name, do_overwrite, timestamp='', batch_mode=False):
    """ Create meta file recorder

    Args:
        filename (str): file name
        fs (int): sampling rate
        adc_mask (str): channel mask
        device_name (str): device name
        do_overwrite (str): overwrite if the file already exists
        timestamp (TimeOffset): Clock diff between device timestamp and machine timestamp when the first packet is received in ExplorePy # noqa: E501

    Returns:
        FileRecorder: file recorder object
    """
    header = ['TimeOffset', 'Device', 'sr', 'adcMask', 'ExGUnits']
    exg_unit = 'mV'
    if EXG_UNITS:
        # we only need the first channel's units as this will correspond with the rest
        exg_unit = EXG_UNITS[0]
    return FileRecorder(filename=filename, file_type='csv', ch_label=header, fs=fs, ch_unit=exg_unit,
                        adc_mask=adc_mask, device_name=device_name, do_overwrite=do_overwrite,
                        timestamp=timestamp, batch_mode=batch_mode)


class FileRecorder:
    """Explorepy file recorder class.

    This class can write ExG, orientation and environment data into (separated) EDF+ files. It can write data while
    streaming from Explore device. The incoming data will be stored in a buffer and after it reached fs samples, it
    writes the buffer in EDF file.

    """

    def __init__(self, filename, ch_label, fs, ch_unit, timestamp=None, adc_mask=None, ch_min=None, ch_max=None,
                 device_name='Explore', file_type='edf', do_overwrite=False, batch_mode=False):
        """
        Args:
            filename (str): File name
            ch_label (list): List of channel labels.
            fs (int): Sampling rate (must be identical for all channels)
            ch_unit (list): List of channels unit (e.g. 'uV', 'mG', 's', etc.)
            timestamp (datetime): The time at which this recording starts
            adc_mask (str): Channel mask
            ch_min (list): List of minimum value of each channel. Only needed in edf mode
            ch_max (list): List of maximum value of each channel. Only needed in edf mode
            device_name (str): Recording device name
            file_type (str): File type. current options: 'edf' and 'csv'
            do_overwrite (bool): Overwrite file if a file with the same name exists already
        """
        # Check invalid characters
        if set(r'<>{}[]~`*%').intersection(filename):
            raise ValueError("Invalid character in file name")

        self._file_obj = None
        self.file_type = file_type
        self.timestamp = timestamp
        self._ch_label = ch_label
        self._ch_unit = ch_unit
        self.adc_mask = adc_mask
        self._ch_max = ch_max
        self._ch_min = ch_min
        self._n_chan = len(ch_label)
        self._device_name = device_name
        self._fs = int(fs)
        self._rec_time_offset = None
        self._batch_mode = batch_mode
        self._buffer_lock = Lock()

        if file_type == 'edf':
            if (len(ch_unit) != len(ch_label)) or (len(ch_label) != len(ch_min)) or (len(ch_label) != len(ch_max)):
                print('{}, \n{}, \n{}, \n{}'.format(ch_unit, ch_label, ch_min, ch_max))
                raise ValueError(
                    'ch_label, ch_unit, ch_min and ch_max must have the same length!')
            self._file_name = filename + '.bdf'
            self._create_edf(do_overwrite=do_overwrite)
            self._init_edf_channels()
            self._data = np.zeros((self._n_chan, 0))
            self._annotations_buffer = []
            self._timestamps = []
        elif file_type == 'csv':
            self._file_name = filename + '.csv'
            self._create_csv(do_overwrite=do_overwrite)
        else:
            raise ValueError("File type must be 'edf' or 'csv'")

    @property
    def fs(self):
        """Sampling frequency"""
        return self._fs

    def _create_edf(self, do_overwrite):
        if (not do_overwrite) and os.path.isfile(self._file_name):
            raise FileExistsError(self._file_name + ' already exists!')
        assert self._file_obj is None, "Usage Error: File object has been created already."
        self._file_obj = pyedflib.EdfWriter(
            self._file_name, self._n_chan, file_type=pyedflib.FILETYPE_BDFPLUS)

    def _create_csv(self, do_overwrite):
        if (not do_overwrite) and os.path.isfile(self._file_name):
            raise FileExistsError(self._file_name + ' already exists!')
        assert self._file_obj is None, "Usage Error: File object has been created already."
        if not self._batch_mode:
            self._file_obj = open(self._file_name, 'w', newline='\n')
            self._csv_obj = csv.writer(self._file_obj, delimiter=",")
            self._csv_obj.writerow(self._ch_label)
        else:
            self._file_obj = open(self._file_name, 'wb')

            # Write headers
            header = ','.join(self._ch_label) + '\n'
            self._file_obj.write(header.encode('utf-8'))

    def _init_edf_channels(self):
        """Initialize EDF channels with signal parameters"""
        self._file_obj.setEquipment(self._device_name)
        self._file_obj.setStartdatetime(datetime.datetime.now())

        ch_info_list = []
        for ch in zip(self._ch_label, self._ch_unit, self._ch_max, self._ch_min):
            ch_info_list.append({
                'label': ch[0],
                'dimension': ch[1],
                'sample_rate': self._fs,
                'physical_max': ch[2],
                'physical_min': ch[3],
                'digital_max': 8388607,
                'digital_min': -8388608,
                'prefilter': '',
                'transducer': ''
            })
        for i, ch_info in enumerate(ch_info_list):
            self._file_obj.setSignalHeader(i, ch_info)

    def _write_edf_anno(self):
        """Write annotations in EDF file"""
        for ts, code in list(self._annotations_buffer):
            idx = np.argmax(np.array(self._timestamps) > ts) - 1
            if idx != -1:
                timestamp = idx / self.fs
                self._file_obj.writeAnnotation(timestamp, 0.001, code)
                self._annotations_buffer.remove((ts, code))

    def stop(self):
        """Stop recording"""
        assert self._file_obj is not None, "Usage Error: File object has not been created yet."
        if self.file_type == 'edf':
            if self._data.shape[1] > 0:
                with lock:
                    self._file_obj.writeSamples(list(self._data))
                    self._write_edf_anno()
            self._file_obj.close()
            self._file_obj = None
        elif self.file_type == 'csv':
            self._file_obj.close()
            self._file_obj = None

    def set_marker(self, packet):
        """Writes a marker event in the file

        Args:
            packet (explorepy.packet.EventMarker): Event marker packet

        """
        timestamp, code = packet.get_data()
        timestamp[0] = round(timestamp[0], 4)
        if self.file_type == 'csv':
            if not self._batch_mode:
                data = timestamp + code
                self._csv_obj.writerow(data)
                self._file_obj.flush()
            else:
                marker_data = np.array(timestamp + code)[:, np.newaxis]
                output = StringIO()
                np.savetxt(output, marker_data.T, fmt='%s', delimiter=',', newline='\n')
                self._file_obj.write(output.getvalue().encode('utf-8'))
                self._file_obj.flush()
        elif self.file_type == 'edf':
            if self._rec_time_offset is None:
                self._rec_time_offset = timestamp[0]
            self._annotations_buffer.append((timestamp[0], code[0]))

    def write_meta(self):
        """Writes meta data in the file"""
        channels = [
            'ch' + str(i + 1) for i, flag in enumerate(reversed(self.adc_mask)) if flag == 1]
        if not self._batch_mode:
            row = [self.timestamp, self._device_name, self._fs, str(' '.join(channels)), self._ch_unit]
            self._csv_obj.writerow(row)
            self._file_obj.flush()
        else:
            meta_row = \
                f"{self.timestamp or ''},{self._device_name},{self._fs},{' '.join(channels)},{''.join(self._ch_unit)}\n"
            self._file_obj.write(meta_row.encode('utf-8'))
            self._file_obj.flush()

    def _write_edf(self, packet):
        time_vector, sig = packet.get_data(self._fs)
        if isinstance(packet, Orientation) and len(time_vector) == 1:
            data = np.array(time_vector + sig)[:, np.newaxis]
        else:
            if self._rec_time_offset is None:
                self._rec_time_offset = time_vector[0]
            data = np.concatenate((np.array(time_vector)[:, np.newaxis].T, np.array(sig)), axis=0)
        data = np.round(data, 4)
        if isinstance(packet, EEG):
            indices = [0] + [i + 1 for i, flag in enumerate(reversed(self.adc_mask)) if flag == 1]
            data = data[indices]
        if data.shape[0] != self._n_chan:
            raise ValueError('Input first dimension must be {}'.format(self._n_chan))
        self._data = np.concatenate((self._data, data), axis=1)
        self._timestamps += list(data[0, :])
        with lock:
            if self._data.shape[1] > self._fs:
                self._file_obj.writeSamples(list(self._data[:, :self._fs]))
                self._write_edf_anno()
                self._data = self._data[:, self._fs:]

    def _process_packet_data(self, packet):
        """Helper function to extract and format data from a packet."""
        time_vector, sig = packet.get_data(self._fs)
        if isinstance(packet, Orientation) and len(time_vector) == 1:
            data = np.array(time_vector + sig)[:, np.newaxis]
        else:
            if self._rec_time_offset is None:
                self._rec_time_offset = time_vector[0]
            data = np.concatenate((np.array(time_vector)[:, np.newaxis].T, np.array(sig)), axis=0)
        data = np.round(data, 4)
        if isinstance(packet, EEG):
            indices = [0] + [i + 1 for i, flag in enumerate(reversed(self.adc_mask)) if flag == 1]
            data = data[indices]
        return data

    def _process_batch_csv(self, packet):
        """Process a batch of packets for CSV output."""
        if isinstance(packet[0], OrientationV1):
            data = np.array([[p.timestamp] + p.acc.tolist() + p.gyro.tolist() + p.
                            mag.tolist() for p in packet]).T
        elif isinstance(packet[0], OrientationV2):
            data = np.array([[p.timestamp] + p.acc.tolist() + p.gyro.tolist() + p.mag.
                            tolist() + p.quat.tolist() for p in packet]).T
        elif isinstance(packet[0], EEG):
            all_data = np.concatenate([p.data for p in packet], axis=1)
            n_total_samples = all_data.shape[1]
            start_time = packet[0].timestamp
            time_vector = np.linspace(start_time, start_time + (n_total_samples - 1) / self._fs,
                                      n_total_samples)
            data = np.concatenate((time_vector[np.newaxis, :], all_data), axis=0)
        else:
            time_vector, sig = packet.get_data(self._fs)
            if self._rec_time_offset is None:
                self._rec_time_offset = time_vector[0]
            data = np.concatenate((np.array(time_vector)[:, np.newaxis].T, np.array(sig)), axis=0)
        return data

    def write_data(self, packet):
        """Writes data to the file
        Notes:
            If file type is set to EDF, this function writes each 1 seconds of data. If the input is less than 1 second,
            it will be buffered in the memory and it will be written in the file when enough data is in the buffer.
        Args:
            packet (explorepy.packet.Packet): ExG or Orientation packet
        """
        if self.file_type == 'edf':
            if not self._batch_mode:
                self._write_edf(packet)
            else:
                for p in packet:
                    self._write_edf(packet=p)
        elif self.file_type == 'csv':
            if not self._batch_mode:
                data = self._process_packet_data(packet)
                try:
                    self._csv_obj.writerows(data.T.tolist())
                    self._file_obj.flush()
                except ValueError as e:
                    logger.debug('Value error on file write: {}'.format(e))
            else:
                data = self._process_batch_csv(packet)
                np.savetxt(self._file_obj, data.T, fmt='%4f', delimiter=',')


class LslServer:
    """Class for LabStreamingLayer integration"""
    def __init__(self, device_info, stream_name=None):
        self.device_name = device_info["device_name"]
        self.marker_outlet = None
        self.exg_outlet = None
        self.orn_outlet = None
        self.adc_mask = SettingsManager(
            device_info["device_name"]).get_adc_mask()
        if len(SettingsManager(device_info["device_name"]).get_channel_names()) == len(self.adc_mask):
            self.channel_names = SettingsManager(device_info["device_name"]).get_channel_names()
        else:
            self.channel_names = EXG_CHANNELS
        self.stream_name = stream_name or device_info["device_name"]
        self.n_chan = self.adc_mask.count(1)
        self.exg_fs = device_info['sampling_rate']
        self.orn_fs = 20
        self.orn_ch = get_orn_chan_len(device_info)

    def initialize_outlets(self):
        info_exg = StreamInfo(name=self.device_name + "_ExG",
                              type='ExG',
                              channel_count=self.n_chan,
                              nominal_srate=self.exg_fs,
                              channel_format='float32',
                              source_id=self.device_name + "_ExG")
        info_exg.desc().append_child_value("manufacturer", "Mentalab")
        channels = info_exg.desc().append_child("channels")
        for i, mask in enumerate(self.adc_mask):
            if mask == 1:
                channels.append_child("channel") \
                    .append_child_value("name", self.channel_names[i]) \
                    .append_child_value("unit", EXG_UNITS[i]) \
                    .append_child_value("type", "ExG")
        info_orn = StreamInfo(name=self.device_name + "_ORN",
                              type='ORN',
                              channel_count=self.orn_ch,
                              nominal_srate=self.orn_fs,
                              channel_format='float32',
                              source_id=self.device_name + "_ORN")
        info_orn.desc().append_child_value("manufacturer", "Mentalab")
        channels = info_orn.desc().append_child("channels")
        for chan, unit in zip(ORN_CHANNELS, ORN_UNITS):
            channels.append_child("channel") \
                .append_child_value("name", chan) \
                .append_child_value("unit", unit) \
                .append_child_value("type", "ORN")
        info_marker = StreamInfo(name=self.device_name + "_Marker",
                                 type='Markers',
                                 channel_count=1,
                                 nominal_srate=0,
                                 channel_format='string',
                                 source_id=self.device_name + "_Markers")
        logger.info(
            f"LSL Streams have been created with names/source IDs as the following:\n"
            f"\t\t\t\t\t {self.device_name}_ExG\n"
            f"\t\t\t\t\t {self.device_name}_ORN\n"
            f"\t\t\t\t\t {self.device_name}_Markers\n"
        )
        self.orn_outlet = StreamOutlet(info_orn)
        self.exg_outlet = StreamOutlet(info_exg)
        self.marker_outlet = StreamOutlet(info_marker)

    def push_exg(self, packet):
        """Push data to ExG outlet

        Args:
            packet (explorepy.packet.EEG): ExG packet
        """
        _, exg_data = packet.get_data(self.exg_fs)
        if isinstance(packet, EEG):
            indices = [i for i, flag in enumerate(
                reversed(self.adc_mask)) if flag == 1]
            exg_data = exg_data[indices]
        self.exg_outlet.push_chunk(exg_data.T.tolist())

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
        self.packet_buffer = []

    def _add_filters(self):
        bp_freq = self._device_info['sampling_rate'] / 4 - \
            1.5, self._device_info['sampling_rate'] / 4 + 1.5
        noise_freq = self._device_info['sampling_rate'] / \
            4 + 2.5, self._device_info['sampling_rate'] / 4 + 5.5
        settings_manager = SettingsManager(self._device_info["device_name"])
        settings_manager.load_current_settings()
        n_chan = settings_manager.settings_dict[settings_manager.channel_count_key]
        # Temporary fix for 16/32 channel filters
        if not is_explore_pro_device():
            if n_chan >= 16:
                n_chan = 32
        self._filters['notch'] = ExGFilter(cutoff_freq=self._notch_freq,
                                           filter_type='notch_imp',
                                           s_rate=self._device_info['sampling_rate'],
                                           n_chan=n_chan)

        self._filters['demodulation'] = ExGFilter(cutoff_freq=bp_freq,
                                                  filter_type='bandpass',
                                                  s_rate=self._device_info['sampling_rate'],
                                                  n_chan=n_chan)

        self._filters['base_noise'] = ExGFilter(cutoff_freq=noise_freq,
                                                filter_type='bandpass',
                                                s_rate=self._device_info['sampling_rate'],
                                                n_chan=n_chan)

    def measure_imp(self, packet):
        """Compute electrode impedances
        """
        self.packet_buffer.append(packet)

        if len(self.packet_buffer) < 16:
            return None
        else:
            timestamp, _ = self.packet_buffer[0].get_data()
            resized_packet = BleImpedancePacket(
                timestamp=timestamp, payload=None)
            resized_packet.populate_packet_with_data(self.packet_buffer)
            self.packet_buffer.clear()
            temp_packet = self._filters['notch'].apply(
                input_data=resized_packet, in_place=False)
            self._calib_param['noise_level'] = self._filters['base_noise']. \
                apply(input_data=temp_packet, in_place=False).get_ptp()
            self._filters['demodulation'].apply(
                input_data=temp_packet, in_place=True
            ).calculate_impedance(self._calib_param)
            return temp_packet


def find_free_port():
    """Find a free port on the localhost

    Returns:
        int: Port number
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as free_socket:
        free_socket.bind(('localhost', 0))
        free_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        port_number = free_socket.getsockname()[1]
        return port_number


def get_raw_data_from_csv(file_name):
    print('File name is {}'.format(file_name))
    meta_ending = "_Meta.csv"
    meta_file = file_name[:-8] + meta_ending
    if not os.path.isfile(file_name[:-8] + meta_ending):
        logger.error("Could not find Meta file while trying to convert from csv, quitting...")
        return None

    sampling_freq = pandas.read_csv(meta_file, delimiter=',')['sr'][0]
    data_frame_exg = pandas.read_csv(file_name, delimiter=',')
    first_exg_ts = data_frame_exg['TimeStamp'].tolist()[0]
    data_frame_exg = data_frame_exg.drop('TimeStamp', axis=1)
    ch_types = ["eeg"] * len(data_frame_exg.columns)
    info = create_info(ch_names=data_frame_exg.columns.tolist(), sfreq=sampling_freq, ch_types=ch_types)

    data_frame_exg = data_frame_exg.div(1e6)
    data_frame_exg = data_frame_exg.transpose()
    raw_data = io.RawArray(data_frame_exg, info)

    data_frame_marker = pandas.read_csv(file_name[:-8] + '_Marker.csv', delimiter=',')
    marker_ts = data_frame_marker['TimeStamp'].tolist()
    onsets = [x - first_exg_ts for x in marker_ts]
    durations = [0 for _ in range(len(onsets))]
    descriptions = data_frame_marker['Code']
    annotations = Annotations(onset=onsets, duration=durations, description=descriptions)
    raw_data.set_annotations(annotations)

    return raw_data


def generate_eeglab_dataset(file_name, output_name):
    """Generates an EEGLab dataset from edf(bdf+) file
    """
    file_ext = os.path.splitext(file_name)[1]
    raw_data = None
    if file_ext == ".csv":
        try:
            raw_data = get_raw_data_from_csv(file_name)
        except Exception as e:
            logger.error(f"Got error {e} for file : {file_name}")
    elif file_ext == ".bdf":
        raw_data = io.read_raw_bdf(file_name)
        raw_data = raw_data.drop_channels(raw_data.ch_names[0])
    else:
        raise ValueError(f"Encountered invalid file extension while trying to generate EEGLab dataset: {file_ext}")

    if raw_data:
        export.export_raw(output_name, raw_data,
                          fmt='eeglab',
                          overwrite=True, physical_range=[-400000, 400000])


def compare_recover_from_bin(file_name_csv, file_name_device):
    """Compares and recovers missing samples of csv file by comparing data from binary file

            Args:
            file_name_csv (str): Name of recorded csv file without extension
            file_name_device_csv (str): Name of converted csv file
        """
    bin_df = pandas.read_csv(file_name_device + '_ExG.csv')
    csv_df = pandas.read_csv(file_name_csv + '_ExG.csv')
    meta_df = pandas.read_csv(file_name_csv + "_Meta.csv")
    timestamp_key = 'TimeStamp'
    sampling_rate = meta_df['sr'][0]
    offset_ = meta_df["TimeOffset"][0]
    offset_ = round(offset_, 4)
    time_period = 1 / sampling_rate

    start = csv_df[timestamp_key][0] - offset_ - time_period
    stop = csv_df[timestamp_key][len(
        csv_df[timestamp_key]) - 1] - offset_ + time_period
    bin_df = bin_df[(bin_df[timestamp_key] >= start)
                    & (bin_df[timestamp_key] <= stop)]
    bin_df[timestamp_key] = bin_df[timestamp_key] + offset_
    bin_df.to_csv(file_name_csv + '_recovered_ExG.csv', index=False)


def setup_usb_marker_port():
    ports = list_ports.comports(False)
    port = 0
    for p in ports:
        if p.vid == 0x0483 and p.pid == 0x5740:
            logger.info('Found an Explore Pro device connected.')
            port = p.device

    if port == 0:
        logger.info('No USB device found, setting up the port as None')
        return None
    else:
        logger.info('Found connected device, opening a USB port')
        return serial.Serial(port=port, baudrate=2000000, timeout=0.5)


def check_bin_compatibility(file_name):
    with open(file_name, "rb") as f:
        b = f.read(1).hex()
        if b != "62":
            raise ExplorePyDeprecationError()


def get_orn_chan_len(device_info):
    fw_version = str.split(device_info["firmware_version"][-3:], '.')
    fw_version = int(10 * fw_version[0] + fw_version[1])
    orn_ch = 13 if fw_version >= 7 else 9
    return orn_ch
