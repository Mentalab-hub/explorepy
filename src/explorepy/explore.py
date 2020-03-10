# -*- coding: utf-8 -*-
"""Explorepy main module

This module provides the main class for interacting with Explore devices.

Examples:
    Before starting a session, make sure your device is paired to your computer. The device will be shown under the
    following name: Explore_XXXX, with the last 4 characters being the last 4 hex numbers of the devices MAC address

    >>> import explorepy
    >>> explore = explorepy.Explore()
    >>> explore.connect(device_name='Explore_1432')  # Put your device Bluetooth name
    >>> explore.visualize(bp_freq=(1, 40), notch_freq=50)
"""

import os
import time
import signal
import sys
from threading import Thread
import csv

import numpy as np

from explorepy.dashboard.dashboard import Dashboard
from explorepy.tools import create_exg_recorder, create_orn_recorder, create_marker_recorder, LslServer
from explorepy.command import MemoryFormat
from explorepy.stream_processor import StreamProcessor, TOPICS


class Explore:
    r"""Mentalab Explore device"""

    def __init__(self):
        self.is_connected = False
        self.stream_processor = None
        self.file_converter = None

    def connect(self, device_name=None, mac_address=None):
        r"""
        Connects to the nearby device. If there are more than one device, the user is asked to choose one of them.

        Args:
            device_name (str): Device name("Explore_XXXX"). Either mac address or name should be in the input
            mac_address (str): The MAC address in format "XX:XX:XX:XX:XX:XX"
        """
        self.stream_processor = StreamProcessor()
        self.stream_processor.start(device_name=device_name, mac_address=mac_address)
        while not self.stream_processor.device_info:
            print('Waiting for device info packet...')
            time.sleep(.3)
        print('Device info packet has been received. Connection has been established. Streaming...')
        self.is_connected = True

    def disconnect(self):
        r"""Disconnects from the device
        """
        self.bt_client.socket.close()
        self.is_connected = False

    def acquire(self, duration=None):
        r"""Start getting data from the device

        Args:
            duration (float): duration of acquiring data (if None it streams data endlessly)
        """
        self._check_connection()

        def callback(packet):
            print(packet)

        self.stream_processor.subscribe(callback=callback, topic=TOPICS.raw_ExG)
        time.sleep(duration)
        self.stream_processor.stop()
        time.sleep(1)

    def record_data(self, file_name, do_overwrite=False, duration=None, file_type='csv'):
        r"""Records the data in real-time

        Args:
            file_name (str): Output file name
            do_overwrite (bool): Overwrite if files exist already
            duration (float): Duration of recording in seconds (if None records endlessly).
            file_type (str): File type of the recorded file. Supported file types: 'csv', 'edf'
        """
        self._check_connection()

        # Check invalid characters
        if set(r'<>{}[]~`*%').intersection(file_name):
            raise ValueError("Invalid character in file name")
        if file_type not in ['edf', 'csv']:
            raise ValueError('{} is not a supported file extension!'.format(file_type))
        duration = self._check_duration(duration)

        exg_out_file = file_name + "_ExG"
        orn_out_file = file_name + "_ORN"
        marker_out_file = file_name + "_Marker"

        exg_recorder = create_exg_recorder(filename=exg_out_file,
                                           file_type=file_type,
                                           fs=self.stream_processor.device_info['sampling_rate'],
                                           adc_mask=self.stream_processor.device_info['adc_mask'],
                                           do_overwrite=do_overwrite)
        orn_recorder = create_orn_recorder(filename=orn_out_file,
                                           file_type=file_type,
                                           do_overwrite=do_overwrite)

        if file_type == 'csv':
            marker_recorder = create_marker_recorder(filename=marker_out_file, do_overwrite=do_overwrite)
        elif file_type == 'edf':
            marker_recorder = exg_recorder
        is_disconnect_occurred = False
        try:
            self.stream_processor.subscribe(callback=exg_recorder.write_data, topic=TOPICS.raw_ExG)
            self.stream_processor.subscribe(callback=orn_recorder.write_data, topic=TOPICS.raw_orn)
            self.stream_processor.subscribe(callback=marker_recorder.set_marker, topic=TOPICS.marker)
            time.sleep(duration)
        except ConnectionAbortedError:
            is_disconnect_occurred = True

        if is_disconnect_occurred:
            print("Error: Recording finished before ", duration, "seconds.")
        else:
            print("Recording finished after ", duration, " seconds.")
        exg_recorder.stop()
        orn_recorder.stop()
        if file_type == 'csv':
            marker_recorder.stop()
        self.stream_processor.stop()
        time.sleep(1)

    def convert_bin(self, bin_file, out_dir='', file_type='edf', do_overwrite=False):
        """Convert a binary file to EDF or CSV file

        Args:
            bin_file (str): Path to the binary file recorded by Explore device
            out_dir (str): Output directory path (must be relative path to the current working directory)
            file_type (str): Output file type: 'edf' for EDF format and 'csv' for CSV format
            do_overwrite (bool): Whether to overwrite an existing file

        """
        if file_type not in ['edf', 'csv']:
            raise ValueError('Invalid file type is given!')
        self.file_type = file_type
        head_path, full_filename = os.path.split(bin_file)
        filename, extension = os.path.splitext(full_filename)
        assert os.path.isfile(bin_file), "Error: File does not exist!"
        assert extension == '.BIN', "File type error! File extension must be BIN."
        exg_out_file = os.getcwd() + out_dir + filename + '_exg'
        orn_out_file = os.getcwd() + out_dir + filename + '_orn'
        marker_out_file = os.getcwd() + out_dir + filename + '_marker'
        self.stream_processor = StreamProcessor()
        self.stream_processor.open_file(bin_file=bin_file)
        self.exg_recorder = create_exg_recorder(filename=exg_out_file,
                                                file_type=self.file_type,
                                                fs=self.stream_processor.device_info['sampling_rate'],
                                                adc_mask=self.stream_processor.device_info['adc_mask'],
                                                do_overwrite=do_overwrite)
        self.orn_recorder = create_orn_recorder(filename=orn_out_file,
                                                file_type=self.file_type,
                                                do_overwrite=do_overwrite)

        if self.file_type == 'csv':
            self.marker_recorder = create_marker_recorder(filename=marker_out_file, do_overwrite=do_overwrite)
        else:
            self.marker_recorder = self.exg_recorder

        self.stream_processor.subscribe(callback=self.exg_recorder.write_data, topic=TOPICS.raw_ExG)
        self.stream_processor.subscribe(callback=self.orn_recorder.write_data, topic=TOPICS.raw_orn)
        self.stream_processor.subscribe(callback=self.marker_recorder.set_marker, topic=TOPICS.marker)

        def device_info_callback(packet):
            new_device_info = packet.get_info()
            if not self.stream_processor.compare_device_info(new_device_info):
                if self.file_type == 'edf':
                    new_file_name = exg_out_file + "_" + str(np.round(packet.timestamp, 0))
                    print("WARNING: Creating a new edf file:", new_file_name + '.edf')
                    self.stream_processor.unsubscribe(callback=self.exg_recorder.write_data, topic=TOPICS.raw_ExG)
                    self.stream_processor.unsubscribe(callback=self.marker_recorder.set_marker, topic=TOPICS.marker)
                    self.exg_recorder.stop()
                    self.exg_recorder = create_exg_recorder(filename=new_file_name,
                                                            file_type=self.file_type,
                                                            fs=self.stream_processor.device_info['sampling_rate'],
                                                            adc_mask=self.stream_processor.device_info['adc_mask'],
                                                            do_overwrite=do_overwrite)
                    self.marker_recorder = self.exg_recorder
                    self.stream_processor.subscribe(callback=self.exg_recorder.write_data, topic=TOPICS.raw_ExG)
                    self.stream_processor.subscribe(callback=self.marker_recorder.set_marker, topic=TOPICS.marker)

        self.stream_processor.subscribe(callback=device_info_callback, topic=TOPICS.device_info)
        self.stream_processor.read()
        print("Converting...")
        while self.stream_processor.is_connected:
            time.sleep(.1)
        print('Conversion finished.')

    def push2lsl(self, duration=None):
        r"""Push samples to two lsl streams

        Args:
            duration (float): duration of data acquiring (if None it streams for one hour).
        """
        self._check_connection()
        duration = self._check_duration(duration)

        lsl_server = LslServer(self.stream_processor.device_info)
        self.stream_processor.subscribe(topic=TOPICS.raw_ExG, callback=lsl_server.push_exg)
        self.stream_processor.subscribe(topic=TOPICS.raw_orn, callback=lsl_server.push_orn)
        self.stream_processor.subscribe(topic=TOPICS.marker, callback=lsl_server.push_marker)
        time.sleep(duration)

        print("Data acquisition finished after ", duration, " seconds.")
        self.stream_processor.stop()
        time.sleep(1)

    def visualize(self, bp_freq=(1, 30), notch_freq=50, calibre_file=None):
        r"""Visualization of the signal in the dashboard

        Args:
            bp_freq (tuple): Bandpass filter cut-off frequencies (low_cutoff_freq, high_cutoff_freq), No bandpass filter
            if it is None.
            notch_freq (int): Line frequency for notch filter (50 or 60 Hz), No notch filter if it is None
            calibre_file (str): Calibration data file name
        """
        assert self.is_connected, "Explore device is not connected. Please connect the device first."

        if notch_freq:
            self.stream_processor.add_filter(cutoff_freq=notch_freq, filter_type='notch')

        if bp_freq:
            if bp_freq[0] and bp_freq[1]:
                self.stream_processor.add_filter(cutoff_freq=bp_freq, filter_type='bandpass')
            elif bp_freq[0]:
                self.stream_processor.add_filter(cutoff_freq=bp_freq[0], filter_type='highpass')
            elif bp_freq[1]:
                self.stream_processor.add_filter(cutoff_freq=bp_freq[1], filter_type='lowpass')

        dashboard = Dashboard(self.stream_processor)
        dashboard.start_server()
        dashboard.start_loop()

    def measure_imp(self, notch_freq=50):
        """
        Visualization of the electrode impedances

        Args:
            notch_freq (int): Notch frequency for filtering the line noise (50 or 60 Hz)
        """
        assert self.is_connected, "Explore device is not connected. Please connect the device first."
        assert self.stream_processor.device_info['sampling_rate'] == 250, \
            "Impedance mode only works in 250 Hz sampling rate!"
        if notch_freq not in [50, 60]:
            raise ValueError('Notch frequency must be either 50 or 60 Hz.')

        self.stream_processor.imp_initialize(notch_freq=notch_freq)

        dashboard = Dashboard(self.stream_processor, mode='impedance')
        dashboard.start_server()
        dashboard.start_loop()

    def set_marker(self, code):
        """Sets a digital event marker while streaming

        Args:
            code (int): Marker code. It must be an integer larger than 7
                        (codes from 0 to 7 are reserved for hardware markers).

        """
        self._check_connection()
        self.stream_processor.set_marker(marker_code=code)

    def format_memory(self):
        """Format memory of the device"""
        cmd = MemoryFormat()
        self.stream_processor.configure_device(cmd)

    def calibrate_orn(self, file_name, do_overwrite=False):
        r"""Calibrate the orientation module of the specified device

        Args:
            file_name (str): filename for calibration. If you pass this parameter, ORN module should be ACTIVE!
            do_overwrite (bool): Overwrite if files exist already
        """
        print("Start recording for 100 seconds, please move the device around during this time, in all directions")
        self.record_data(file_name, do_overwrite=do_overwrite, duration=100, file_type='csv')
        calibre_out_file = file_name + "_calibre_coef.csv"
        assert not (os.path.isfile(calibre_out_file) and do_overwrite), calibre_out_file + " already exists!"
        with open((file_name + "_ORN.csv"), "r") as f_set, open(calibre_out_file, "w") as f_coef:
            f_coef.write("kx, ky, kz, mx_offset, my_offset, mz_offset\n")
            csv_reader = csv.reader(f_set, delimiter=",")
            csv_coef = csv.writer(f_coef, delimiter=",")
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
            calibre_set = np.array([kx, ky, kz, mx_offset, my_offset, mz_offset])
            csv_coef.writerow(calibre_set)
            f_set.close()
            f_coef.close()
        os.remove((file_name + "_ORN.csv"))
        os.remove((file_name + "_ExG.csv"))
        os.remove((file_name + "_Marker.csv"))

    def _check_connection(self):
        assert self.is_connected, "Explore device is not connected. Please connect the device first."

    @staticmethod
    def _check_duration(duration):
        if duration:
            if duration <= 0:
                raise ValueError("Recording time must be a positive number!")
        else:
            duration = 60 * 60  # one hour
        return duration
