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

import logging
import os
import re
import time
from datetime import datetime
from threading import Timer

import numpy as np
from appdirs import user_cache_dir

import explorepy
from explorepy.command import (
    MemoryFormat,
    ModuleDisable,
    ModuleEnable,
    SetCh,
    SetChTest,
    SetSPS,
    SoftReset
)
from explorepy.settings_manager import SettingsManager
from explorepy.stream_processor import (
    TOPICS,
    StreamProcessor
)
from explorepy.tools import (
    LslServer,
    PhysicalOrientation,
    create_exg_recorder,
    create_marker_recorder,
    create_meta_recorder,
    create_orn_recorder
)


logger = logging.getLogger(__name__)


class Explore:
    r"""Mentalab Explore device"""

    def __init__(self):
        self.is_connected = False
        self.stream_processor = None
        self.recorders = {}
        self.lsl = {}
        self.device_name = None

    @property
    def is_measuring_imp(self):
        """Return impedance status"""
        imp_mode = False
        if self.stream_processor:
            imp_mode = self.stream_processor._is_imp_mode
        return imp_mode

    def connect(self, device_name=None, mac_address=None):
        r"""
        Connects to the nearby device. If there are more than one device, the user is asked to choose one of them.

        Args:
            device_name (str): Device name("Explore_XXXX"). Either mac address or name should be in the input
            mac_address (str): The MAC address in format "XX:XX:XX:XX:XX:XX"
        """
        if device_name:
            self.device_name = device_name
        else:
            self.device_name = 'Explore_' + mac_address[-5:-3] + mac_address[-2:]
        logger.info(f"Connecting to {self.device_name} ...")
        self.stream_processor = StreamProcessor()
        self.stream_processor.start(device_name=device_name, mac_address=mac_address)
        cnt = 0
        while "adc_mask" not in self.stream_processor.device_info:
            logger.info("Waiting for device info packet...")
            time.sleep(1)
            if cnt >= 10:
                raise ConnectionAbortedError("Could not get info packet from the device")
            cnt += 1

        logger.info('Device info packet has been received. Connection has been established. Streaming...')
        logger.info("Device info: " + str(self.stream_processor.device_info))
        self.is_connected = True
        self.stream_processor.send_timestamp()

    def disconnect(self):
        r"""Disconnects from the device
        """
        self.is_connected = False
        self.device_name = None

        if self.lsl:
            self.stop_lsl()

        if self.recorders:
            self.stop_recording()

        if self.is_measuring_imp:
            self.stream_processor.disable_imp()
            self.is_measuring_imp = False

        self.stream_processor.stop()
        logger.debug("Device has been disconnected.")

    def acquire(self, duration=None):
        r"""Start getting data from the device

        Args:
            duration (float): duration of acquiring data (if None it streams data endlessly)
        """
        self._check_connection()
        duration = self._check_duration(duration)

        def callback(packet):
            print(packet)

        self.stream_processor.subscribe(callback=callback, topic=TOPICS.raw_ExG)
        logger.debug(f"Acquiring and printing data stream for {duration}s ...")
        time.sleep(duration)
        self.stream_processor.unsubscribe(callback=callback, topic=TOPICS.raw_ExG)

    def record_data(
        self, file_name, do_overwrite=False, duration=None, file_type='csv', block=False, exg_ch_names=None
    ):
        r"""Records the data in real-time

        Args:
            file_name (str): Output file name
            do_overwrite (bool): Overwrite if files exist already
            duration (float): Duration of recording in seconds (if None records endlessly).
            file_type (str): File type of the recorded file. Supported file types: 'csv', 'edf'
            block (bool): Record in blocking mode if 'block' is True
            exg_ch_names (list): list of channel names. If None, default names are used.
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
        meta_out_file = file_name + "_Meta"

        self.recorders['exg'] = create_exg_recorder(filename=exg_out_file,
                                                    file_type=file_type,
                                                    fs=self.stream_processor.device_info['sampling_rate'],
                                                    adc_mask=SettingsManager(self.device_name).get_adc_mask(),
                                                    do_overwrite=do_overwrite,
                                                    exg_ch=exg_ch_names)
        self.recorders['orn'] = create_orn_recorder(filename=orn_out_file,
                                                    file_type=file_type,
                                                    do_overwrite=do_overwrite)

        if file_type == 'csv':
            self.recorders['marker'] = create_marker_recorder(filename=marker_out_file, do_overwrite=do_overwrite)
            self.recorders['meta'] = create_meta_recorder(filename=meta_out_file,
                                                          fs=self.stream_processor.device_info['sampling_rate'],
                                                          adc_mask=SettingsManager(self.device_name).get_adc_mask(),
                                                          device_name=self.device_name,
                                                          do_overwrite=do_overwrite,
                                                          timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            self.recorders['meta'].write_meta()
            self.recorders['meta'].stop()

        elif file_type == 'edf':
            self.recorders['marker'] = self.recorders['exg']
            logger.warning("Markers' timing might not be precise in EDF files. We recommend recording in CSV format "
                           "if you are setting markers during the recording.")

        self.stream_processor.subscribe(callback=self.recorders['exg'].write_data, topic=TOPICS.raw_ExG)
        self.stream_processor.subscribe(callback=self.recorders['orn'].write_data, topic=TOPICS.raw_orn)
        self.stream_processor.subscribe(callback=self.recorders['marker'].set_marker, topic=TOPICS.marker)
        logger.info("Recording...")

        self.recorders['timer'] = Timer(duration, self.stop_recording)

        self.recorders['timer'].start()
        if block:
            try:
                while 'timer' in self.recorders.keys() and self.recorders['timer'].is_alive():
                    time.sleep(.3)
            except KeyboardInterrupt:
                logger.info("Got Keyboard Interrupt while recording in blocked mode!")
                self.stop_recording()
                self.stream_processor.stop()
                time.sleep(1)

    def stop_recording(self):
        """Stop recording"""
        if self.recorders:
            self.stream_processor.unsubscribe(callback=self.recorders['exg'].write_data, topic=TOPICS.raw_ExG)
            self.stream_processor.unsubscribe(callback=self.recorders['orn'].write_data, topic=TOPICS.raw_orn)
            self.stream_processor.unsubscribe(callback=self.recorders['marker'].set_marker, topic=TOPICS.marker)
            self.recorders['exg'].stop()
            self.recorders['orn'].stop()
            if self.recorders['exg'].file_type == 'csv':
                self.recorders['marker'].stop()
            if 'timer' in self.recorders.keys() and self.recorders['timer'].is_alive():
                self.recorders['timer'].cancel()
            self.recorders = {}
            logger.info('Recording stopped.')
        else:
            logger.debug("Tried to stop recording while no recorder is running!")

    def convert_bin(self, bin_file, out_dir='', file_type='edf', do_overwrite=False, out_dir_is_full=False):
        """Convert a binary file to EDF or CSV file

        Args:
            bin_file (str): Path to the binary file recorded by Explore device
            out_dir (str): Output directory path (must be relative path to the current working directory)
            file_type (str): Output file type: 'edf' for EDF format and 'csv' for CSV format
            do_overwrite (bool): Whether to overwrite an existing file

        """
        if file_type not in ['edf', 'csv']:
            raise ValueError('Invalid file type is given!')
        self.recorders['file_type'] = file_type
        head_path, full_filename = os.path.split(bin_file)
        filename, extension = os.path.splitext(full_filename)
        assert os.path.isfile(bin_file), "Error: File does not exist!"
        assert extension == '.BIN', "File type error! File extension must be BIN."
        if out_dir_is_full:
            out_full_path = out_dir
        else:
            out_full_path = os.path.join(os.getcwd(), out_dir)
        exg_out_file = os.path.join(out_full_path, filename + '_ExG')
        orn_out_file = os.path.join(out_full_path, filename + '_ORN')
        marker_out_file = os.path.join(out_full_path, filename + '_Marker')
        meta_out_file = os.path.join(out_full_path, filename + '_Meta')

        self.stream_processor = StreamProcessor()
        self.stream_processor.read_device_info(bin_file=bin_file)
        self.mask = self.stream_processor.device_info['adc_mask']
        if 'board_id' in self.stream_processor.device_info:
            if 'PCB_304_801_XXX' in self.stream_processor.device_info['board_id']:
                self.mask = [1 for i in range(0, 32)]

        self.recorders['exg'] = create_exg_recorder(filename=exg_out_file,
                                                    file_type=self.recorders['file_type'],
                                                    fs=self.stream_processor.device_info['sampling_rate'],
                                                    adc_mask=self.mask,
                                                    do_overwrite=do_overwrite)
        self.recorders['orn'] = create_orn_recorder(filename=orn_out_file,
                                                    file_type=self.recorders['file_type'],
                                                    do_overwrite=do_overwrite)

        if self.recorders['file_type'] == 'csv':
            self.recorders['marker'] = create_marker_recorder(filename=marker_out_file, do_overwrite=do_overwrite)
            self.recorders['meta'] = create_meta_recorder(filename=meta_out_file,
                                                          fs=self.stream_processor.device_info['sampling_rate'],
                                                          adc_mask=self.mask,
                                                          device_name=self.device_name,
                                                          do_overwrite=do_overwrite)
            self.recorders['meta'].write_meta()
            self.recorders['meta'].stop()
        else:
            self.recorders['marker'] = self.recorders['exg']

        self.stream_processor.subscribe(callback=self.recorders['exg'].write_data, topic=TOPICS.raw_ExG)
        self.stream_processor.subscribe(callback=self.recorders['orn'].write_data, topic=TOPICS.raw_orn)
        self.stream_processor.subscribe(callback=self.recorders['marker'].set_marker, topic=TOPICS.marker)

        def device_info_callback(packet):
            new_device_info = packet.get_info()
            if not self.stream_processor.compare_device_info(new_device_info):
                new_file_name = exg_out_file + "_" + str(np.round(packet.timestamp, 0))
                new_meta_name = meta_out_file + "_" + str(np.round(packet.timestamp, 0))
                logger.warning("Creating a new file: " + new_file_name + '.' + self.recorders['file_type'])
                self.stream_processor.unsubscribe(callback=self.recorders['exg'].write_data, topic=TOPICS.raw_ExG)
                self.stream_processor.unsubscribe(callback=self.recorders['marker'].set_marker, topic=TOPICS.marker)
                self.recorders['exg'].stop()
                self.recorders['exg'] = create_exg_recorder(filename=new_file_name,
                                                            file_type=self.recorders['file_type'],
                                                            fs=self.stream_processor.device_info['sampling_rate'],
                                                            adc_mask=self.stream_processor.device_info['adc_mask'],
                                                            do_overwrite=do_overwrite)

                if self.recorders['file_type'] == 'edf':
                    self.recorders['marker'] = self.recorders['exg']

                self.stream_processor.subscribe(callback=self.recorders['exg'].write_data, topic=TOPICS.raw_ExG)
                self.stream_processor.subscribe(callback=self.recorders['marker'].set_marker, topic=TOPICS.marker)

                if self.recorders['file_type'] == 'csv':
                    self.recorders['meta'] = create_meta_recorder(
                        filename=new_meta_name,
                        fs=self.stream_processor.device_info['sampling_rate'],
                        adc_mask=self.stream_processor.device_info['adc_mask'],
                        device_name=self.device_name,
                        do_overwrite=do_overwrite)
                    self.recorders['meta'].write_meta()
                    self.recorders['meta'].stop()

        self.stream_processor.subscribe(callback=device_info_callback, topic=TOPICS.device_info)
        self.stream_processor.open_file(bin_file=bin_file)
        logger.info("Converting...")
        while self.stream_processor.is_connected:
            time.sleep(.1)

        if self.recorders['file_type'] == 'csv':
            self.recorders["marker"].stop()
        self.recorders["exg"].stop()
        self.recorders["orn"].stop()
        logger.info('Conversion finished.')

    def push2lsl(self, duration=None, block=False):
        r"""Push samples to two lsl streams (ExG and ORN streams)

        Args:
            duration (float): duration of data acquiring (if None it streams for one hour).
            block (bool): blocking mode
        """
        self._check_connection()
        duration = self._check_duration(duration)

        self.lsl['timer'] = Timer(duration, self.stop_lsl)
        self.lsl['server'] = LslServer(self.stream_processor.device_info)
        self.stream_processor.subscribe(topic=TOPICS.raw_ExG, callback=self.lsl['server'].push_exg)
        self.stream_processor.subscribe(topic=TOPICS.raw_orn, callback=self.lsl['server'].push_orn)
        self.stream_processor.subscribe(topic=TOPICS.marker, callback=self.lsl['server'].push_marker)
        self.lsl['timer'].start()

        if block:
            try:
                while 'timer' in self.lsl.keys() and self.lsl['timer'].is_alive():
                    time.sleep(.3)
            except KeyboardInterrupt:
                logger.info("Got Keyboard Interrupt while pushing data to LSL in blocked mode!")
                self.stream_processor.stop()
                self.stop_lsl()
                time.sleep(1)

    def stop_lsl(self):
        """Stop pushing data to LSL streams"""
        if self.lsl:
            self.stream_processor.unsubscribe(topic=TOPICS.raw_ExG, callback=self.lsl['server'].push_exg)
            self.stream_processor.unsubscribe(topic=TOPICS.raw_orn, callback=self.lsl['server'].push_orn)
            self.stream_processor.unsubscribe(topic=TOPICS.marker, callback=self.lsl['server'].push_marker)
            if self.lsl['timer'].is_alive():
                self.lsl['timer'].cancel()
            self.lsl = {}
            logger.info("Push2lsl has been stopped.")
        else:
            logger.debug("Tried to stop LSL while no LSL server is running!")

    def visualize(self, bp_freq=(1, 30), notch_freq=50):
        r"""Visualization of the signal in the dashboard: only works for 4 and 8 channel devices

        Args:
            bp_freq (tuple): Bandpass filter cut-off frequencies (low_cutoff_freq, high_cutoff_freq), No bandpass filter
            if it is None.
            notch_freq (int): Line frequency for notch filter (50 or 60 Hz), No notch filter if it is None
        """
        self._check_connection()

        if notch_freq:
            self.stream_processor.add_filter(cutoff_freq=notch_freq, filter_type='notch')

        if bp_freq:
            if bp_freq[0] and bp_freq[1]:
                self.stream_processor.add_filter(cutoff_freq=bp_freq, filter_type='bandpass')
            elif bp_freq[0]:
                self.stream_processor.add_filter(cutoff_freq=bp_freq[0], filter_type='highpass')
            elif bp_freq[1]:
                self.stream_processor.add_filter(cutoff_freq=bp_freq[1], filter_type='lowpass')

        dashboard = explorepy.Dashboard(explore=self)
        dashboard.start_server()
        dashboard.start_loop()

    def measure_imp(self):
        """
        Visualization of the electrode impedance
        """
        self._check_connection()
        assert self.stream_processor.device_info['sampling_rate'] == 250, \
            "Impedance mode only works at 250 Hz sampling rate. Please set the sampling rate to 250Hz."

        self.stream_processor.imp_initialize(notch_freq=50)

        try:
            dashboard = explorepy.Dashboard(explore=self, mode='impedance')
            dashboard.start_server()
            dashboard.start_loop()
        except KeyboardInterrupt:
            self.stream_processor.disable_imp()

    def set_marker(self, code):
        """Sets a digital event marker while streaming

        Args:
            code (int): Marker code (must be in range of 0-65535)

        """
        self._check_connection()
        self.stream_processor.set_marker(code=code)

    def format_memory(self):
        """Format memory of the device

        Returns:
            bool: True for success, False otherwise.
        """
        self._check_connection()
        cmd = MemoryFormat()
        return self.stream_processor.configure_device(cmd)

    def set_sampling_rate(self, sampling_rate):
        """Set sampling rate

        Args:
            sampling_rate (int): Desired sampling rate. Options: 250, 500, 1000

        Returns:
            bool: True for success, False otherwise
        """
        self._check_connection()
        if sampling_rate not in [250, 500, 1000]:
            raise ValueError("Sampling rate must be 250, 500 or 1000.")
        cmd = SetSPS(sampling_rate)
        if self.stream_processor.configure_device(cmd):
            SettingsManager(self.device_name).set_sampling_rate(sampling_rate)
            return True

    def reset_soft(self):
        """Reset the device to the default settings

        Note:
            The Bluetooth will be disconnected by the Explore device after resetting.

        Returns:
            bool: True for success, False otherwise
        """
        self._check_connection()
        cmd = SoftReset()
        if self.stream_processor.configure_device(cmd):
            self.disconnect()
            return True
        return False

    def set_channels(self, channel_mask):
        """Set the channel mask of the device

        The channels can be disabled/enabled by calling this function and passing either bytes or binary string
         representing the mask. For example in a 4 channel device, if you want to disable channel 4, the adc mask
         should be b'0111' (LSB is channel 1). The inputs to this function can be b'0111' '0111'.

        Args:
            channel_mask (bytes str): Bytes or String representing the binary channel mask

        Example:
            >>> from explorepy.explore import Explore
            >>> explore = Explore()
            >>> explore.connect(device_name='Explore_2FA2')
            >>> explore.set_channels(channel_mask='0111')  # disable channel 4 - mask:0111

        Returns:
            bool: True for success, False otherwise
        """
        if SettingsManager(self.device_name).get_channel_count() > 8:
            SettingsManager(self.device_name).set_adc_mask(channel_mask)
            return True
        channel_mask_int = self._convert_chan_mask(channel_mask)
        self._check_connection()
        cmd = SetCh(channel_mask_int)
        if self.stream_processor.configure_device(cmd):
            return True

    def disable_module(self, module_name):
        """Disable module

        Args:
            module_name (str): Module to be disabled (options: 'ENV', 'ORN', 'EXG')

        Examples:
            >>> from explorepy.explore import Explore
            >>> explore = Explore()
            >>> explore.connect(device_name='Explore_2FA2')
            >>> explore.disable_module('ORN')

        Returns:
            bool: True for success, False otherwise
        """
        if module_name not in ['ORN', 'ENV', 'EXG']:
            raise ValueError('Module name must be one of ORN, ENV or EXG.')
        self._check_connection()
        cmd = ModuleDisable(module_name)
        return self.stream_processor.configure_device(cmd)

    def enable_module(self, module_name):
        """Enable module

        Args:
            module_name (str): Module to be disabled (options: 'ENV', 'ORN', 'EXG')

        Examples:
            >>> from explorepy.explore import Explore
            >>> explore = Explore()
            >>> explore.connect(device_name='Explore_2FA2')
            >>> explore.enable_module('ORN')

        Returns:
            bool: True for success, False otherwise
        """
        if module_name not in ['ORN', 'ENV', 'EXG']:
            raise ValueError('Module name must be one of ORN, ENV or EXG.')
        self._check_connection()
        cmd = ModuleEnable(module_name)
        return self.stream_processor.configure_device(cmd)

    def calibrate_orn(self, do_overwrite=False):
        """Calibrate orientation module

        This method calibrates orientation sensors in order to get the real physical orientation in addition to raw
        sensor data. While running this function you would need to move and rotate the device. This function will store
        calibration info in the configuration file which will be used later during streaming to calculate physical
        orientation from raw sensor data.

        Args:
            do_overwrite: to overwrite the calibration data if already exists or not

        """
        assert not (PhysicalOrientation.check_calibre_data(device_name=self.device_name) and not do_overwrite), \
            "Calibration data already exists!"
        PhysicalOrientation.init_dir()
        logger.info("Start recording for 100 seconds, "
                    "please move the device around during this time, in all directions")
        file_name = user_cache_dir(appname="explorepy", appauthor="Mentalab") + '//temp_' + self.device_name
        self.record_data(file_name, do_overwrite=do_overwrite, duration=100, file_type='csv')
        time.sleep(105)
        PhysicalOrientation.calibrate(cache_dir=file_name, device_name=self.device_name)

    def _activate_test_sig(self, channel_mask):
        """ Activate the internal ADS test signals
        """
        channel_mask_int = self._convert_chan_mask(channel_mask)
        self._check_connection()
        cmd = SetChTest(channel_mask_int)
        self.stream_processor.configure_device(cmd)

    def _convert_chan_mask(self, mask):
        c = re.compile('[^01]')

        if (isinstance(mask, str) and len(c.findall(mask)) == 0) or (isinstance(mask, bytes)):
            return int(mask, 2)
        else:
            raise TypeError("Input must be bytes or binary string!")

    def _check_connection(self):
        assert self.is_connected, "Explore device is not connected. Please connect the device first."

    @staticmethod
    def _check_duration(duration):
        if duration:
            if duration <= 0:
                raise ValueError("Duration must be a positive number!")
        else:
            logger.warning("Duration has not been set by the user. The duration is 3 hours by default.")
            duration = 3 * 60 * 60  # 3 hours
        return duration
