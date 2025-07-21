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
from threading import Timer

import numpy as np

import explorepy
from explorepy.command import (
    MemoryFormat,
    SetChTest,
    SetSPS,
    SoftReset
)
from explorepy.debug import Debug
from explorepy.settings_manager import SettingsManager
from explorepy.stream_processor import (
    TOPICS,
    StreamProcessor
)
from explorepy.tools import (
    LslServer,
    check_bin_compatibility,
    create_exg_recorder,
    create_marker_recorder,
    create_meta_recorder,
    create_orn_recorder,
    get_orn_chan_len,
    local_clock,
    setup_usb_marker_port
)


logger = logging.getLogger(__name__)


class Explore:
    r"""Mentalab Explore device"""

    def __init__(self, debug=False, debug_settings=None):
        self.debug = Debug(settings=debug_settings) if debug else None
        self.is_connected = False
        self.stream_processor = None
        self.recorders = {}
        self.lsl = {}
        self.device_name = None
        self.initial_count = None
        self.last_rec_stat = 0
        self.last_rec_start_time = 0

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
            self.device_name = 'Explore_' + \
                mac_address[-5:-3] + mac_address[-2:]
        logger.info(f"Connecting to {self.device_name} ...")
        self.stream_processor = StreamProcessor(
            debug=True if self.debug else False)
        self.stream_processor.start(
            device_name=device_name, mac_address=mac_address)
        cnt = 0
        cnt_limit = 20 if self.debug else 15
        while "adc_mask" not in self.stream_processor.device_info:
            logger.info("Waiting for device info packet...")
            time.sleep(1)
            if cnt >= cnt_limit:
                raise ConnectionAbortedError(
                    "Could not get info packet from the device")
            cnt += 1
        # check if fw_version follows x.x.x pattern, abort connection otherwise
        if len(str.split(self.stream_processor.device_info['firmware_version'], '.')) > 3:
            self.stream_processor.stop()
            raise ConnectionAbortedError('This device is not supported. Please use a compatible device')
        if self.stream_processor.device_info['is_imp_mode'] is True:
            self.stream_processor.disable_imp()
        logger.info(
            'Device info packet has been received. Connection has been established. Streaming...')
        logger.info("Device info: " + str(self.stream_processor.device_info))
        self.is_connected = True
        if self.debug:
            self.stream_processor.subscribe(
                callback=self.debug.process_bin, topic=TOPICS.packet_bin)

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

        self.stream_processor.subscribe(
            callback=callback, topic=TOPICS.raw_ExG)
        logger.debug(f"Acquiring and printing data stream for {duration}s ...")
        time.sleep(duration)
        self.stream_processor.unsubscribe(
            callback=callback, topic=TOPICS.raw_ExG)

    def record_data(
        self, file_name, do_overwrite=False, duration=None, file_type='csv', block=False, exg_ch_names=None
    ):
        r"""Records the data in real-time

        Args:
            file_name (str): Output file name
            do_overwrite (bool): Overwrite if files exist already
            duration (float): Duration of recording in seconds (if None records for 3 hours).
            file_type (str): File type of the recorded file. Supported file types: 'csv', 'edf'
            block (bool): Record in blocking mode if 'block' is True
            exg_ch_names (list): list of channel names. If None, default names are used.
        """
        self._check_connection()

        # Check invalid characters
        if set(r'<>{}[]~`*%').intersection(file_name):
            raise ValueError("Invalid character in file name")
        if file_type not in ['edf', 'csv']:
            raise ValueError(
                '{} is not a supported file extension!'.format(file_type))
        duration = self._check_duration(duration)

        exg_out_file = file_name + "_ExG"
        orn_out_file = file_name + "_ORN"
        marker_out_file = file_name + "_Marker"
        meta_out_file = file_name + "_Meta"

        self.recorders['exg'] = create_exg_recorder(filename=exg_out_file,
                                                    file_type=file_type,
                                                    fs=self.stream_processor.device_info['sampling_rate'],
                                                    adc_mask=SettingsManager(
                                                        self.device_name).get_adc_mask(),
                                                    do_overwrite=do_overwrite,
                                                    exg_ch=exg_ch_names)
        self.recorders['orn'] = create_orn_recorder(filename=orn_out_file,
                                                    file_type=file_type,
                                                    do_overwrite=do_overwrite,
                                                    n_chan=get_orn_chan_len(self.stream_processor.device_info))

        #  TODO: make sure older timestamp in meta file was not used in any other software!
        if file_type == 'csv':
            self.recorders['marker'] = create_marker_recorder(
                filename=marker_out_file, do_overwrite=do_overwrite)
            self.recorders['meta'] = create_meta_recorder(filename=meta_out_file,
                                                          fs=self.stream_processor.device_info['sampling_rate'],
                                                          adc_mask=SettingsManager(
                                                              self.device_name).get_adc_mask(),
                                                          device_name=self.device_name,
                                                          do_overwrite=do_overwrite,
                                                          timestamp=str(self.stream_processor.parser._time_offset))  # noqa: E501
            self.recorders['meta'].write_meta()
            self.recorders['meta'].stop()

        elif file_type == 'edf':
            self.recorders['marker'] = self.recorders['exg']
            logger.warning("Markers' timing might not be precise in EDF files. We recommend recording in CSV format "
                           "if you are setting markers during the recording.")

        self.stream_processor.subscribe(
            callback=self.recorders['exg'].write_data, topic=TOPICS.raw_ExG)
        self.stream_processor.subscribe(
            callback=self.recorders['orn'].write_data, topic=TOPICS.raw_orn)
        self.stream_processor.subscribe(
            callback=self.recorders['marker'].set_marker, topic=TOPICS.marker)
        logger.info("Recording...")

        self.recorders['timer'] = Timer(duration, self.stop_recording)
        self.last_rec_start_time = local_clock()
        self.initial_count = self.stream_processor.packet_count
        self.recorders['timer'].start()
        if block:
            try:
                while 'timer' in self.recorders.keys() and self.recorders['timer'].is_alive():
                    time.sleep(.3)
            except KeyboardInterrupt:
                logger.info(
                    "Got Keyboard Interrupt while recording in blocked mode!")
                self.stop_recording()
                self.stream_processor.stop()
                time.sleep(1)

    def stop_recording(self):
        """Stop recording"""
        if self.recorders:
            self.stream_processor.unsubscribe(
                callback=self.recorders['exg'].write_data, topic=TOPICS.raw_ExG)
            self.stream_processor.unsubscribe(
                callback=self.recorders['orn'].write_data, topic=TOPICS.raw_orn)
            self.stream_processor.unsubscribe(
                callback=self.recorders['marker'].set_marker, topic=TOPICS.marker)
            self.recorders['exg'].stop()
            self.recorders['orn'].stop()
            if self.recorders['exg'].file_type == 'csv':
                self.recorders['marker'].stop()
            if 'timer' in self.recorders.keys() and self.recorders['timer'].is_alive():
                self.recorders['timer'].cancel()
            self.recorders = {}
            logger.info('Recording stopped.')
            try:
                self.last_rec_stat = (
                    (self.stream_processor.packet_count - self.initial_count) / (
                        (local_clock() - self.last_rec_start_time)
                        * self.stream_processor.device_info['sampling_rate']
                    )
                )
                # clamp the stat variable
                self.last_rec_stat = max(1, min(self.last_rec_stat, 1))
                logger.info('last recording stat : {}'.format(
                    self.last_rec_stat))
            except TypeError:
                # handle uninitialized state
                pass
            self.initial_count = None
        else:
            logger.debug(
                "Tried to stop recording while no recorder is running!")

    def get_last_record_stat(self):
        """Gets the last recording statistics as a number between 0 and 1"""
        return self.last_rec_stat

    def convert_bin(self, bin_file, out_dir='', file_type='edf', do_overwrite=False, out_dir_is_full=False,
                    progress_callback=None, progress_dialog=None):
        """Convert a binary file to EDF(BDF+) or CSV file

        Args:
            bin_file (str): Path to the binary file recorded by Explore device
            out_dir (str): Output directory path (must be relative path to the current working directory)
            file_type (str): Output file type: 'edf' for EDF(BDF+) format and 'csv' for CSV format
            do_overwrite (bool): Whether to overwrite an existing file
            out_dir_is_full(bool): Whether output directory is a full file path
            progress_callback: Callback to get progress update
            progress_dialog: Dialog instance

        """
        check_bin_compatibility(bin_file)
        bt_interface = explorepy.get_bt_interface()
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
                self.mask = [1 for _ in range(0, 32)]
            if 'PCB_305_801_XXX' in self.stream_processor.device_info['board_id']:
                self.mask = [1 for _ in range(0, 16)]
            if 'PCB_304_801p2_X' in self.stream_processor.device_info['board_id']:
                self.mask = [1 for _ in range(0, 32)]
            if 'PCB_304_891p2_X' in self.stream_processor.device_info['board_id']:
                self.mask = [1 for _ in range(0, 16)]

        self.recorders['exg'] = create_exg_recorder(filename=exg_out_file,
                                                    file_type=self.recorders['file_type'],
                                                    fs=self.stream_processor.device_info['sampling_rate'],
                                                    adc_mask=self.mask,
                                                    do_overwrite=do_overwrite, batch_mode=True)
        self.recorders['orn'] = create_orn_recorder(filename=orn_out_file,
                                                    file_type=self.recorders['file_type'],
                                                    do_overwrite=do_overwrite, batch_mode=True,
                                                    n_chan=get_orn_chan_len(self.stream_processor.device_info))

        if self.recorders['file_type'] == 'csv':
            self.recorders['marker'] = create_marker_recorder(
                filename=marker_out_file, do_overwrite=do_overwrite, batch_mode=True)
            self.recorders['meta'] = create_meta_recorder(filename=meta_out_file,
                                                          fs=self.stream_processor.device_info['sampling_rate'],
                                                          adc_mask=self.mask,
                                                          device_name=self.device_name,
                                                          do_overwrite=do_overwrite, batch_mode=True)
            self.recorders['meta'].write_meta()
            self.recorders['meta'].stop()
        else:
            self.recorders['marker'] = self.recorders['exg']

        self.stream_processor.subscribe(
            callback=self.recorders['exg'].write_data, topic=TOPICS.raw_ExG)
        self.stream_processor.subscribe(
            callback=self.recorders['orn'].write_data, topic=TOPICS.raw_orn)
        self.stream_processor.subscribe(
            callback=self.recorders['marker'].set_marker, topic=TOPICS.marker)

        def device_info_callback(packet):
            new_device_info = packet.get_info()
            # TODO add 16 channel board id and refactor
            if not self.stream_processor.compare_device_info(new_device_info):
                new_file_name = exg_out_file[:-4] + "_" + \
                    str(np.round(packet.timestamp, 0)) + '_ExG'
                new_meta_name = meta_out_file[:-4] + "_" + \
                    str(np.round(packet.timestamp, 0)) + '_Meta'
                logger.warning("Creating a new file: "
                               + new_file_name + '.' + self.recorders['file_type'])
                self.stream_processor.unsubscribe(
                    callback=self.recorders['exg'].write_data, topic=TOPICS.raw_ExG)
                self.stream_processor.unsubscribe(
                    callback=self.recorders['marker'].set_marker, topic=TOPICS.marker)
                self.recorders['exg'].stop()
                self.recorders['exg'] = create_exg_recorder(filename=new_file_name,
                                                            file_type=self.recorders['file_type'],
                                                            fs=self.stream_processor.device_info['sampling_rate'],
                                                            adc_mask=self.stream_processor.device_info['adc_mask'],
                                                            do_overwrite=do_overwrite, batch_mode=True)

                if self.recorders['file_type'] == 'edf':
                    self.recorders['marker'] = self.recorders['exg']

                self.stream_processor.subscribe(
                    callback=self.recorders['exg'].write_data, topic=TOPICS.raw_ExG)
                self.stream_processor.subscribe(
                    callback=self.recorders['marker'].set_marker, topic=TOPICS.marker)

                if self.recorders['file_type'] == 'csv':
                    self.recorders['meta'] = create_meta_recorder(
                        filename=new_meta_name,
                        fs=self.stream_processor.device_info['sampling_rate'],
                        adc_mask=self.stream_processor.device_info['adc_mask'],
                        device_name=self.device_name,
                        do_overwrite=do_overwrite, batch_mode=True)
                    self.recorders['meta'].write_meta()
                    self.recorders['meta'].stop()

        self.stream_processor.subscribe(
            callback=device_info_callback, topic=TOPICS.device_info)

        def stream_progress_handler(progress):
            progress = np.floor(progress)
            if progress_callback:
                progress_callback(progress)
            if progress_dialog:
                if progress >= 100:
                    progress_dialog._close()
                    return
                elif progress_dialog.close:
                    raise InterruptedError("Conversion cancelled by user")
            else:
                bar_width = 50
                filled = int(bar_width * progress / 100)
                bar = '=' * filled + '-' * (bar_width - filled)
                print(f'\rProgress: [{bar}] {progress}%', end='', flush=True)
                if progress >= 100:
                    print()

        logger.info("Converting...")
        try:
            self.stream_processor.open_file(bin_file, progress_callback=stream_progress_handler)

        except InterruptedError:
            logger.info("Conversion process interrupted.")
        finally:
            if self.recorders['file_type'] == 'csv':
                self.stream_processor.unsubscribe(callback=self.recorders['marker'].set_marker, topic=TOPICS.marker)
                self.recorders["marker"].stop()
            self.stream_processor.unsubscribe(callback=self.recorders['exg'].write_data, topic=TOPICS.raw_ExG)
            self.stream_processor.unsubscribe(callback=self.recorders['orn'].write_data, topic=TOPICS.raw_orn)
            self.recorders["exg"].stop()
            self.recorders["orn"].stop()
            self.recorders = {}
            explorepy.set_bt_interface(bt_interface)
            logger.info('Conversion process terminated.')

    def push2lsl(self, duration=None, block=False):
        """Push samples to three lsl streams (ExG, Marker and ORN streams)

        Args:
            duration (float): duration of data acquiring (if None it streams for three hours).
            block (bool): blocking mode
        """
        self._check_connection()
        duration = self._check_duration(duration)

        self.lsl['timer'] = Timer(duration, self.stop_lsl)
        self.lsl['server'] = LslServer(self.stream_processor.device_info)
        self.stream_processor.subscribe(
            topic=TOPICS.raw_ExG, callback=self.lsl['server'].push_exg)
        self.stream_processor.subscribe(
            topic=TOPICS.raw_orn, callback=self.lsl['server'].push_orn)
        self.stream_processor.subscribe(
            topic=TOPICS.marker, callback=self.lsl['server'].push_marker)
        self.lsl['timer'].start()

        if block:
            try:
                while 'timer' in self.lsl.keys() and self.lsl['timer'].is_alive():
                    time.sleep(.3)
            except KeyboardInterrupt:
                logger.info(
                    "Got Keyboard Interrupt while pushing data to LSL in blocked mode!")
                self.stream_processor.stop()
                self.stop_lsl()
                time.sleep(1)

    def stop_lsl(self):
        """Stop pushing data to LSL streams"""
        if self.lsl:
            self.stream_processor.unsubscribe(
                topic=TOPICS.raw_ExG, callback=self.lsl['server'].push_exg)
            self.stream_processor.unsubscribe(
                topic=TOPICS.raw_orn, callback=self.lsl['server'].push_orn)
            self.stream_processor.unsubscribe(
                topic=TOPICS.marker, callback=self.lsl['server'].push_marker)
            if self.lsl['timer'].is_alive():
                self.lsl['timer'].cancel()
            self.lsl = {}

            logger.info("Push2lsl has been stopped.")
        else:
            logger.debug("Tried to stop LSL while no LSL server is running!")

    def set_marker(self, marker_string, time_lsl=None):
        """Sets a digital event marker while streaming

        Args:
            time_lsl (timestamp): timestamp from external marker)
            marker_string (string): string to save as experiment marker)
        """
        self._check_connection()
        self.stream_processor.set_ext_marker(marker_string=str(marker_string))

    def send_8_bit_trigger(self, eight_bit_value):
        eight_bit_value = eight_bit_value % 256
        trigger_id = 0xAB
        cmd = [trigger_id, eight_bit_value, 1, 2, 3, 4, 5, 6, 7, 8, 0xAF, 0xBE, 0xAD, 0xDE]
        explore_port = setup_usb_marker_port()
        explore_port.write(bytearray(cmd))
        explore_port.close()

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
        if sampling_rate not in [250, 500, 1000, 2000, 4000, 8000, 16000]:
            raise ValueError(
                "Sampling rate must be 250, 500, 2000, 4000, 8000 or 16000.")
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
        # sets virtual channel mask for all device variants
        SettingsManager(self.device_name).set_adc_mask(channel_mask)
        return True

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
            logger.warning(
                "Duration has not been set by the user. The duration is 3 hours by default.")
            duration = 3 * 60 * 60  # 3 hours
        return duration

    def is_explore_plus_device(self):
        return True if 'board_id' in self.stream_processor.device_info.keys() else False

    def is_bt_link_unstable(self):
        if not self.stream_processor:
            return False
        else:
            return self.stream_processor.is_connection_unstable()

    def get_channel_mask(self):
        return SettingsManager(self.device_name).get_adc_mask()
