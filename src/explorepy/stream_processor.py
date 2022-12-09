# -*- coding: utf-8 -*-
"""Stream Processor module
This module is responsible for processing incoming stream from Explore device and publishing data to subscribers.
"""
import logging
import time
from enum import Enum
from threading import Lock

from explorepy.command import (
    DeviceConfiguration,
    ZMeasurementDisable,
    ZMeasurementEnable
)
from explorepy.filters import ExGFilter
from explorepy.packet import (
    EEG,
    CalibrationInfo,
    CalibrationInfo_USBC,
    CommandRCV,
    CommandStatus,
    DeviceInfo,
    DeviceInfoV2,
    Environment,
    EventMarker,
    Orientation,
    SoftwareMarker
)
from explorepy.parser import Parser
from explorepy.settings_manager import SettingsManager
from explorepy.tools import (
    ImpedanceMeasurement,
    PhysicalOrientation,
    get_local_time
)


TOPICS = Enum('Topics', 'raw_ExG filtered_ExG device_info marker raw_orn mapped_orn cmd_ack env cmd_status imp')
logger = logging.getLogger(__name__)
lock = Lock()


class StreamProcessor:
    """Stream processor class"""

    def __init__(self):
        self.parser = None
        self.filters = []
        self.orn_calibrator = None
        self.device_info = {}
        self.old_device_info = {}
        self.imp_calib_info = {}
        self.subscribers = {key: set() for key in TOPICS}  # keys are topics and values are sets of callbacks
        self._device_configurator = None
        self.imp_calculator = None
        self.is_connected = False
        self._is_imp_mode = False
        self.physical_orn = PhysicalOrientation()
        self._last_packet_timestamp = 0
        self._last_packet_rcv_time = 0
        self.is_bt_streaming = True

    def subscribe(self, callback, topic):
        """Subscribe a function to a topic

        Args:
            callback (function): Callback function to be called when there is a new packet in the topic
            topic (enum 'Topics'): Topic type
        """
        with lock:
            logger.debug(f"Subscribe {callback.__name__} to {topic}")
            self.subscribers[topic].add(callback)

    def unsubscribe(self, callback, topic):
        """Unsubscribe a function from a topic

        Args:
            callback (function): Callback function to be called when there is a new packet in the topic
            topic (enum 'Topics'): Topic type
        """
        with lock:
            logger.debug(f"Unsubscribe {callback} from {topic}")
            self.subscribers[topic].discard(callback)

    def start(self, device_name=None, mac_address=None):
        """Start streaming from Explore device

        Args:
            device_name (str): Explore device name in form of <Explore_####>
            mac_address (str): MAC address of Explore device
        """
        if device_name is None:
            device_name = "Explore_" + str(mac_address[-5:-3]) + str(mac_address[-2:])
        self.device_info["device_name"] = device_name
        self.parser = Parser(callback=self.process, mode='device')
        self.parser.start_streaming(device_name, mac_address)
        self.is_connected = True
        self._device_configurator = DeviceConfiguration(bt_interface=self.parser.stream_interface)
        self.subscribe(callback=self._device_configurator.update_ack, topic=TOPICS.cmd_ack)
        self.subscribe(callback=self._device_configurator.update_cmd_status, topic=TOPICS.cmd_status)
        self.orn_initialize(device_name)

    def open_file(self, bin_file):
        """Open the binary file and read until it gets device info packet
        Args:
            bin_file (str): Path to binary file
        """
        self.is_bt_streaming = False
        self.parser = Parser(callback=self.process, mode='file')
        self.is_connected = True
        self.parser.start_reading(filename=bin_file)

    def read_device_info(self, bin_file):
        self.is_bt_streaming = False
        self.parser = Parser(callback=self.process, mode='file')
        self.parser.read_device_info(bin_file)

    def stop(self):
        """Stop streaming"""
        self.is_connected = False
        self.parser.stop_streaming()

    def process(self, packet):
        """Process incoming packet

        Args:
            packet (explorepy.packet.Packet): Data packet
        """
        received_time = get_local_time()
        if isinstance(packet, Orientation):
            self.dispatch(topic=TOPICS.raw_orn, packet=packet)
            if self.physical_orn.status == "READY":
                packet = self.physical_orn.calculate(packet=packet)
                self.dispatch(topic=TOPICS.mapped_orn, packet=packet)
        elif isinstance(packet, EEG):
            self._update_last_time_point(packet, received_time)
            self.dispatch(topic=TOPICS.raw_ExG, packet=packet)
            if self._is_imp_mode and self.imp_calculator:
                packet_imp = self.imp_calculator.measure_imp(packet=packet)
                self.dispatch(topic=TOPICS.imp, packet=packet_imp)
            self.apply_filters(packet=packet)
            self.dispatch(topic=TOPICS.filtered_ExG, packet=packet)
        elif isinstance(packet, DeviceInfo) or isinstance(packet, DeviceInfoV2):
            self.old_device_info = self.device_info.copy()
            self.device_info.update(packet.get_info())
            if self.is_bt_streaming:
                settings_manager = SettingsManager(self.device_info["device_name"])
                settings_manager.update_device_settings(packet.get_info())
            self.dispatch(topic=TOPICS.device_info, packet=packet)
        elif isinstance(packet, CommandRCV):
            self.dispatch(topic=TOPICS.cmd_ack, packet=packet)
        elif isinstance(packet, CommandStatus):
            self.dispatch(topic=TOPICS.cmd_status, packet=packet)
        elif isinstance(packet, Environment):
            self.dispatch(topic=TOPICS.env, packet=packet)
        elif isinstance(packet, EventMarker):
            self.dispatch(topic=TOPICS.marker, packet=packet)
        elif isinstance(packet, CalibrationInfo) or isinstance(packet, CalibrationInfo_USBC):
            self.imp_calib_info = packet.get_info()
        elif not packet:
            self.is_connected = False

    def _update_last_time_point(self, packet, received_time):
        """Update the last PCB time point and the local time the last packet is received.

        The goal is to keep track of the PCB clock (i.e. the PCB timestamp of the last packet that has been received)
        and the local (client) time it has been received by the computer.

        Args:
            packet (explorepy.packet.EEG): ExG data packet
            received_time (float): Local time of receiving the packet
        """
        if 'sampling_rate' in self.device_info:
            timestamp, _ = packet.get_data(exg_fs=self.device_info['sampling_rate'])
            timestamp = timestamp[-1]
            with lock:
                if timestamp > self._last_packet_timestamp:
                    self._last_packet_timestamp = timestamp
                    self._last_packet_rcv_time = received_time

    def _get_sw_marker_time(self):
        """Returns a timestamp to be used in software marker

        This method gives an estimation of a timestamp relative to Explore internal clock. This timestamp can be used
        for generating a software marker.
        """
        with lock:
            return self._last_packet_timestamp + get_local_time() - self._last_packet_rcv_time

    def dispatch(self, topic, packet):
        """Dispatch a packet to subscribers

        Args:
            topic (Enum 'Topics'): Topic enum which packet should be sent to
            packet (explorepy.packet.Packet): Data packet
        """
        if self.subscribers:
            with lock:
                for callback in self.subscribers[topic]:
                    callback(packet)

    def add_filter(self, cutoff_freq, filter_type):
        """Add filter to the stream
        Args:
            cutoff_freq (Union[float, tuple]): Cut-off frequency (frequencies) for the filter
            filter_type (str): Filter type ['bandpass', 'lowpass', 'highpass', 'notch']
        """
        logger.info(f"Adding a {filter_type} filter with cut-off freqs of {cutoff_freq}.")
        while not self.device_info:
            logger.warning('No device info is available. Waiting for device info packet...')
            time.sleep(.2)

        settings_manager = SettingsManager(self.device_info["device_name"])
        settings_manager.load_current_settings()
        n_chan = settings_manager.settings_dict[settings_manager.channel_count_key]

        self.filters.append(ExGFilter(cutoff_freq=cutoff_freq,
                                      filter_type=filter_type,
                                      s_rate=self.device_info['sampling_rate'],
                                      n_chan=n_chan))

    def remove_filters(self):
        """
        Remove all filters from the stream
        """
        logger.info("Removing all filters.")
        while not self.device_info:
            logger.warning('No device info is available. Waiting for device info packet...')
            time.sleep(.2)
        self.filters = []

    def apply_filters(self, packet):
        """Apply temporal filters to a packet"""
        for filt in self.filters:
            packet = filt.apply(packet)

    def configure_device(self, cmd):
        """Change device configuration

        Args:
            cmd (explorepy.command.Command): Command to be sent

        Returns:
            bool: True for success, False otherwise.
        """
        if not self.is_connected:
            raise ConnectionError("No Explore device is connected!")
        return self._device_configurator.change_setting(cmd)

    def imp_initialize(self, notch_freq):
        """Activate impedance mode in the device"""
        logger.info("Starting impedance measurement mode...")
        cmd = ZMeasurementEnable()
        if self.configure_device(cmd):
            self.imp_calculator = ImpedanceMeasurement(device_info=self.device_info,
                                                       calib_param=self.imp_calib_info,
                                                       notch_freq=notch_freq)
            self._is_imp_mode = True
        else:
            raise ConnectionError('Device configuration process failed!')

    def disable_imp(self):
        """Disable impedance mode in the device"""
        cmd = ZMeasurementDisable()
        if self.configure_device(cmd):
            self._is_imp_mode = False
            print("Impedance measurement mode has been disabled.")
            return True
        print("WARNING: Couldn't disable impedance measurement mode. "
              "Please restart your device manually.")
        return False

    def orn_initialize(self, device_name):
        res = self.physical_orn.read_calibre_data(device_name)
        if res:
            self.physical_orn.status = "READY"
        else:
            self.physical_orn.status = "NOT READY"
            logger.debug('Calibration coefficients for physical orientation do not exist. '
                         'If you need physical orientation, calibrate the device first.')

    def set_marker(self, code):
        """Set a marker in the stream"""
        logger.info(f"Setting a software marker with code: {code}")
        if not isinstance(code, int):
            raise TypeError('Marker code must be an integer!')
        if not 0 <= code <= 65535:
            raise ValueError('Marker code value is not valid! Code must be in range of 0-65535.')

        marker = SoftwareMarker.create(self._get_sw_marker_time(), code)
        self.process(marker)

    def compare_device_info(self, new_device_info):
        """Compare a device info dict with the current version

        Args:
            new_device_info (dict): Device info dictionary to be compared with the internal one

        Returns:
            bool: whether they are equal
        """
        assert self.device_info, "The internal device info has not been set yet!"
        if new_device_info['sampling_rate'] != self.old_device_info['sampling_rate']:
            logger.info(f"Sampling rate has been changed to {new_device_info['sampling_rate']} in the file.")
            return False
        if new_device_info['adc_mask'] != self.old_device_info['adc_mask']:
            print(f"ADC mask has been changed to {new_device_info['adc_mask']} in the file.")
            return False
        return True

    def send_timestamp(self):
        """Send host timestamp to the device"""
        self._device_configurator.send_timestamp()
