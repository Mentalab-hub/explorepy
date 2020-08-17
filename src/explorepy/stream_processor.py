# -*- coding: utf-8 -*-
"""Stream Processor module
This module is responsible for processing incoming stream from Explore device and publishing data to subscribers.
"""
from enum import Enum
import time
import struct

from explorepy.parser import Parser
from explorepy.packet import DeviceInfo, CommandRCV, CommandStatus, EEG, Orientation, \
    Environment, EventMarker, CalibrationInfo
from explorepy.filters import ExGFilter
from explorepy.command import DeviceConfiguration, ZMeasurementEnable, ZMeasurementDisable
from explorepy.tools import ImpedanceMeasurement, PhysicalOrientation

TOPICS = Enum('Topics', 'raw_ExG filtered_ExG device_info marker raw_orn mapped_orn cmd_ack env cmd_status imp')


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

    def subscribe(self, callback, topic):
        """Subscribe a function to a topic

        Args:
            callback (function): Callback function to be called when there is a new packet in the topic
            topic (enum 'Topics'): Topic type
        """
        self.subscribers.setdefault(topic, set()).add(callback)

    def unsubscribe(self, callback, topic):
        """Unsubscribe a function from a topic

        Args:
            callback (function): Callback function to be called when there is a new packet in the topic
            topic (enum 'Topics'): Topic type
        """
        self.subscribers.setdefault(topic, set()).discard(callback)

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
        self.parser = Parser(callback=self.process, mode='file')
        self.is_connected = True
        self.parser.start_reading(filename=bin_file)

    def read_device_info(self, bin_file):
        self.parser = Parser(callback=self.process, mode='file')
        self.parser.read_device_info(bin_file)

    def stop(self):
        """Stop streaming"""
        self.parser.stop_streaming()
        self.is_connected = False

    def process(self, packet):
        """Process incoming packet

        Args:
            packet (explorepy.packet.Packet): Data packet
        """
        if isinstance(packet, Orientation):
            self.dispatch(topic=TOPICS.raw_orn, packet=packet)
            if self.physical_orn.status == "READY":
                packet = self.physical_orn.calculate(packet=packet)
                self.dispatch(topic=TOPICS.mapped_orn, packet=packet)
        elif isinstance(packet, EEG):
            self.dispatch(topic=TOPICS.raw_ExG, packet=packet)
            if self._is_imp_mode:
                packet_imp = self.imp_calculator.measure_imp(packet=packet)
                self.dispatch(topic=TOPICS.imp, packet=packet_imp)
            self.apply_filters(packet=packet)
            self.dispatch(topic=TOPICS.filtered_ExG, packet=packet)
        elif isinstance(packet, DeviceInfo):
            self.old_device_info = self.device_info
            self.device_info.update(packet.get_info())
            self.dispatch(topic=TOPICS.device_info, packet=packet)
        elif isinstance(packet, CommandRCV):
            self.dispatch(topic=TOPICS.cmd_ack, packet=packet)
        elif isinstance(packet, CommandStatus):
            self.dispatch(topic=TOPICS.cmd_status, packet=packet)
        elif isinstance(packet, Environment):
            self.dispatch(topic=TOPICS.env, packet=packet)
        elif isinstance(packet, EventMarker):
            self.dispatch(topic=TOPICS.marker, packet=packet)
        elif isinstance(packet, CalibrationInfo):
            self.imp_calib_info = packet.get_info()
        elif not packet:
            self.is_connected = False

    def dispatch(self, topic, packet):
        """Dispatch a packet to subscribers

        Args:
            topic (Enum 'Topics'): Topic enum which packet should be sent to
            packet (explorepy.packet.Packet): Data packet
        """
        if self.subscribers:
            for callback in self.subscribers[topic]:
                callback(packet)

    def add_filter(self, cutoff_freq, filter_type):
        """Add filter to the stream
        Args:
            cutoff_freq (Union[float, tuple]): Cut-off frequency (frequencies) for the filter
            filter_type (str): Filter type ['bandpass', 'lowpass', 'highpass', 'notch']
        """
        while not self.device_info:
            print('Waiting for device info packet...')
            time.sleep(.2)
        self.filters.append(ExGFilter(cutoff_freq=cutoff_freq,
                                      filter_type=filter_type,
                                      s_rate=self.device_info['sampling_rate'],
                                      n_chan=self.device_info['adc_mask'].count(1)))

    def apply_filters(self, packet):
        """Apply temporal filters to a packet"""
        for filt in self.filters:
            packet = filt.apply(packet)

    def configure_device(self, cmd):
        """Change device configuration

        Args:
            cmd (explorepy.command.Command): Command to be sent
        """
        if not self.is_connected:
            raise ConnectionError("No Explore device is connected!")
        return self._device_configurator.change_setting(cmd)

    def imp_initialize(self, notch_freq):
        """Activate impedance mode in the device"""
        cmd = ZMeasurementEnable()
        if self.configure_device(cmd):
            self._is_imp_mode = True
            self.imp_calculator = ImpedanceMeasurement(device_info=self.device_info,
                                                       calib_param=self.imp_calib_info,
                                                       notch_freq=notch_freq)
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
            print('Calibration data does not exist. If you need physical orientation, calibrate your device first.')

    def set_marker(self, code):
        """Set a marker in the stream"""
        if not isinstance(code, int):
            raise TypeError('Marker code must be an integer!')
        if 0 <= code <= 7:
            raise ValueError('Marker code value is not valid')

        self.process(EventMarker(timestamp=time.time() - self.parser.start_time,
                                 payload=bytearray(struct.pack('<H', code) + b'\xaf\xbe\xad\xde')))

    def compare_device_info(self, new_device_info):
        """Compare a device info dict with the current version

        Args:
            new_device_info (dict): Device info dictionary to be compared with the internal one

        Returns:
            bool: whether they are equal
        """
        assert self.device_info, "The internal device info has not been set yet!"
        if new_device_info['sampling_rate'] != self.old_device_info['sampling_rate']:
            print("Sampling rate has been changed in the file.")
            return False
        if new_device_info['adc_mask'] != self.old_device_info['adc_mask']:
            print("ADC mask has been changed in the file.")
            return False
        return True
