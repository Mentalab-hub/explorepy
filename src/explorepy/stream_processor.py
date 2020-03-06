# -*- coding: utf-8 -*-
"""Stream Processor module
This module is responsible for processing incoming stream from Explore device and publishing data to subscribers.
"""
from enum import Enum
import time
import struct

from explorepy.parser import Parser
from explorepy.packet import DeviceInfo, CommandRCV, CommandStatus, EEG, Orientation, Environment, EventMarker
from explorepy.filters import ExGFilter
from explorepy.command import DeviceConfiguration

TOPICS = Enum('Topics', 'raw_ExG filtered_ExG device_info marker raw_orn mapped_orn cmd_ack env cmd_status')


class StreamProcessor:
    """Stream processor class"""
    def __init__(self):
        self.parser = None
        self.filters = []
        self.orn_calibrator = None
        self.device_info = {}
        self.subscribers = {key: set() for key in TOPICS}  # keys are topics and values are sets of callbacks
        self._device_configurator = None
        self.is_connected = False

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
        self.parser = Parser(callback=self.process, mode='device')
        self.parser.start_streaming(device_name, mac_address)
        self.is_connected = True
        self._device_configurator = DeviceConfiguration(bt_interface=self.parser.stream_interface)
        self.subscribe(callback=self._device_configurator.update_ack, topic=TOPICS.cmd_ack)
        self.subscribe(callback=self._device_configurator.update_cmd_status, topic=TOPICS.cmd_status)

    def stop(self):
        """Stop streaming"""
        self.parser.stop_streaming()

    def process(self, packet):
        """Process incoming packet

        Args:
            packet (explorepy.packet.Packet): Data packet
        """
        if self.subscribers:
            if isinstance(packet, DeviceInfo):
                self.device_info = packet.get_info()
                self.dispatch(topic=TOPICS.device_info, packet=packet)
            elif isinstance(packet, CommandRCV):
                self.dispatch(topic=TOPICS.cmd_ack, packet=packet)
            elif isinstance(packet, CommandStatus):
                self.dispatch(topic=TOPICS.cmd_status, packet=packet)
            elif isinstance(packet, EEG):
                self.dispatch(topic=TOPICS.raw_ExG, packet=packet)
                self.apply_filters(packet=packet)
                self.dispatch(topic=TOPICS.filtered_ExG, packet=packet)
            elif isinstance(packet, Orientation):
                self.dispatch(topic=TOPICS.raw_orn, packet=packet)
                self.calculate_phys_orn(packet=packet)
            elif isinstance(packet, Environment):
                self.dispatch(topic=TOPICS.env, packet=packet)
            elif isinstance(packet, EventMarker):
                self.dispatch(topic=TOPICS.marker, packet=packet)

    def dispatch(self, topic, packet):
        """Dispatch a packet to subscribers

        Args:
            topic (Enum 'Topics'): Topic enum which packet should be sent to
            packet (explorepy.packet.Packet): Data packet
        """
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
            time.sleep(.5)
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
        self._device_configurator.change_setting(cmd)

    def calculate_phys_orn(self, packet):
        """Calculate physical orientation"""
        raise NotImplementedError

    def set_marker(self, code):
        """Set a marker in the stream"""
        if type(code) is not int:
            raise TypeError('Marker code must be an integer!')
        if 0 <= code <= 7:
            raise ValueError('Marker code value is not valid')

        self.process(EventMarker(timestamp=time.time() - self.parser.start_time,
                                 payload=bytearray(struct.pack('<H', code) + b'\xaf\xbe\xad\xde')))
