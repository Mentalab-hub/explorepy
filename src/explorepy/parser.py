# -*- coding: utf-8 -*-
"""Parser module"""
from threading import Thread
import time
import struct

from explorepy.packet import PACKET_CLASS_DICT
from explorepy.bt_client import BtClient


class Parser:
    """Data parser class"""
    def __init__(self, callback, mode='device'):
        """Parser class for explore device

        Args:
            callback (function): function to be called when new packet is received
            mode (str): Parsing mode either from an Explore device or a binary file {'device', 'file'}
        """
        self.mode = mode
        self.stream_interface = None
        self.device_configurator = None
        self.callback = callback

        self._time_offset = None
        self._start_time = None
        self._do_streaming = False
        self._stream_thread = None

    def start_streaming(self, device_name, mac_address):
        """Start streaming data from Explore device"""
        self.stream_interface = BtClient(device_name=device_name, mac_address=mac_address)
        self.stream_interface.connect()
        self._stream()

    def stop_streaming(self):
        """Stop streaming data"""
        self._do_streaming = False
        self._stream_thread.join(timeout=1)
        self.stream_interface.disconnect()

    def open_file(self):
        """Open the binary file"""
        self.stream_interface = FileHandler()

    def _stream(self):
        self._do_streaming = True
        self._stream_thread = Thread(target=self._stream_loop)
        self._stream_thread.setDaemon(True)
        self._stream_thread.start()

    def _stream_loop(self):
        import asyncio
        asyncio.set_event_loop(asyncio.new_event_loop())
        while self._do_streaming:
            try:
                packet = self._generate_packet()
                self.callback(packet=packet)
            except ConnectionAbortedError:
                print("Device has been disconnected! Scanning for the last connected device...")
                self.stream_interface.reconnect()

    def _generate_packet(self):
        """Reads and parses a package from a file or socket

        Args:

        Returns:
            packet object
        """
        pid = struct.unpack('B', self.stream_interface.read(1))[0]
        cnt = self.stream_interface.read(1)[0]
        payload = struct.unpack('<H', self.stream_interface.read(2))[0]
        timestamp = struct.unpack('<I', self.stream_interface.read(4))[0]

        # Timestamp conversion
        if self._time_offset is None:
            self._time_offset = timestamp * .0001
            timestamp = 0
            self._start_time = time.time()
        else:
            timestamp = timestamp * .0001 - self._time_offset   # Timestamp unit is .1 ms

        payload_data = self.stream_interface.read(payload - 4)
        packet = self._parse_packet(pid, timestamp, payload_data)
        return packet

    @staticmethod
    def _parse_packet(pid, timestamp, bin_data):
        """Generates the packets according to the pid

        Args:
            pid (int): Packet ID
            timestamp (int): Timestamp
            bin_data: Binary dat

        Returns:
            Packet
        """

        if pid in PACKET_CLASS_DICT:
            packet = PACKET_CLASS_DICT[pid](timestamp, bin_data)
        else:
            print("Unknown Packet ID:" + str(pid))
            print("Length of the binary data:", len(bin_data))
            packet = None
        return packet


class DeviceConfigurator:
    """Explore device configurator class"""
    def __init__(self):
        pass


class FileHandler:
    """Binary file handler"""
    def __init__(self):
        pass
