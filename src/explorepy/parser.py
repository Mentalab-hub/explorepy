# -*- coding: utf-8 -*-
"""Parser module"""
import asyncio
import logging
import struct
from threading import Thread

import explorepy
from explorepy._exceptions import FletcherError
from explorepy.packet import (
    PACKET_CLASS_DICT,
    TIMESTAMP_SCALE,
    DeviceInfo,
    DeviceInfoV2
)
from explorepy.tools import get_local_time


logger = logging.getLogger(__name__)


class Parser:
    """Data parser class"""
    def __init__(self, callback, mode='device'):
        """
        Args:
            callback (function): function to be called when new packet is received
            mode (str): Parsing mode either from an Explore device or a binary file {'device', 'file'}
        """
        self.mode = mode
        self.stream_interface = None
        self.device_configurator = None
        self.callback = callback

        if self.mode == 'file':
            self._time_offset = 0
        else:
            self._time_offset = None

        self._do_streaming = False
        self.is_waiting = False
        self._stream_thread = None
        self._is_reconnecting = False

    def start_streaming(self, device_name, mac_address):
        """Start streaming data from Explore device"""
        if explorepy.get_bt_interface() == 'sdk':
            from explorepy.btcpp import SDKBtClient
            self.stream_interface = SDKBtClient(device_name=device_name, mac_address=mac_address)
        else:
            raise ValueError("Invalid Bluetooth interface: " + explorepy.get_bt_interface())
        self.stream_interface.connect()
        self._stream()

    def stop_streaming(self):
        """Stop streaming data"""
        if self._do_streaming:
            self._do_streaming = False
            self.callback(None)
            self.stream_interface.disconnect()

    def start_reading(self, filename):
        """Open the binary file
        Args:
            filename (str): Binary file name
        """
        self.stream_interface = FileHandler(filename)
        self._stream(new_thread=True)

    def read_device_info(self, filename):
        self.stream_interface = FileHandler(filename)
        packet = None
        while not (isinstance(packet, DeviceInfo) or isinstance(packet, DeviceInfoV2)):
            try:
                packet = self._generate_packet()
                self.callback(packet=packet)
            except (IOError, ValueError, FletcherError) as error:
                logger.error('Conversion ended incomplete. The binary file is corrupted.')
                raise error
            except EOFError:
                logger.info('Reached end of the file')
            finally:
                self.stream_interface.disconnect()

    def _stream(self, new_thread=True):
        self._do_streaming = True
        if new_thread:
            logger.debug("Creating a new thread for streaming.")
            self._stream_thread = Thread(name="ParserThread", target=self._stream_loop)
            self._stream_thread.setDaemon(True)
            self._stream_thread.start()
        else:
            self._stream_loop()

    def _stream_loop(self):
        asyncio.set_event_loop(asyncio.new_event_loop())
        while self._do_streaming:
            try:
                packet = self._generate_packet()
                self.callback(packet=packet)
            except ConnectionAbortedError as error:
                logger.debug(f"Got this error while streaming: {error}")
                logger.warning("Device has been disconnected! Scanning for the last connected device...")
                self._is_reconnecting = True
                if self.stream_interface.reconnect() is None:
                    logger.warning("Could not find the device! "
                                   "Please make sure the device is on and in advertising mode.")
                    self.stop_streaming()
                    print("Press Ctrl+c to exit...")
                self._is_reconnecting = False
            except (IOError, ValueError, MemoryError, FletcherError) as error:
                logger.debug(f"Got this error while streaming: {error}")
                if self.mode == 'device':
                    if str(error) != 'connection has been closed':
                        logger.error('Bluetooth connection error! Make sure your device is on and in advertising mode.')
                        self.stop_streaming()
                        print("Press Ctrl+c to exit...")
                        raise error
                else:
                    logger.warning('The binary file is corrupted. Conversion has ended incompletely.')
                self.stop_streaming()
            except EOFError:
                logger.info('End of file')
                self.stop_streaming()
            except Exception as error:
                logger.critical('Unexpected error: ', error)
                self.stop_streaming()
                raise error

    def _generate_packet(self):
        """Reads and parses a package from a file or socket

        Returns:
            packet object
        """
        pid = struct.unpack('B', self.stream_interface.read(1))[0]
        self.stream_interface.read(1)[0]  # read cnt
        payload = struct.unpack('<H', self.stream_interface.read(2))[0]
        timestamp = struct.unpack('<I', self.stream_interface.read(4))[0]

        # Timestamp conversion
        if self._time_offset is None:
            self._time_offset = get_local_time() - timestamp / TIMESTAMP_SCALE
            timestamp = 0

        payload_data = self.stream_interface.read(payload - 4)
        packet = self._parse_packet(pid, timestamp, payload_data)
        return packet

    def _parse_packet(self, pid, timestamp, bin_data):
        """Generates the packets according to the pid

        Args:
            pid (int): Packet ID
            timestamp (int): Timestamp
            bin_data: Binary data

        Returns:
            Packet
        """

        if pid in PACKET_CLASS_DICT:
            packet = PACKET_CLASS_DICT[pid](timestamp, bin_data, self._time_offset)
        else:
            logger.debug("Unknown Packet ID:" + str(pid))
            packet = None
        return packet


class FileHandler:
    """Binary file handler"""
    def __init__(self, filename):
        """
        Args:
            filename (str): Binary file name
        """
        self.fid = open(filename, mode='rb')

    def read(self, n_bytes):
        """Read n bytes from file
        Args:
            n_bytes (int): Number of bytes to be read
        """
        if n_bytes <= 0:
            raise ValueError('Read length must be a positive number!')
        if not self.fid.closed:
            data = self.fid.read(n_bytes)
            if len(data) < n_bytes:
                raise EOFError('End of file!')
            return data
        raise IOError("File has not been opened or already closed!")

    def disconnect(self):
        """Close file"""
        if not self.fid.closed:
            self.fid.close()
