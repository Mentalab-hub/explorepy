# -*- coding: utf-8 -*-
"""Parser module"""
import asyncio
import binascii
import logging
import struct
import sys
from threading import Thread

import explorepy
from explorepy._exceptions import (
    BleDisconnectionError,
    FletcherError,
    ReconnectionFlowError
)
from explorepy.packet import (
    PACKET_CLASS_DICT,
    DeviceInfo,
    PacketBIN
)
from explorepy.settings_manager import SettingsManager
from explorepy.tools import (
    TIMESTAMP_SCALE,
    TIMESTAMP_SCALE_BLE,
    get_local_time,
    is_ble_mode,
    is_explore_pro_device,
    is_usb_mode
)


logger = logging.getLogger(__name__)


class Parser:
    """Data parser class"""

    def __init__(self, callback, mode='device', debug=True):
        """
        Args:
            callback (function): function to be called when new packet is received
            mode (str): Parsing mode either from an Explore device or a binary file {'device', 'file'}
        """
        self.device_name = None
        self.mode = mode
        self.debug = debug
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
        self.seek_new_pid = asyncio.Event()
        self.usb_marker_port = None
        self.total_packet_size_read = 0

    def start_streaming(self, device_name, mac_address):
        """Start streaming data from Explore device"""
        self.device_name = device_name
        if not device_name[-4:].isalpha():
            interface = 'pyserial' if sys.platform == "darwin" else 'sdk'
            explorepy.set_bt_interface(interface)
        if explorepy.get_bt_interface() == 'sdk':
            from explorepy.btcpp import SDKBtClient
            self.stream_interface = SDKBtClient(device_name=device_name, mac_address=mac_address)
        elif is_ble_mode():
            from explorepy.btcpp import BLEClient
            self.stream_interface = BLEClient(device_name=device_name, mac_address=mac_address)
        elif explorepy.get_bt_interface() == 'mock':
            from explorepy.bt_mock_client import MockBtClient
            self.stream_interface = MockBtClient(device_name=device_name, mac_address=mac_address)
        elif explorepy.get_bt_interface() == 'pyserial':
            from explorepy.serial_client import SerialClient
            self.stream_interface = SerialClient(device_name=device_name)
        elif explorepy.get_bt_interface() == 'usb':
            from explorepy.serial_client import SerialStream
            self.stream_interface = SerialStream(device_name=device_name)
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
            self.stream_interface = None
            if self.usb_marker_port is not None:
                self.usb_marker_port.close()

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
        try:
            while True:
                packet, _ = self._generate_packet()
                if isinstance(packet, DeviceInfo):
                    self.callback(packet=packet)
                    break
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
                packet, packet_size = self._generate_packet()
                self.total_packet_size_read += packet_size
                self.callback(packet=packet)
            except ReconnectionFlowError:
                logger.info('Got exception in reconnection flow, normal operation continues.')
                pass
            except ConnectionAbortedError as error:
                logger.debug(f"Got this error while streaming: {error}")
                logger.warning("Device has been disconnected! Scanning for the last connected device...")
                # saves current settings file
                SettingsManager(self.device_name).save_current_session()
                self._is_reconnecting = True
                if self.stream_interface.reconnect() is None:
                    logger.warning("Could not find the device! "
                                   "Please make sure the device is on and in advertising mode.")
                    self.stop_streaming()
                    print("Press Ctrl+c to exit...")
                self._is_reconnecting = False
            except (IOError, ValueError, MemoryError) as error:
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
            except FletcherError:
                if is_explore_pro_device():
                    logger.warning('Incomplete packet received, parsing will continue.')
                    self.seek_new_pid.set()
                else:
                    if self.mode == 'file':
                        logger.debug('Got Fletcher error in parsing BIN file, will continue')
                        self.seek_new_pid.set()
                    else:
                        self.stop_streaming()
            except BleDisconnectionError:
                logger.info('Explore pro disconnected, stopping streaming')
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
        while self.seek_new_pid.is_set():
            if self._is_reconnecting:
                raise ReconnectionFlowError()
            try:
                bytes_out = binascii.hexlify(bytearray(self.stream_interface.read(1)))
            except TypeError:
                if is_usb_mode():
                    self.stop_streaming()
                    break
                logger.info('No data in interface, seeking again.....')
                continue
            if bytes_out == b'af' and binascii.hexlify(bytearray(self.stream_interface.read(3))) == b'beadde':
                self.seek_new_pid.clear()
                break
        raw_header = self.stream_interface.read(8)
        try:
            pid = raw_header[0]
            raw_payload = raw_header[2:4]
            raw_timestamp = raw_header[4:8]

        except BaseException:
            raise FletcherError

        # pid = struct.unpack('B', raw_pid)[0]
        payload = struct.unpack('<H', raw_payload)[0]
        # max payload among all devices is 503, we need to make sure there is no corrupted data in payload length field
        if payload > 550:
            print('payload is {}'.format(payload))
            logger.debug('Got exception in payload determination, raising fletcher error')
            raise FletcherError

        timestamp = struct.unpack('<I', raw_timestamp)[0]
        if is_explore_pro_device():
            timestamp /= TIMESTAMP_SCALE_BLE
        else:
            timestamp /= TIMESTAMP_SCALE
        # Timestamp conversion
        if self._time_offset is None:
            self._time_offset = get_local_time() - timestamp

        payload_data = self.stream_interface.read(payload - 4)
        if self.debug:
            self.callback(packet=PacketBIN(raw_header + payload_data))
        try:
            packet = self._parse_packet(pid, timestamp, payload_data)
        except (AssertionError, TypeError, ValueError, struct.error) as error:
            logger.debug('Raising Fletcher error for: {}'.format(error))
            raise FletcherError
        packet_size = 8 + (payload - 4)
        return packet, packet_size

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
            raise FletcherError
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
