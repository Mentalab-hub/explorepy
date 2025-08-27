# -*- coding: utf-8 -*-
"""Parser module"""
import asyncio
import binascii
import logging
import mmap
import multiprocessing
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
from typing import (
    Generator,
    List,
    Optional,
    Tuple
)

import numpy as np

import explorepy
from explorepy._exceptions import (
    BleDisconnectionError,
    FletcherError,
    ReconnectionFlowError
)
from explorepy.packet import (
    PACKET_CLASS_DICT,
    DeviceInfo,
    Packet,
    PacketBIN
)
from explorepy.settings_manager import SettingsManager
from explorepy.tools import (
    TIMESTAMP_SCALE_BLE,
    is_ble_mode,
    is_explore_pro_device,
    is_usb_mode
)


logger = logging.getLogger(__name__)


class Parser:
    """Data parser class"""

    def __init__(self, callback, progress_callback=None, mode='device', debug=True):
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

        self._do_streaming = False
        self.is_waiting = False
        self._stream_thread = None
        self._is_reconnecting = False
        self.seek_new_pid = asyncio.Event()
        self.usb_marker_port = None
        self.total_packet_size_read = 0
        self.progress = 0
        self.progress_callback = progress_callback
        self.header_len = 0
        self.data_len = 0

    def start_streaming(self, device_name, mac_address):
        """Start streaming data from Explore device"""
        self.device_name = device_name
        if is_ble_mode():
            from explorepy.BLEClient import BLEClient
            self.stream_interface = BLEClient(device_name=device_name, mac_address=mac_address)
        elif explorepy.get_bt_interface() == 'mock':
            from explorepy.bt_mock_client import MockBtClient
            self.stream_interface = MockBtClient(device_name=device_name, mac_address=mac_address)
        elif is_usb_mode():
            from explorepy.serial_client import SerialStream
            self.stream_interface = SerialStream(device_name=device_name)
        else:
            raise ValueError("Support for legacy Explore devices is deprecated starting from ExplorePy 4.0.0.\n"
                             "Please use the following command to use ExplorePy with a legacy device\n"
                             "pip install explorepy==3.2.1\n"
                             "https://explorepy.readthedocs.io/en/latest/explore_legacy_devices\n")
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
            self.header_len = 0

    def start_reading(self, filename):
        """Open the binary file and start reading packets
        Args:
            filename (str): Binary file name
        """
        self.stream_interface = FileHandler(filename)
        total_packet_batch = 0
        packet_generator = self._generate_packets_from_file()

        try:
            while True:
                batch, total_markers = next(packet_generator)
                self.callback(packet_batch=batch)
                self.progress += (len(batch) / total_markers) * 100
                if self.progress_callback:
                    self.progress_callback(min(self.progress, 100.0))
                total_packet_batch += 1

        except StopIteration:
            logger.debug(f"Total batches of packets collected: {total_packet_batch}")
            if self.progress_callback:
                self.progress_callback(100.0)
        except EOFError:
            logger.info('Reached end of the file')
        finally:
            self.stream_interface.disconnect()

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
        raw_header = self.get_header_bytes()
        try:
            pid, timestamp, payload = self.parser_header(raw_header)
        except BaseException:
            raise FletcherError

        # max payload among all devices is 503, we need to make sure there is no corrupted data in payload length field
        if payload > 550:
            print('payload is {}'.format(payload))
            logger.debug('Got exception in payload determination, raising fletcher error')
            raise FletcherError

        payload_data = self.stream_interface.read((payload - self.data_len))
        if self.debug:
            self.callback(packet=PacketBIN(raw_header + payload_data))
        try:
            packet = self._parse_packet(pid, timestamp, payload_data)
        except (AssertionError, TypeError, ValueError, struct.error) as error:
            logger.debug('Raising Fletcher error for: {}'.format(error))
            raise FletcherError
        packet_size = self.header_len + payload - self.data_len
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
            packet = PACKET_CLASS_DICT[pid](timestamp, bin_data)
        else:
            logger.debug("Unknown Packet ID:" + str(pid))
            packet = None
            raise FletcherError
        return packet

    def _process_packet_chunk(self, marker_positions: List[int], buffer: bytearray) -> List[Tuple]:
        """Process a single batch of packets in one thread."""
        PACKET_MARKER = b'\xaf\xbe\xad\xde'
        chunk_packets = []
        parse_time = 0
        payload_time = 0

        for current_pos in marker_positions:
            try:
                parse_start = time.time()
                header_start = current_pos + len(PACKET_MARKER)
                if header_start + self.header_len > len(buffer):
                    continue
                raw_header = buffer[header_start:header_start + self.header_len]
                pid, timestamp, payload_length = self.parser_header(raw_header)
                parse_time += time.time() - parse_start
                if payload_length > 550:
                    continue
                payload_start = time.time()
                payload_start_idx = header_start + self.header_len
                payload_end = payload_start_idx + payload_length - self.data_len
                payload_end = payload_start_idx + payload_length - 4
                if payload_end > len(buffer):
                    continue
                payload_data = buffer[payload_start_idx:payload_end]
                chunk_packets.append((pid, timestamp, payload_data))
                payload_time += time.time() - payload_start
            except (IndexError, struct.error) as e:
                logger.debug(f'Error parsing packet at position {current_pos}: {e}')
                continue
        return chunk_packets

    def _generate_packets_from_file(self, batch_size: int = 10000,
                                    num_threads: int = 4) -> Generator[List[Tuple], None, None]:
        """Reads and parses packets from file in parallel, aligning batch size with thread processing."""
        PACKET_MARKER = b'\xaf\xbe\xad\xde'
        num_threads = multiprocessing.cpu_count()
        try:
            # Time: File reading
            buffer = bytearray(self.stream_interface.read())
            # Time: Finding markers
            arr = np.frombuffer(buffer, dtype=np.uint8)
            marker_arr = np.frombuffer(PACKET_MARKER, dtype=np.uint8)
            matches = np.where(arr[:-3] == marker_arr[0])[0]
            marker_positions = [
                pos for pos in matches
                if buffer[pos:pos + 4] == PACKET_MARKER
            ]
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = []
                for i in range(0, len(marker_positions), batch_size):
                    chunk = marker_positions[i:i + batch_size]
                    futures.append(executor.submit(self._process_packet_chunk, chunk, buffer))
                for future in futures:
                    try:
                        chunk_packets = future.result()
                        processed_packets = Packet.parse_packets_batch(chunk_packets)
                        batch = [(packet, self.header_len + len(info[2])) for packet, info in
                                 zip(processed_packets, chunk_packets)]
                        yield batch, len(marker_positions)
                    except FletcherError:
                        print('Fletcher checksum error in batch, skipping affected packets')
                        continue
        except (IOError, ValueError) as e:
            print(f'Error reading file: {e}')
            raise

    @staticmethod
    def unpack_timestamp(raw_bytes):
        # choose right unpacking format, unsigned long long/unsigned int
        fmt = '<Q' if len(raw_bytes) == 8 else '<I'
        return struct.unpack(fmt, raw_bytes)[0]

    def parser_header(self, raw_header):
        pid = raw_header[0]
        raw_payload = raw_header[2:4]
        raw_timestamp = raw_header[4:self.header_len]
        timestamp = self.unpack_timestamp(raw_timestamp) / TIMESTAMP_SCALE_BLE
        payload = struct.unpack('<H', raw_payload)[0]
        return pid, timestamp, payload

    def get_header_bytes(self):
        if self.header_len == 0:
            pid_bin = self.stream_interface.read(1)
            self.header_len = 12 if pid_bin[0] == 99 else 8
            self.data_len = self.header_len - 4
            return pid_bin + self.stream_interface.read(self.header_len - 1)
        else:
            return self.stream_interface.read(self.header_len)


class FileHandler:
    """Binary file handler with conditional memory mapping for improved performance"""

    def __init__(self, filename: str):
        """
        Initialize file handler.

        Args:
            filename (str): Path to the binary file
        """
        self.filename = filename
        self.file = open(filename, mode='rb')
        self.mmap = None

    def read(self, n_bytes: Optional[int] = None) -> bytes:
        """
        Read from file, using mmap only when reading entire file.
        Args:
            n_bytes: Number of bytes to read. If None, reads entire file with mmap.
        Returns:
            bytes: The data read from file
        Raises:
            ValueError: If n_bytes is negative
            EOFError: If reached end of file while reading n_bytes
            IOError: If file is not open or already closed
        """
        if self.file.closed:
            raise IOError("File has not been opened or already closed!")
        if n_bytes is None:
            # Only create mmap when needed for full file read
            if self.mmap is None:
                self.mmap = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
            return self.mmap[:]
        if n_bytes <= 0:
            raise ValueError('Read length must be positive!')
        # Use regular file I/O for partial reads
        data = self.file.read(n_bytes)
        if len(data) < n_bytes:
            raise EOFError('End of file!')

        return data

    def disconnect(self):
        """Close both the memory map (if exists) and file"""
        if self.mmap is not None:
            self.mmap.close()
            self.mmap = None
        if not self.file.closed:
            self.file.close()
