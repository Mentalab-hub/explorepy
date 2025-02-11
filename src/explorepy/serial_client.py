# -*- coding: utf-8 -*-
"""A module for bluetooth connection"""
import logging
import threading
import time
from collections import deque

import serial
from serial.tools import list_ports

from explorepy import (
    exploresdk,
    settings_manager
)
from explorepy._exceptions import DeviceNotFoundError


logger = logging.getLogger(__name__)


class SerialClient:
    """ Responsible for Connecting and reconnecting explore devices via bluetooth"""

    def __init__(self, device_name):
        """Initialize Bluetooth connection
        """
        self.mac_address = None
        self.is_connected = False
        self.device_name = device_name
        self.bt_serial_port_manager = None
        self.device_manager = None
        self.bt_sdk = None

    def connect(self):
        """Connect to the device and return the socket

        Returns:
            socket (bluetooth.socket)
        """
        config_manager = settings_manager.SettingsManager(self.device_name)
        self.mac_address = config_manager.get_mac_address()
        if self.mac_address is None:
            self._find_mac_address()
            config_manager.set_mac_address(self.mac_address)
        result = exploresdk.BTSerialPortBinding.Create(self.mac_address, 5).Connect()
        print('result is {}'.format(result))
        for _ in range(5):
            try:
                self.bt_serial_port_manager = serial.Serial('/dev/tty.' + self.device_name, 9600, timeout=5)
                print('/dev/tty.' + self.device_name)
                self.is_connected = True
                return 0
            except Exception as error:
                self.is_connected = False
                logger.debug(
                    "Got an exception while connecting to the device: {} of type: {}".format(error, type(error))
                )
                logger.debug('trying to connect again as tty port is not visible yet')
                logger.warning("Could not connect; Retrying in 2s...")
                time.sleep(2)

        self.is_connected = False
        raise DeviceNotFoundError(
            "Could not find the device! Please make sure the device is on and connected to the computer"
        )

    def reconnect(self):
        """Reconnect to the last used bluetooth socket.

        This function reconnects to the last bluetooth socket. If after 1 minute the connection doesn't succeed,
        program will end.
        """

    def disconnect(self):
        """Disconnect from the device"""
        self.is_connected = False
        self.bt_serial_port_manager.close()

    def _find_mac_address(self):
        if self.device_name[8] == '8':
            self.mac_default = '00:13:43:A1:'
        else:
            self.mac_default = '00:13:43:93:'
        print(self.device_name[8])
        id_to_mac = self.device_name[-4:-2] + ':' + self.device_name[-2:]

        self.mac_address = self.mac_default + id_to_mac

    def read(self, n_bytes):
        """Read n_bytes from the socket

            Args:
                n_bytes (int): number of bytes to be read

            Returns:
                list of bytes
        """
        try:
            read_output = self.bt_serial_port_manager.read(n_bytes)
            return read_output
        except Exception as error:
            print(error)
            logger.error(
                "unknown error occurred while reading bluetooth data by "
                "pyserial {} of type:{}".format(error, type(error))
            )

    def send(self, data):
        """Send data to the device

        Args:
            data (bytearray): Data to be sent
        """
        self.bt_serial_port_manager.write(data)

    @staticmethod
    def _check_mac_address(device_name, mac_address):
        return (device_name[-4:-2] == mac_address[-5:-3]) and (device_name[-2:] == mac_address[-2:])


class SerialStream:
    """ Responsible for Connecting and reconnecting explore devices via bluetooth"""

    def __init__(self, device_name):
        """Initialize Bluetooth connection
        """
        self.is_connected = False
        self.device_name = device_name
        self.comm_manager = None
        self.device_manager = None
        self.bt_sdk = None
        self.usb_stop_flag = threading.Event()
        self.copy_buffer = deque()
        self.reader_thread = None
        self.lock = threading.Lock()

    def read_serial_in_chunks(self):
        """Reads data in fixed-size chunks from the serial port until stopped."""
        while not self.usb_stop_flag.is_set():
            try:
                bytes_available = self.comm_manager.in_waiting
                if bytes_available > 0:
                    data = self.comm_manager.read(bytes_available)
                    if data is not None:
                        self.copy_buffer.extend(data)
            except Exception as e:
                logger.debug('Got Exception in USB read method: {}'.format(e))
            time.sleep(0.000100)
        logger.debug('Stopping USB data retrieval thread')

    def connect(self):
        """Connect to the device and return the socket

        Returns:
            socket (bluetooth.socket)
        """
        for _ in range(5):
            try:
                port = get_correct_com_port(self.device_name)
                self.comm_manager = serial.Serial(port=port, baudrate=115200, timeout=2)

                # stop stream
                cmd = b'\xE5' * 14
                self.comm_manager.write(cmd)
                time.sleep(1)

                cmd = b'\xE4' * 14
                self.comm_manager.write(cmd)
                time.sleep(1)

                self.reader_thread = threading.Thread(
                    target=self.read_serial_in_chunks,
                    daemon=True
                )
                self.reader_thread.start()
                self.is_connected = True
                # wait to populate data buffer
                time.sleep(1)
                return 0
            except PermissionError:
                # do nothing here as this comes from posix
                pass
            except serial.serialutil.SerialException:
                logger.info(
                    "Permission denied on serial port access, please run this command via"
                    "terminal: sudo chmod 777 {}".format(port)
                )
            except Exception as error:
                self.is_connected = False
                logger.info(
                    "Got an exception while connecting to the device: {} of type: {}".format(error, type(error))
                )
                logger.debug('trying to connect again as tty port is not visible yet')
                logger.warning("Could not connect; Retrying in 2s...")
                time.sleep(2)

        self.is_connected = False
        raise DeviceNotFoundError(
            "Could not find the device! Please turn on the device, wait a few seconds and connect to"
            "serial port before starting ExplorePy"
        )

    def reconnect(self):
        """Reconnect to the last connected device

        Generally speaking this is not necessary for USB stream, but we keep it as placeholder
        in case it is needed in future
        """
        raise NotImplementedError

    def disconnect(self):
        """Disconnect from the device"""
        self.usb_stop_flag.set()
        self.reader_thread.join(timeout=2)
        self.is_connected = False
        self.comm_manager.cancel_read()
        self.comm_manager.close()

        time.sleep(1)

    def read(self, n_bytes):
        """Read n_bytes from the socket

            Args:
                n_bytes (int): number of bytes to be read

            Returns:
                list of bytes
        """
        try:
            chunk = bytearray()
            for i in range(1000):
                if len(self.copy_buffer) < n_bytes:
                    time.sleep(0.001)
                else:
                    break
            while len(chunk) < n_bytes:
                chunk.append(self.copy_buffer.popleft())
            return chunk
        except Exception as error:
            logger.debug('Got error or read request: {}'.format(error))

    def send(self, data):
        """Send data to the device

        Args:
            data (bytearray): Data to be sent
        """
        with threading.Lock():
            self.comm_manager.write(data)


def get_correct_com_port(device_name):
    """ Returns correct COM/tty port for usb connection
    Args: device name: the name of the device to connect to
    """
    ports = list(list_ports.comports())
    for p in ports:
        if p.vid == 0x0483 and p.pid == 0x5740:
            serial_port = serial.Serial(port=p.device, baudrate=115200, timeout=2)

            # stop stream
            cmd = b'\xE5' * 14
            serial_port.write(cmd)
            time.sleep(.1)
            # read all the stream data
            serial_port.readall()

            get_name_cmd = b'\xC6' * 14
            serial_port.write(get_name_cmd)
            data = serial_port.read(24)
            if len(data) == 0:
                # incompatible explore device, continue connection
                print('got data as zero')
                serial_port.close()
                return p.device
            name = data[8:-4].decode('utf-8', errors='ignore')
            if name == device_name:
                return p.device
