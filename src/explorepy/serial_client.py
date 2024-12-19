# -*- coding: utf-8 -*-
"""A module for bluetooth connection"""
import logging
import struct
import threading
import time

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


def get_device_name(p):
    """ Gets name of the Explore device
    Args:
        p (port instance): number of bytes to be read
    """
    serial_port = serial.Serial(port=p.device, baudrate=115200, timeout=2)
    get_name_cmd = b'\xC6' * 14
    serial_port.write(get_name_cmd)
    data = serial_port.read(4)
    length = struct.unpack('<H', data[2:])[0]
    data = serial_port.read(length)
    name = data[4:-4].decode('utf-8', errors='ignore')
    serial_port.close()
    time.sleep(1)
    return name


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
        self.copy_buffer = bytearray()
        self.reader_thread = None

    def read_serial_in_chunks(self):
        """Reads data in fixed-size chunks from the serial port until stopped."""
        while not self.usb_stop_flag.is_set():
            try:
                data = self.comm_manager.read(2048)
                if data is not None:
                    self.copy_buffer.extend(data)
            except Exception as e:
                print('Got Exception in USB read method: {}'.format(e))
                time.sleep(0.0001)
        logger.debug('Stopping USB data retrieval thread')

    def connect(self):
        """Connect to the device and return the socket

        Returns:
            socket (bluetooth.socket)
        """
        port = self.scan_usb_ports()

        for _ in range(5):
            try:
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
                return 0
            except PermissionError:
                # do nothing here as this comes from posix
                pass
            except serial.serialutil.SerialException:
                logger.info(
                    'Permission denied on serial port access, please run this command via \
                    terminal: sudo chmod 777 {}'.format(port)
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
            "Could not find the device! Please turn on the device, wait a few seconds and connect to \
            serial port before starting ExplorePy"
        )

    def scan_usb_ports(self):
        ports = list(list_ports.comports())
        for p in ports:
            try:
                if p.vid == 0x0483 and p.pid == 0x5740:
                    # Check device name
                    name = get_device_name(p)
                    if name == self.device_name:
                        logger.info('Device connected to USB port.')
                        return p.device
            except PermissionError:
                # do nothing here as this comes from posix
                pass
            except serial.serialutil.SerialException:
                logger.info(
                    "Permission denied on serial port access, please run this command via terminal:\
                    sudo chmod 777 {}".format(p)
                )
            except Exception as error:
                self.is_connected = False
                logger.info(
                    "Got an exception while connecting to the device: {} of type: {}".format(error, type(error))
                )
                logger.debug('trying to connect again as tty port is not visible yet')
                logger.warning("Could not connect; Retrying in 2s...")
                time.sleep(2)
        raise DeviceNotFoundError(
            "Could not find the device! Please turn on the device,\
            wait a few seconds and connect to serial port before starting ExplorePy"
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
            count = 10
            while len(self.copy_buffer) < n_bytes and count > 0 :
                time.sleep(.1)
                count -= 1
            data = self.copy_buffer[:n_bytes]
            self.copy_buffer = self.copy_buffer[n_bytes:]
            return data
        except Exception as error:
            logger.debug('Got error or read request: {}'.format(error))

    def send(self, data):
        """Send data to the device

        Args:
            data (bytearray): Data to be sent
        """
        self.comm_manager.write(data)
