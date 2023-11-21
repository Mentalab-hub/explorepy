# -*- coding: utf-8 -*-
"""A module for bluetooth connection"""
import logging
import subprocess
import time

import serial

from explorepy import settings_manager
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

        for _ in range(5):
            try:
                self.connect_bluetooth_device()
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
                return -1

        self.is_connected = False
        raise DeviceNotFoundError(
            "Could not find the device! Please make sure the device is on and connected to the computer"
        )

    def reconnect(self):
        """Reconnect to the last used bluetooth socket.

        This function reconnects to the last bluetooth socket. If after 1 minute the connection doesn't succeed,
        program will end.
        """
        self.is_connected = False
        for _ in range(5):
            try:
                self.connect_bluetooth_device()
                self.bt_serial_port_manager = serial.Serial('/dev/tty.' + self.device_name, 9600, timeout=5)
                self.is_connected = True
                logger.info('Connected to the device')
                return self.bt_serial_port_manager
            except Exception as error:
                self.is_connected = False
                logger.warning("Couldn't connect to the device. Trying to reconnect...")
                time.sleep(2)
        logger.error("Could not reconnect after 5 attempts. Closing the socket.")
        return None

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
            if len(read_output) == 0:
                logger.debug(
                    "Could not read from inout stream. Raising exception")
                raise ConnectionAbortedError("device disconnected, attempting to reconnect..")
            return read_output
        except Exception as error:
            print(error)
            logger.error(
                "unknown error occured while reading bluetooth data by "
                "pyserial {} of type:{}".format(error, type(error))
            )
            raise ConnectionAbortedError("device disconnected, attempting to reconnect..")

    def send(self, data):
        """Send data to the device

        Args:
            data (bytearray): Data to be sent
        """
        self.bt_serial_port_manager.write(data)

    @staticmethod
    def _check_mac_address(device_name, mac_address):
        return (device_name[-4:-2] == mac_address[-5:-3]) and (device_name[-2:] == mac_address[-2:])

    def connect_bluetooth_device(self):
        try:
            subprocess.run(["blueutil", '--connect', self.mac_address], check=True)
            print(f"Attempted to connect to the device with address: {self.mac_address}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to connect to the device: {e}")
