# -*- coding: utf-8 -*-
"""A module for bluetooth connection"""
import logging
import serial
import serial.tools.list_ports as lp
import time

from explorepy import (
    exploresdk,
    settings_manager
)
from explorepy._exceptions import (
    DeviceNotFoundError,
    InputError
)


logger = logging.getLogger(__name__)


class SerialClient:
    """ Responsible for Connecting and reconnecting explore devices via bluetooth"""
    def __init__(self, device_name):
        """Initialize Bluetooth connection
        """
        self.is_connected = False
        self.device_name = device_name
        self.bt_serial_port_manager = None
        self.device_manager = None

    def connect(self):
        """Connect to the device and return the socket

        Returns:
            socket (bluetooth.socket)
        """

        self.device_name = "Explore_" + self.device_name

        for _ in range(5):

            try:
                #if len(lp.grep(self.device_name)) > 0:
                self.bt_serial_port_manager = serial.Serial('/dev/tty.Explore_84E4', 9600, timeout=1)
                self.bt_serial_port_manager.reset_input_buffer()
                self.is_connected = True
                return 0
            except Exception as error:
                self.is_connected = False
                logger.debug(
                    "Got an exception while connecting to the device: {} of type: {}".format(error, type(error))
                )
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
        self.is_connected = False
        logger.error("Could not reconnect after 5 attempts. Closing the socket.")
        return None

    def disconnect(self):
        """Disconnect from the device"""
        self.is_connected = False
        self.bt_serial_port_manager.Close()

    def _find_mac_address(self):
        self.device_manager = exploresdk.ExploreSDK.Create()
        for _ in range(5):
            available_list = self.device_manager.PerformDeviceSearch()
            logger.debug("Number of devices found: {}".format(len(available_list)))
            for bt_device in available_list:
                if bt_device.name == self.device_name:
                    self.mac_address = bt_device.address
                    return

            logger.warning("No device found with the name: %s, searching again...", self.device_name)
            time.sleep(0.1)
        raise DeviceNotFoundError("No device found with the name: {}".format(self.device_name))

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
                "unknown error occured while reading bluetooth data by "
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
