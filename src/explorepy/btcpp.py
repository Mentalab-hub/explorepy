# -*- coding: utf-8 -*-
"""A module for bluetooth connection"""
import time
import logging

from explorepy import exploresdk
from explorepy._exceptions import DeviceNotFoundError, InputError

logger = logging.getLogger(__name__)


class SDKBtClient:
    """ Responsible for Connecting and reconnecting explore devices via bluetooth"""
    def __init__(self, device_name=None, mac_address=None):
        """Initialize Bluetooth connection

        Args:
            device_name(str): Name of the device (either device_name or device address should be given)
            mac_address(str): Devices MAC address
        """
        if (mac_address is None) and (device_name is None):
            raise InputError("Either name or address options must be provided!")
        self.is_connected = False
        self.mac_address = mac_address
        self.device_name = device_name
        self.bt_serial_port_manager = None

    def connect(self):
        """Connect to the device and return the socket

        Returns:
            socket (bluetooth.socket)
        """

        if self.mac_address is None:
            self._find_mac_address()
        else:
            self.device_name = "Explore_" + str(self.mac_address[-5:-3]) + str(self.mac_address[-2:])

        for _ in range(5):
            try:
                self.bt_serial_port_manager = exploresdk.BTSerialPortBinding_Create(self.mac_address, 5)
                return_code = self.bt_serial_port_manager.Connect()

                if return_code == 0:
                    self.is_connected = True
                    return
                else:
                    self.is_connected = False
                    logger.warning("Could not connect; Retrying in 2s...")
                    time.sleep(2)
            except Exception as e:
                self.is_connected = False
                logger.debug(f"Got an exception while connecting to the device: {e}")
                logger.warning("Could not connect; Retrying in 2s...")
                time.sleep(2)
        
        self.is_connected = False
        raise DeviceNotFoundError("Could not find the device! Please make sure"
                                  " the device is on and in advertising mode.")

    def reconnect(self):
        """Reconnect to the last used bluetooth socket.

        This function reconnects to the the last bluetooth socket. If after 1 minute the connection doesn't succeed,
        program will end.
        """
        self.is_connected = False
        for _ in range(5):
            self.bt_serial_port_manager = exploresdk.BTSerialPortBinding_Create(self.mac_address, 5)
            connection_error_code = self.bt_serial_port_manager.Connect()
            print("connection error code is " , connection_error_code)
            if connection_error_code == 0:
                self.is_connected = True
                print("INFO SDK: device is connected!")
                logger.info('Connected to the device')
                return self.bt_serial_port_manager
            else:
                self.is_connected = False
                print("INFO SDK: Could not connect to the device")  
                logger.warning("Couldn't connect to the device. Trying to reconnect...")
                time.sleep(2)
        logger.error("Could not reconnect after 5 attempts. Closing the socket.")
        return None

    def disconnect(self):
        """Disconnect from the device"""
        self.bt_serial_port_manager.Close()
        self.is_connected = False

    def _find_mac_address(self):
        self.device_manager = exploresdk.ExploreSDK_Create()
        for _ in range(5):
            available_list = self.device_manager.PerformDeviceSearch()
            for bt_device in available_list:
                if bt_device.name == self.device_name:
                    self.mac_address = bt_device.address
                    return

            logger.warning("No device found with the name: {}, searching again...".format(self.device_name))
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
            read_output = self.bt_serial_port_manager.Read(n_bytes)
            actual_byte_data = read_output.encode('utf-8', errors='surrogateescape')
            return actual_byte_data
        except Exception as error:
            logger.debug(f"Got an exception while reading data from socket: {error}")
            if error.args[0] == "EMPTY_BUFFER_ERROR":
                raise ConnectionAbortedError(error)

    def send(self, data):
        """Send data to the device

        Args:
            data (bytearray): Data to be sent
        """
        self.bt_serial_port_manager.Write(data)

    @staticmethod
    def _check_mac_address(device_name, mac_address):
        return (device_name[-4:-2] == mac_address[-5:-3]) and (device_name[-2:] == mac_address[-2:])
