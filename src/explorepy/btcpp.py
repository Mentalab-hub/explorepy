# -*- coding: utf-8 -*-
"""A module for bluetooth connection"""
import logging
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
        self.device_manager = None

    def connect(self):
        """Connect to the device and return the socket

        Returns:
            socket (bluetooth.socket)
        """
        config_manager = settings_manager.SettingsManager(self.device_name)
        mac_address = config_manager.get_mac_address()

        if mac_address is None:
            self._find_mac_address()
            config_manager.set_mac_address(self.mac_address)
        else:
            self.mac_address = mac_address
            self.device_name = "Explore_" + str(self.mac_address[-5:-3]) + str(self.mac_address[-2:])

        for _ in range(5):
            try:
                self.bt_serial_port_manager = exploresdk.BTSerialPortBinding_Create(self.mac_address, 5)
                return_code = self.bt_serial_port_manager.Connect()
                logger.debug("Return code for connection attempt is : {}".format(return_code))

                if return_code == 0:
                    self.is_connected = True
                    return
                else:
                    self.is_connected = False
                    logger.warning("Could not connect; Retrying in 2s...")
                    time.sleep(2)

            except TypeError as error:
                self.is_connected = False
                logger.debug(
                    "Got an exception while connecting to the device: {} of type: {}".format(error, type(error))
                )
                raise ConnectionRefusedError("Please unpair Explore device manually or use a Bluetooth dongle")
            except Exception as error:
                self.is_connected = False
                logger.debug(
                    "Got an exception while connecting to the device: {} of type: {}".format(error, type(error))
                )
                logger.warning("Could not connect; Retrying in 2s...")
                time.sleep(2)

        self.is_connected = False
        raise DeviceNotFoundError(
            "Could not find the device! Please make sure the device is on and in advertising mode."
        )

    def reconnect(self):
        """Reconnect to the last used bluetooth socket.

        This function reconnects to the last bluetooth socket. If after 1 minute the connection doesn't succeed,
        program will end.
        """
        self.is_connected = False
        for _ in range(5):
            self.bt_serial_port_manager = exploresdk.BTSerialPortBinding_Create(self.mac_address, 5)
            connection_error_code = self.bt_serial_port_manager.Connect()
            logger.debug("Got an exception while connecting to the device: {}".format(connection_error_code))
            if connection_error_code == 0:
                self.is_connected = True
                logger.info('Connected to the device')
                return self.bt_serial_port_manager
            else:
                self.is_connected = False
                logger.warning("Couldn't connect to the device. Trying to reconnect...")
                time.sleep(2)
        logger.error("Could not reconnect after 5 attempts. Closing the socket.")
        return None

    def disconnect(self):
        """Disconnect from the device"""
        self.is_connected = False
        self.bt_serial_port_manager.Close()

    def _find_mac_address(self):
        self.device_manager = exploresdk.ExploreSDK_Create()
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
            read_output = self.bt_serial_port_manager.Read(n_bytes)
            actual_byte_data = read_output.encode('utf-8', errors='surrogateescape')
            return actual_byte_data
        except OverflowError as error:
            if not self.is_connected:
                raise IOError("connection has been closed")
            else:
                logger.debug(
                    "Got an exception while reading data from "
                    "socket which connection is open: {} of type:{}".format(error, type(error)))
                raise ConnectionAbortedError(error)
        except IOError as error:
            if not self.is_connected:
                raise IOError(str(error))
        except (MemoryError, OSError) as error:
            logger.debug("Got an exception while reading data from socket: {} of type:{}".format(error, type(error)))
            raise ConnectionAbortedError(str(error))
        except Exception as error:
            print(error)
            logger.error(
                "unknown error occured while reading bluetooth data by "
                "exploresdk {} of type:{}".format(error, type(error))
            )

    def send(self, data):
        """Send data to the device

        Args:
            data (bytearray): Data to be sent
        """
        self.bt_serial_port_manager.Write(data)

    @staticmethod
    def _check_mac_address(device_name, mac_address):
        return (device_name[-4:-2] == mac_address[-5:-3]) and (device_name[-2:] == mac_address[-2:])
