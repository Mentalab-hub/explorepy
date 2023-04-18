# -*- coding: utf-8 -*-
"""A module for bluetooth connection"""
import logging
import time

from explorepy import (
    exploresdk,
    settings_manager
)

from explorepy.bt_mock_server import (
    MockBtServer
)

from explorepy._exceptions import (
    InputError
)

logger = logging.getLogger(__name__)


class MockBtClient:
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
        self.mac_address = 'ABCD_EFGH_IJKL_MNOP'  # dummy name as MAC address
        self.device_name = 'Explore_Mock_Device'  # dummy name, can be changed
        self.bt_serial_port_manager = None
        self.device_manager = None

    def connect(self):
        """Connect to the device and return the socket

        Returns:
            socket (bluetooth.socket)
        """
        config_manager = settings_manager.SettingsManager(self.device_name)
        config_manager.set_mac_address(self.mac_address)

        # sets up necessary variables
        self.bt_serial_port_manager = MockBtServer()
        self.bt_serial_port_manager.Connect()
        logger.info('Connected to the device')
        self.is_connected = True

    def reconnect(self):
        """Reconnect to the last used bluetooth socket.

        This function reconnects to the last bluetooth socket. If after 1 minute the connection doesn't succeed,
        program will end.
        """
        self.is_connected = False
        for _ in range(5):
            self.bt_serial_port_manager = MockBtServer()
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
        self.mac_address = 'ABCD_EFGH_IJKL_MNOP' #  dummy MAC address

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
        return True
