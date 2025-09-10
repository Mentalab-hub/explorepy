import abc

from explorepy._exceptions import InputError


class BTClient(abc.ABC):
    @abc.abstractmethod
    def __init__(self, device_name=None, mac_address=None):
        if (mac_address is None) and (device_name is None):
            raise InputError("Either name or address options must be provided!")
        self.is_connected = False
        self.mac_address = mac_address
        self.device_name = device_name
        self.bt_serial_port_manager = None

    @abc.abstractmethod
    def connect(self):
        """Connect to the device and return the socket

        Returns:
            socket (bluetooth.socket)
        """

    @abc.abstractmethod
    def reconnect(self):
        """Reconnect to the last used bluetooth socket.

        This function reconnects to the last bluetooth socket. If after 1 minute the connection doesn't succeed,
        program will end.
        """

    @abc.abstractmethod
    def disconnect(self):
        """Disconnect from the device"""

    @abc.abstractmethod
    def read(self, n_bytes):
        """Read n_bytes from the socket

            Args:
                n_bytes (int): number of bytes to be read

            Returns:
                list of bytes
        """

    @abc.abstractmethod
    def send(self, data):
        """Send data to the device

        Args:
            data (bytearray): Data to be sent
        """
