import numpy as np
from .bt_client import BtClient
from .parser import Parser


class Explore:
    r"""Mentalab Explore device"""
    def __init__(self, n_device=1):
        r"""

        Args:
            n_device (int): Number of devices to be connected
        """
        self.device = []
        for i in range(n_device):
            self.device.append(BtClient())

    def connect(self, device_id=0):
        r"""

        Args:
            id (int): device id

        Returns:

        """
        self.device[device_id].connect()

    def disconnect(self, device_id=None):
        r"""

        Args:
            id: device id (id=None for disconnecting all devices)

        Returns:

        """
        self.device[device_id].socket.close()

    def acquire(self, device_id=0):
        r"""
        Start getting data from the device

        """
        exp_parser = Parser(socket=self.device[device_id].socket)
        try:
            while True:
                pid, timestamp, data = exp_parser.parse_packet()
                print("package ID: [%i]" % pid)
        except ValueError:
            # If value error happens, scan again for devices and try to reconnect (see reconnect function)
            print("Disconnected, scanning for last connected device")
            self.device[device_id].is_connected = False
            self.disconnect(device_id)

    def logdata(self):
        r"""
        Print the data in the terminal/console

        Returns:

        """
        pass

    def push2lsl(self):
        r"""
        push the stream to lsl

        Returns:

        """
        pass

    def visualize(self):
        r"""
        Start visualization of the data in the viewer
        Returns:

        """
        pass


if __name__ == '__main__':
    pass
