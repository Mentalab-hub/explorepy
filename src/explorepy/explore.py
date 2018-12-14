import numpy as np


class explore:
    r"""Mentalab Explore device"""
    def __init__(self, n_device=1):
        r"""

        Args:
            n_device (int): Number of devices to be connected
        """
        self.device = []

    def connect(self, id=0):
        r"""

        Args:
            id (int): device id

        Returns:

        """
        pass

    def disconnect(self, id=None):
        r"""

        Args:
            id: device id (id=None for disconnecting all devices)

        Returns:

        """
        pass

    def logdata(self):
        r"""
        Print the data in the terminal/console

        Returns:

        """
        pass

    def acquire(self):
        r"""
        Start getting data from the device

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

