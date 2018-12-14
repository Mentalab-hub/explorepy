import numpy as np
from abc import ABC, abstractmethod
import struct


class Packet:
    """An abstract base class for Explore packet"""
    def __init__(self, pid, data):
        """
        Should get the packet id and the remaining packet bytes and convert it to the
        Args:
            pid:
            data:
        """
        self.pid = pid
        self._check_validity(pid, data)
        self.cnt = struct.unpack('B', data[0])
        self.payload = struct.unpack('<H', data[1:3])
        self.timestamp = struct.unpack('<I', data[3:5])
        # TO DO: remove the converted bytes?

    @abstractmethod
    def _check_validity(self, pid, data):
        """
        Check if the length of the data is valid
        Returns:
            Validity: (bool)

        """
        # TO DO: Assert an error
        pass

    @abstractmethod
    def _convert(self, data):
        """
        Read the binary data and convert it to real values
        Args:
            data:

        Returns:

        """
        pass

    @abstractmethod
    def _check_fletcher(self, data):
        """

        Args:
            data:

        Returns:

        """
        # TO DO: check the validity of the Fletcher
        pass

class EEG4(Packet):
    """EEG packet for 4 channel device"""
    def __init__(self, pid, data):
        """

        Args:
            pid:
            data:
        """
        super().__init__(pid, data)
        self._convert(data)

    def _check_validity(self, pid, data):
        # if len(data) != SOME_CONSTANT:
        #     assert an error
        pass

    def _convert(self, data):
        # Implement the formula for the value
        self.data = None
        self.status = None


class EEG8(Packet):
    """EEG packet for 8 channel device"""
    def __init__(self, pid, data):
        """

        Args:
            pid:
            data:
        """
        super().__init__(pid, data)
        self._convert(data)

    def _check_validity(self, pid, data):
        # if len(data) != SOME_CONSTANT:
        #     assert an error
        pass

    def _convert(self, data):
        # Implement the formula for the value
        self.data = None
        self.status = None


class Orientation(Packet):
    """Orientation data packet"""
    def __init__(self, pid, data):
        """

        Args:
            pid:
            data:
        """
        super().__init__(pid, data)
        self._convert(data)

    def _check_validity(self, pid, data):
        # if len(data) != SOME_CONSTANT:
        #     assert an error
        pass

    def _convert(self, data):
        # Implement the formula for the value
        self.acc = None
        self.gyro = None
        self.mag = None

class Environment(Packet):
    """Environment data packet"""
    def __init__(self, pid, data):
        """

        Args:
            pid:
            data:
        """
        super().__init__(pid, data)
        self._convert(data)

    def _check_validity(self, pid, data):
        # if len(data) != SOME_CONSTANT:
        #     assert an error
        pass

    def _convert(self, data):
        # Implement the formula for the value
        self.battery = None
        self.light = None

class TimeStamp(Packet):
    """Time stamp data packet"""
    def __init__(self, pid, data):
        """

        Args:
            pid:
            data:
        """
        super().__init__(pid, data)

    def _check_validity(self, pid, data):
        # if len(data) != SOME_CONSTANT:
        #     assert an error
        pass

    def _convert(self, data):
        # Implement the formula for the value
        pass