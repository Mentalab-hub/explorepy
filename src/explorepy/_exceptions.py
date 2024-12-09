# -*- coding: utf-8 -*-

"""
Customized exceptions module
"""
import sys


class InputError(Exception):
    """
    User input exception
    """
    pass


class DeviceNotFoundError(IOError):
    """
    Device not found exception
    """
    pass


class UnsupportedBtHardwareError(Exception):
    """
    Exception for Bluetooth hardwares needing manual intervention
    """
    pass


class FletcherError(Exception):
    """
    Fletcher value error
    """
    pass


class ReconnectionFlowError(Exception):
    """
    Reconnection flow error, only thrown when device is reconnecting
    """
    pass

class BleDisconnectionError(Exception):
    """
    Reconnection flow error, only thrown when device is reconnecting
    """
    pass

if sys.platform == "darwin":
    class BluetoothError(Exception):
        """
        mock exception class for mac OS
        """
        pass
