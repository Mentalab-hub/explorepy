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


class FletcherError(Exception):
    """
    Fletcher value error
    """
    pass

class BluetoothError(Exception):
    """
    mock exception class for mac OS
    """
    pass

if sys.platform == "darwin":
    BluetoothError = BluetoothError
else:
    from bluetooth import BluetoothError

