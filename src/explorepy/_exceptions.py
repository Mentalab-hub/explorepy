# -*- coding: utf-8 -*-

"""
Customized exceptions module
"""
import sys


class UnexpectedConnectionError(ConnectionError):
    """
    Generic exception thrown if an unexpected error occurs during connection
    """
    pass


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


class BleDisconnectionFailedError(Exception):
    """
    Exception for client fails to achieve disconnected state
    """
    pass


class ImpedanceModeActiveError(Exception):
    """
    Exception for ASR, raised when impedance is running
    """

    def __init__(self, message="ASR can not run because impedance mode is active.\n"
                               "Please disable impedance and try again\n"):
        super().__init__(message)


class ExplorePyDeprecationError(Exception):
    def __init__(self, message="Explorepy support for legacy devices is deprecated.\n"
                               "Please install explorepy 3.2.1 from Github or use the following command from Anaconda "
                               "prompt:\npip install explorepy==3.2.1 \n"):
        super().__init__(message)


if sys.platform == "darwin":
    class BluetoothError(Exception):
        """
        mock exception class for mac OS
        """
        pass
