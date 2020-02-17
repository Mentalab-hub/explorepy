# -*- coding: utf-8 -*-

"""
Customized exceptions module
"""


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


