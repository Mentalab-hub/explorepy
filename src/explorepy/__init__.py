from .explore import Explore
from . import tools, command
import _exploresdk
from explorepy.dashboard.dashboard import Dashboard

__version__ = '1.1.0'
_bt_interface = 'pybluez'


def set_bt_interface(bt_interface):
    """Set Explorepy Bluetooth interface type

    Args:
        bt_interface (str): Bluetooth interfacce type. Options: 'sdk' or 'pybluez'

    """
    if bt_interface not in ['sdk', 'pybluez']:
        raise ValueError
    import explorepy
    explorepy._bt_interface = bt_interface
