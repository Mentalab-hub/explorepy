from .explore import Explore
from . import tools, command, exploresdk
from explorepy.dashboard.dashboard import Dashboard
import explorepy
from sys import platform as _platform

__version__ = '1.3.0'
_bt_interface = 'sdk'

def set_bt_interface(bt_interface):
    """Set Explorepy Bluetooth interface type

    Args:
        bt_interface (str): Bluetooth interface type. Options: 'sdk' or 'pybluez'

    """
    if bt_interface not in ['sdk', 'pybluez']:
        raise ValueError
    
    if _platform == 'darwin' and bt_interface == 'pybluez':
        print('Setting Pybluez as Bluetooth backend is not supported in Mac OSX')
        return
    
    explorepy._bt_interface = bt_interface


def get_bt_interface():
    """Get Explorepy Bluetooth interface 

    Returns:
        bt_interface (str): Current Bluetooth interface: 'sdk' or 'pybluez'

    """
    return explorepy._bt_interface
