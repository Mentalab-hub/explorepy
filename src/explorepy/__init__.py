import os
import shutil
from sys import platform as _platform

parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

files = (file for file in os.listdir(parent_directory)
         if os.path.isfile(os.path.join(parent_directory, file)))

if _platform == "linux" or _platform == "linux2":
    for file in files:
        if '_exploresdk' in file:
            full_path = os.path.join(parent_directory, file)
            shutil.copy(full_path, os.path.dirname(__file__))

elif _platform == "win32" or _platform == "win64":
    for file in files:
        if '_exploresdk' in file:
            full_path = os.path.join(parent_directory, file)
            shutil.copy(full_path, os.path.dirname(__file__))


from .explore import Explore
from . import tools, command, exploresdk
from explorepy.dashboard.dashboard import Dashboard

__version__ = '1.0.0'
_bt_interface = 'sdk'


def set_bt_interface(bt_interface):
    """Set Explorepy Bluetooth interface type

    Args:
        bt_interface (str): Bluetooth interfacce type. Options: 'sdk' or 'pybluez'

    """
    if bt_interface not in ['sdk', 'pybluez']:
        raise ValueError
    import explorepy
    explorepy._bt_interface = bt_interface
