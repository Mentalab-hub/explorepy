import os
import shutil

parent_directory = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

files = (file for file in os.listdir(parent_directory)
         if os.path.isfile(os.path.join(parent_directory, file)))

for file in files:
    if '.pyd' in file:
        full_path = os.path.join(parent_directory, file)
        shutil.copy(full_path, os.getcwd())

from .explore import Explore
from . import tools, command

__version__ = '0.6.0'
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








