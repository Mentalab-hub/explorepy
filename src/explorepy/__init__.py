import logging
import platform
import sys
from . import log_config
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.debug("NEW SESSION ---------------------------------------------")
logger.debug("Explorepy starting...")
logger.debug(f"OS: {platform.platform()} - {sys.version}")

from .explore import Explore
from . import tools, command, exploresdk
from explorepy.dashboard.dashboard import Dashboard


__version__ = '1.3.0'
_bt_interface = 'pybluez'
if not sys.version_info >= (3, 6):
    raise EnvironmentError("Explorepy requires python versions 3.6 or newer!")
logger.debug(f"Explorepy version: {__version__}")
logger.debug(f"BT interface: {_bt_interface}")


def set_bt_interface(bt_interface):
    """Set Explorepy Bluetooth interface type

    Args:
        bt_interface (str): Bluetooth interfacce type. Options: 'sdk' or 'pybluez'

    """
    if bt_interface not in ['sdk', 'pybluez']:
        raise ValueError("bt_interface must be either sdk or pybluez")
    import explorepy
    explorepy._bt_interface = bt_interface
    logger.info(f"BT interface was changed to {bt_interface}")


def get_bt_interface():
    """Get current Bluetooth interface name

    Returns:
            string: Bluetooth interface name
    """
    return _bt_interface
