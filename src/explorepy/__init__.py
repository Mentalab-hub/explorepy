import logging
import platform
import sys
from . import log_config

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.debug("NEW SESSION ---------------------------------------------")
logger.debug("OS: %s - %s ", platform.platform(), sys.version)
log_config.log_breadcrumb(f"OS: {platform.platform()} - {sys.version}", "info")

from .explore import Explore
from . import tools, command, exploresdk
from .dashboard.dashboard import Dashboard


__version__ = '1.5.2'
this = sys.modules[__name__]
this._bt_interface = 'sdk'

if not sys.version_info >= (3, 6):
    raise EnvironmentError("Explorepy requires python versions 3.6 or newer!")

logger.debug("Explorepy version: %s", __version__)
log_config.set_sentry_tag("explorepy.version", __version__)


def set_bt_interface(bt_interface):
    """Set Explorepy Bluetooth interface type

    Args:
        bt_interface (str): Bluetooth interface type. Options: 'sdk' or 'pybluez'

    """
    if bt_interface not in ['sdk', 'pybluez']:
        raise ValueError

    if sys.platform == 'darwin' and bt_interface == 'pybluez':
        logger.warning('Setting Pybluez as Bluetooth backend is not supported in Mac OSX')
        return
    this._bt_interface = bt_interface
    logger.info("BT interface is set to %s", bt_interface)


def get_bt_interface():
    """Get Explorepy Bluetooth interface name

    Returns:
        bt_interface (str): Current Bluetooth interface: 'sdk' or 'pybluez'

    """
    return this._bt_interface

