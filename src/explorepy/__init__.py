import logging
import platform
import sys

from . import log_config


# need to import logger before importing other modules to catch logs during initialization
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.debug("NEW SESSION ---------------------------------------------")
logger.debug("OS: %s - %s ", platform.platform(), sys.version)
log_config.log_breadcrumb(f"OS: {platform.platform()} - {sys.version}", "info")

from . import (  # noqa ignore E402
    command,
    exploresdk,
    tools
)
from .dashboard.dashboard import Dashboard  # noqa
from .explore import Explore  # noqa


__all__ = ["Explore", "Dashboard", "command", "exploresdk", "tools", "log_config"]
__version__ = '1.7.0'

this = sys.modules[__name__]
this._bt_interface = 'sdk'

if not sys.version_info >= (3, 6):
    raise EnvironmentError("Explorepy requires python versions 3.6 or newer!")

logger.debug("Explorepy version: %s", __version__)
log_config.set_sentry_tag("explorepy.version", __version__)


def set_bt_interface(bt_interface):
    """Set Explorepy Bluetooth interface type

    Args:
        bt_interface (str): Bluetooth interface type. Options:'sdk'

    """
    if bt_interface not in ['sdk']:
        raise ValueError(("Invalid Bluetooth interface: " + bt_interface))

    this._bt_interface = bt_interface
    logger.info("BT interface is set to %s", bt_interface)


def get_bt_interface():
    """Get Explorepy Bluetooth interface name

    Returns:
        bt_interface (str): Current Bluetooth interface: 'sdk'
    """
    return this._bt_interface
