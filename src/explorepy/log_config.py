# -*- coding: utf-8 -*-
"""Logging configurator module"""
import configparser
import logging
import logging.handlers
import os
import sys
import threading
import time

import sentry_sdk
from appdirs import (
    user_config_dir,
    user_log_dir
)

from explorepy._exceptions import DeviceNotFoundError


_IGNORED_EXC_BY_SENTRY = [DeviceNotFoundError, FileExistsError]

USER_SETTING_KEY = "user settings"
SHARE_LOG_PERMISSION_KEY = "share_logs"
APP_AUTHOUR = "Mentalab"
APP_NAME = "explorepy"

explorepy_logger = logging.getLogger('explorepy')
explorepy_logger.propagate = False
logger = logging.getLogger(__name__)

log_path = user_log_dir(appname=APP_NAME, appauthor=APP_AUTHOUR)
log_filename = 'explorepy.log'
os.makedirs(log_path, exist_ok=True)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - [%(levelname)s] - [%(threadName)-10s] - %(message)s')
console_formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')

file_log_handler = logging.handlers.RotatingFileHandler(filename=os.path.join(log_path, log_filename),
                                                        maxBytes=5e5, backupCount=5)
file_log_handler.setLevel(logging.DEBUG)
file_log_handler.setFormatter(file_formatter)

console_log_handler = logging.StreamHandler()
console_log_handler.setFormatter(console_formatter)
console_log_handler.setLevel(logging.INFO)

explorepy_logger.addHandler(file_log_handler)
explorepy_logger.addHandler(console_log_handler)
logging.getLogger().addHandler(file_log_handler)


def setup_thread_excepthook():
    """
    Workaround for `sys.excepthook` thread bug from:
    http://bugs.python.org/issue1230540

    Source:
        https://stackoverflow.com/questions/1643327/sys-excepthook-and-threading

    Notes:
        The 'sys.excepthook' bug has been fixed in Python 3.8 but as we want to keep support for older version
        of Python, we need this workaround.
    """
    init_original = threading.Thread.__init__

    def init(self, *args, **kwargs):

        init_original(self, *args, **kwargs)
        run_original = self.run

        def run_with_except_hook(*args2, **kwargs2):
            try:
                run_original()
            except Exception:
                sys.excepthook(*sys.exc_info())

        self.run = run_with_except_hook

    threading.Thread.__init__ = init


def read_config(section, var):
    """Read config file and return the value of the requested variable

    Args:
        section (str): section name
        var (str): variable name

    Returns:
        str: content of the requested variable, empty string if the file/section/variable is not found
    """
    config_file = os.path.join(user_config_dir(appname="explorepy", appauthor="Mentalab"), "conf.ini")
    if os.path.isfile(config_file):
        config = configparser.ConfigParser()
        config.read(config_file)
        if section in config and var in config[section]:
            return config[section][var]
    return ""


def write_config(section, var, val):
    """ Write to config file
    Args:
        section (str): section name
        var (str): variable name
        val (str): value
    Exceptions:
        ValueError: Raised if the inputs types are other than string.
        OSError: Raised if there is an error during reading/writing config file.
    Returns:
        bool: True if the action is successful, otherwise False
    """
    if not (type(val) == str and type(section) == str and type(var) == str):
        logger.error("Wrong input type. Inputs must be strings.")
        raise ValueError("Wrong input type. Inputs must be strings.")

    config_file = os.path.join(user_config_dir(appname="explorepy", appauthor="Mentalab"), "conf.ini")

    config = configparser.ConfigParser()
    if os.path.isfile(config_file):
        config.read(config_file)
        if not config.has_section(section):
            config.add_section(section)
    if not os.path.isfile(config_file):  # Create the dir and file
        os.makedirs(user_config_dir(appname="explorepy", appauthor="Mentalab"), exist_ok=True)
        config['DEFAULT'] = {'description': 'configurations for Explorepy'}
        config.add_section(section)

    config[section][var] = val

    try:
        with open(config_file, "w") as fid:
            config.write(fid)
    except OSError as error:
        error_msg = f"Could not open file: {config_file}"
        logger.error(error_msg)
        raise error(error_msg)


def uncaught_exception_handler(exctype, value, trace_back):
    """Handler of unhandled exceptions"""
    if exctype not in _IGNORED_EXC_BY_SENTRY:
        permission = None

        share_log_flag = read_config(USER_SETTING_KEY, SHARE_LOG_PERMISSION_KEY)
        if share_log_flag == "True":
            permission = True
        elif share_log_flag == "False":
            permission = False

        time.sleep(2)
        if permission is None:  # Then ask for permission
            while True:
                try:
                    txt = input("An unexpected error occurred! "
                                "Do you want to send the error log to Mentalab? (y/n) \n>")
                except (KeyboardInterrupt, EOFError):
                    sentry_sdk.init()  # disable sentry
                    break

                if txt in ['n', 'no', 'N', 'No']:
                    sentry_sdk.init()  # disable sentry
                    break
                if txt in ['y', 'yes', 'Y', 'Yes']:
                    logger.info("Thanks for helping us to improve Explorepy. Sending the error log to Mentalab ...")
                    break
        elif not permission:
            sentry_sdk.init()  # disable sentry
    else:
        sentry_sdk.init()  # disable sentry for ignored exceptions

    logger.error("Unhandled exception:", exc_info=(exctype, value, trace_back))


def log_breadcrumb(message, level):
    """Log breadcrumb messages to be sent to sentry"""
    sentry_sdk.add_breadcrumb(
        message=message,
        level=level
    )


def set_sentry_tag(tag_key, tag_value):
    """Set a tag in sentry"""
    sentry_sdk.set_tag(tag_key, tag_value)


sentry_sdk.init(
    "https://aefd994b53a54554b771899782581728@o522106.ingest.sentry.io/5633082",
    traces_sample_rate=1.0
)

setup_thread_excepthook()
sys.excepthook = uncaught_exception_handler
