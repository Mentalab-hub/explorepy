import logging
import logging.handlers
import sys
import os
import threading
from appdirs import user_log_dir


explorepy_logger = logging.getLogger('explorepy')
explorepy_logger.propagate = False
logger = logging.getLogger(__name__)

log_path = user_log_dir(appname="explorepy", appauthor="Mentalab")
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
        The 'sys.excepthook' bug has been fixed in Python 3.8 but as we want to keep support for older version of Python,
        we need this workaround.
    """
    init_original = threading.Thread.__init__

    def init(self, *args, **kwargs):

        init_original(self, *args, **kwargs)
        run_original = self.run

        def run_with_except_hook(*args2, **kwargs2):
            try:
                run_original(*args2, **kwargs2)
            except Exception:
                sys.excepthook(*sys.exc_info())

        self.run = run_with_except_hook

    threading.Thread.__init__ = init


def uncaught_exception_handler(exctype, value, tb):
    logger.error("Unhandled exception:", exc_info=(exctype, value, tb))


setup_thread_excepthook()
sys.excepthook = uncaught_exception_handler


