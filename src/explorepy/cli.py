"""
Module that contains the command line app.

Why does this file exist, and why not put this in __main__?

  You might be tempted to import things from __main__ later, but that will cause
  problems: the code will get executed twice:

  - When you run `python -mexplorepy` python will execute
    ``__main__.py`` as a script. That means there won't be any
    ``explorepy.__main__`` in ``sys.modules``.
  - When you import __main__ it will get executed again (as a module) because
    there's no ``explorepy.__main__`` in ``sys.modules``.

  Also see (1) from http://click.pocoo.org/5/setuptools/#setuptools-integration
"""
import sys
import argparse


class CLI:
    def __init__(self, command):
    # use dispatch pattern to invoke method with same name
        getattr(self, command)()

    def livestream(self):
        parser = argparse.ArgumentParser(
            description='Stream Data live from the mentalab explore device')
        parser.add_argument("-s", "--stream",
                            dest="stream", type=str, default=None,)

        args = parser.parse_args(sys.argv)

        explorer = explore.Explore()

        explorer.connect(device_id=0)
        explorer.acquire(device_id=0)

    def select_device(self):
        return

    def push2LSL(self):
        return

    def record_data(self):
        return

