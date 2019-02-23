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
from explorepy.tools import bt_scan
import explorepy


class CLI:
    def __init__(self, command):
    # use dispatch pattern to invoke method with same name
        getattr(self, command)()


    def find_device(self):
        parser = argparse.ArgumentParser(
            description='List available Explore devices.')
        bt_scan()

    def acquire(self):

        explorer = explorepy.Explore()
        parser = argparse.ArgumentParser(
            description='Connect to a device with selected name or address. Only one input is necessary')

        parser.add_argument("-a", "--address",
                            dest="address", type=str, default=None,
                            help="Explore device's MAC address.")

        parser.add_argument("-n", "--name",
                            dest="name", type=str, default=None,
                            help="Name of the device.")

        args = parser.parse_args(sys.argv[2:])

        if args.name is None:
            explorer.connect(args.address)
        else:
            explorer.connect(args.name)
        explorer.acquire()

    def record2CSV(self):

        explorer = explorepy.Explore()
        parser = argparse.ArgumentParser(
            description = 'Connect to a device with selected name')

        parser.add_argument("-a", "--address",
                            dest="address", type=str, default=None,
                            help="Explore device's MAC address.")

        parser.add_argument("-n", "--name",
                            dest="name", type=str, default=None,
                            help="Name of the device.")

        parser.add_argument("-f", "--filename",
                            dest="filename", type=str, default=None,
                            help="Name of the CSV_Files.")

        parser.add_argument("-o", "--overwrite", dest="overwrite", type=str, default=None,
                            help="Overwrite files with same name")

        args = parser.parse_args(sys.argv[2:])

        if args.name is None:
            explorer.connect(args.address)
        else:
            explorer.connect(args.name)

        if args.overwrite is not None:
            do_overwrite = True

        explorer.record_data(args.filename, do_overwrite)

    def push2LSL(self):

        explorer = explorepy.Explore()
        parser = argparse.ArgumentParser(
            description = 'Push data to lsl')

        parser.add_argument("-a", "--address",
                            dest="address", type=str, default=None,
                            help="Explore device's MAC address.")

        parser.add_argument("-n", "--name",
                            dest="name", type=str, default=None,
                            help="Name of the device.")

        args = parser.parse_args(sys.argv[2:])

        if args.name is None:
            explorer.connect(args.address)
        else:
            explorer.connect(args.name)

        explorer.push2lsl()
