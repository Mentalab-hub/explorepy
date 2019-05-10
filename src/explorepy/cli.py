import sys
import argparse
from explorepy.tools import bin2csv, bt_scan
import explorepy


class CLI:
    def __init__(self, command):
        getattr(self, command)()

    def find_device(self):
        self.is_not_used()
        parser = argparse.ArgumentParser(
            description='List available Explore devices.')

        parser.parse_args(sys.argv[2:])
        bt_scan()

    def acquire(self):
        self.is_not_used()
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
            explorer.connect(device_addr=args.address)
        elif args.address is None:
            explorer.connect(device_name=args.name)
        explorer.acquire()

    def record_data(self):
        self.is_not_used()
        explorer = explorepy.Explore()
        parser = argparse.ArgumentParser(
            description='Connect to a device with selected name')

        parser.add_argument("-a", "--address",
                            dest="address", type=str, default=None,
                            help="Explore device's MAC address.")

        parser.add_argument("-n", "--name",
                            dest="name", type=str, default=None,
                            help="Name of the device.")

        parser.add_argument("-f", "--filename",
                            dest="filename", type=str, default=None,
                            help="Name of the CSV_Files.")

        parser.add_argument("-o", "--overwrite", action='store_false',
                            help="Overwrite files with same name.")

        args = parser.parse_args(sys.argv[2:])

        if args.name is None:
            explorer.connect(device_addr=args.address)
        else:
            explorer.connect(device_name=args.name)

        assert (args.filename is not None), "Missing Filename"
        explorer.record_data(args.filename, args.overwrite)

    def push2lsl(self):
        self.is_not_used()
        explorer = explorepy.Explore()
        parser = argparse.ArgumentParser(
            description='Push data to lsl')

        parser.add_argument("-a", "--address",
                            dest="address", type=str, default=None,
                            help="Explore device's MAC address.")

        parser.add_argument("-n", "--name",
                            dest="name", type=str, default=None,
                            help="Name of the device.")

        parser.add_argument("-c", "--channels",
                            dest="channels", type=int, default=None,
                            help="the device's number of channels (4 or 8)")

        args = parser.parse_args(sys.argv[2:])

        if args.name is None:
            explorer.connect(device_addr=args.address)
        else:
            explorer.connect(device_name=args.name)

        explorer.push2lsl(n_chan=args.channels)

    def bin2csv(self):
        self.is_not_used()
        parser = argparse.ArgumentParser(
            description='Convert Binary data to CSV')

        parser.add_argument("-i", "--inputfile",
                            dest="inputfile", type=str, default=None,
                            help="Name of the Bin_File. ")

        parser.add_argument("-o", "--overwrite", action='store_false',
                            help="Overwrite files with same name.")

        args = parser.parse_args(sys.argv[2:])

        bin2csv(args.inputfile, args.overwrite)

    def is_not_used(self):
        pass
