# -*- coding: utf-8 -*-
import sys
import argparse
from explorepy.tools import bin2csv, bt_scan
from explorepy.explore import Explore
from explorepy.command import Command


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
        explorer = Explore()
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
        explorer = Explore()
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

        parser.add_argument("-d", "--duration", type=int, default=None,
                            help="Recording duration in seconds")

        args = parser.parse_args(sys.argv[2:])

        if args.name is None:
            explorer.connect(device_addr=args.address)
        else:
            explorer.connect(device_name=args.name)

        assert (args.filename is not None), "Missing Filename"
        explorer.record_data(file_name=args.filename, do_overwrite=args.overwrite, duration=args.duration)

    def push2lsl(self):
        self.is_not_used()
        explorer = Explore()
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
                            help="the device's number of channels")

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

    def visualize(self):
        self.is_not_used()
        explorer = Explore()

        parser = argparse.ArgumentParser(
            description='Visualizing signal in a browser-based dashboard')

        parser.add_argument("-a", "--address",
                            dest="address", type=str, default=None,
                            help="Explore device's MAC address.")

        parser.add_argument("-n", "--name",
                            dest="name", type=str, default=None,
                            help="Name of the device.")

        parser.add_argument("-c", "--channels",
                            dest="channels", type=int, default=None,
                            help="the device's number of channels (2, 4 or 8)")

        parser.add_argument("-nf", "--notchfreq",
                            dest="notchfreq", type=int, default=None,
                            help="Frequency of notch filter.")

        args = parser.parse_args(sys.argv[2:])

        if args.name is None:
            explorer.connect(device_addr=args.address)
        else:
            explorer.connect(device_name=args.name)

        explorer.visualize(n_chan=args.channels)

    def is_not_used(self):
        pass

    def pass_msg(self):
        self.is_not_used()
        explorer = Explore()
        parser = argparse.ArgumentParser(
            description='Connect to a device with selected name or address. and send the desired message')

        parser.add_argument("-a", "--address",
                            dest="address", type=str, default=None,
                            help="Explore device's MAC address.")

        parser.add_argument("-n", "--name",
                            dest="name", type=str, default=None,
                            help="Name of the device.")

        parser.add_argument("-m", "--message",
                            dest="message", type=bytearray, default=None,
                            help="the command to be sent.")

        args = parser.parse_args(sys.argv[2:])

        if args.name is None:
            explorer.connect(device_addr=args.address)
        elif args.address is None:
            explorer.connect(device_name=args.name)
        explorer.pass_msg(msg2send=args.message)

    def format_memory(self):
        self.is_not_used()
        explorer = Explore()
        parser = argparse.ArgumentParser(
            description='format the memory of selected explore device')

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
        explorer.pass_msg(msg2send=Command.FORMAT_MEMORY.value)

    def set_sampling_rate(self):
        self.is_not_used()
        explorer = Explore()
        parser = argparse.ArgumentParser(
            description='format the memory of selected explore device')

        parser.add_argument("-a", "--address",
                            dest="address", type=str, default=None,
                            help="Explore device's MAC address.")

        parser.add_argument("-n", "--name",
                            dest="name", type=str, default=None,
                            help="Name of the device.")

        parser.add_argument("-r", "--sampling_rate",
                            dest="sampling_rate", type=str, default=None,
                            help="Sampling rate of ExG channels, it can be 250, 500 or 1000.")

        args = parser.parse_args(sys.argv[2:])

        if args.name is None:
            explorer.connect(device_addr=args.address)
        elif args.address is None:
            explorer.connect(device_name=args.name)
        if args.sampling_rate is None:
            print("Please specify the sampling rate")
        elif args.sampling_rate == '250':
            explorer.pass_msg(msg2send=Command.SPS_250.value)
        elif args.sampling_rate == '500':
            explorer.pass_msg(msg2send=Command.SPS_250.value)
        elif args.sampling_rate == '1000':
            explorer.pass_msg(msg2send=Command.SPS_250.value)
        else:
            print("The only acceptable values are 250, 500 or 1000.")


