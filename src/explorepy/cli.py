# -*- coding: utf-8 -*-
import sys
import argparse
from explorepy.tools import bin2csv, bt_scan
from explorepy.explore import Explore
from explorepy.command import Command


class CLI:
    def __init__(self, command):
        getattr(self, command)()

    @staticmethod
    def find_device():
        parser = argparse.ArgumentParser(
            description='List available Explore devices.')

        parser.parse_args(sys.argv[2:])
        bt_scan()

    @staticmethod
    def acquire():
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

    @staticmethod
    def record_data():
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

    @staticmethod
    def push2lsl():
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

    @staticmethod
    def bin2csv():
        parser = argparse.ArgumentParser(
            description='Convert Binary data to CSV')

        parser.add_argument("-i", "--inputfile",
                            dest="inputfile", type=str, default=None,
                            help="Name of the Bin_File. ")

        parser.add_argument("-o", "--overwrite", action='store_false',
                            help="Overwrite files with same name.")

        args = parser.parse_args(sys.argv[2:])

        bin2csv(args.inputfile, args.overwrite)

    @staticmethod
    def visualize():
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

    @staticmethod
    def impedance():
        explorer = Explore()

        parser = argparse.ArgumentParser(
            description='Impedance measurement in a browser-based dashboard')

        parser.add_argument("-a", "--address",
                            dest="address", type=str, default=None,
                            help="Explore device's MAC address.")

        parser.add_argument("-n", "--name",
                            dest="name", type=str, default=None,
                            help="Name of the device.")

        parser.add_argument("-c", "--channels",
                            dest="channels", type=int, default=None,
                            help="the device's number of channels")

        parser.add_argument("-nf", "--notchfreq",
                            dest="notchfreq", type=int, default=50,
                            help="Frequency of notch filter.")

        args = parser.parse_args(sys.argv[2:])

        if args.name is None:
            explorer.connect(device_addr=args.address)
        else:
            explorer.connect(device_name=args.name)

        explorer.measure_imp(n_chan=args.channels, notch_freq=args.notchfreq)

    @staticmethod
    def format_memory():
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

        from explorepy import command
        memory_format_cmd = command.MemoryFormat()
        explorer.change_settings(memory_format_cmd)

    @staticmethod
    def set_sampling_rate():
        explorer = Explore()
        parser = argparse.ArgumentParser(
            description='format the memory of selected explore device (yet in alpha state)')

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

        from explorepy import command
        if args.sampling_rate is None:
            raise ValueError("Please specify the sampling rate")
        elif args.sampling_rate == '250':
            explorer.change_settings(command.SetSPS(250))
        elif args.sampling_rate == '500':
            explorer.change_settings(command.SetSPS(500))
        elif args.sampling_rate == '1000':
            explorer.change_settings(command.SetSPS(1000))
        else:
            raise ValueError("The only acceptable values are 250, 500 or 1000.")

    @staticmethod
    def soft_reset():
        explorer = Explore()
        parser = argparse.ArgumentParser(
            description='Terminate the recording session and reset the selected explore device')

        parser.add_argument("-a", "--address",
                            dest="address", type=str, default=None,
                            help="Explore device's MAC address.")

        parser.add_argument("-n", "--name",
                            dest="name", type=str, default=None,
                            help="Name of the device.")

        parser.add_argument("-i", "--deviceID",
                            dest="device_id", type=int, default=0,
                            help="ID of the device.")

        args = parser.parse_args(sys.argv[2:])

        if args.name is None:
            explorer.connect(device_addr=args.address)
        elif args.address is None:
            explorer.connect(device_name=args.name)

        from explorepy import command
        soft_reset_cmd = command.SoftReset()
        explorer.change_settings(soft_reset_cmd)



