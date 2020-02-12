# -*- coding: utf-8 -*-
import sys
import argparse
from explorepy.tools import bin2csv, bt_scan, bin2edf
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
        explore = Explore()
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
            explore.connect(device_addr=args.address)
        elif args.address is None:
            explore.connect(device_name=args.name)
        explore.acquire()

    @staticmethod
    def record_data():
        explore = Explore()
        parser = argparse.ArgumentParser(
            description='Record data from a device with Specified name')

        parser.add_argument("-a", "--address",
                            dest="address", type=str, default=None,
                            help="Explore device's MAC address.")

        parser.add_argument("-n", "--name",
                            dest="name", type=str, default=None,
                            help="Name of the device.")

        parser.add_argument("-f", "--filename",
                            dest="filename", type=str, default=None,
                            help="Name of the CSV_Files.")

        parser.add_argument("-ow", "--overwrite", action='store_true', default=False,
                            help="Overwrite files with same name.")

        parser.add_argument("-d", "--duration", type=int, default=None,
                            help="Recording duration in seconds")

        parser.add_argument("-t", "--type",
                            dest="file_type", type=str, default='edf',
                            help="File type (edf or csv).")

        args = parser.parse_args(sys.argv[2:])

        if args.name is None:
            explore.connect(device_addr=args.address)
        else:
            explore.connect(device_name=args.name)

        assert (args.filename is not None), "Missing Filename"
        explore.record_data(file_name=args.filename, file_type=args.file_type,
                            do_overwrite=args.overwrite, duration=args.duration)

    @staticmethod
    def push2lsl():
        explore = Explore()
        parser = argparse.ArgumentParser(
            description='Push data to lsl')

        parser.add_argument("-a", "--address",
                            dest="address", type=str, default=None,
                            help="Explore device's MAC address.")

        parser.add_argument("-n", "--name",
                            dest="name", type=str, default=None,
                            help="Name of the device.")

        args = parser.parse_args(sys.argv[2:])

        if args.name is None:
            explore.connect(device_addr=args.address)
        else:
            explore.connect(device_name=args.name)

        explore.push2lsl()

    @staticmethod
    def bin2csv():
        parser = argparse.ArgumentParser(
            description='Convert Binary data to CSV')

        parser.add_argument("-i", "--inputfile",
                            dest="inputfile", type=str, default=None,
                            help="Name of the Bin_File.")

        parser.add_argument("-ow", "--overwrite", action='store_true', default=False,
                            help="Overwrite files with same name.")

        args = parser.parse_args(sys.argv[2:])

        bin2csv(bin_file=args.inputfile, do_overwrite=args.overwrite)

    @staticmethod
    def bin2edf():
        parser = argparse.ArgumentParser(
            description='Convert Binary data to EDF')

        parser.add_argument("-i", "--inputfile",
                            dest="inputfile", type=str, default=None,
                            help="Name of the Bin_File.")

        parser.add_argument("-ow", "--overwrite", action='store_true', default=False,
                            help="Overwrite files with same name.")

        args = parser.parse_args(sys.argv[2:])
        bin2edf(bin_file=args.inputfile, do_overwrite=args.overwrite)

    @staticmethod
    def visualize():
        explore = Explore()

        parser = argparse.ArgumentParser(
            description='Visualizing signal in a browser-based dashboard')

        parser.add_argument("-a", "--address",
                            dest="address", type=str, default=None,
                            help="Explore device's MAC address.")

        parser.add_argument("-n", "--name",
                            dest="name", type=str, default=None,
                            help="Name of the device.")

        parser.add_argument("-nf", "--notchfreq",
                            dest="notchfreq", type=int, default=None,
                            help="Frequency of notch filter.")

        parser.add_argument("-lf", "--lowfreq",
                            dest="lf", type=float, default=None,
                            help="Low cutoff frequency of bandpass filter.")

        parser.add_argument("-hf", "--highfreq",
                            dest="hf", type=float, default=None,
                            help="High cutoff frequency of bandpass filter.")

        parser.add_argument("-cf", "--calibration_file",
                            dest="cf", type=str, default=None,
                            help="Calibration file name")

        args = parser.parse_args(sys.argv[2:])

        if args.name is None:
            explore.connect(device_addr=args.address)
        else:
            explore.connect(device_name=args.name)

        if (args.lf is not None) and (args.hf is not None):
            explore.visualize(notch_freq=args.notchfreq, bp_freq=(args.lf, args.hf), calibre_file=args.cf)
        else:
            explore.visualize(notch_freq=args.notchfreq, bp_freq=None, calibre_file=args.cf)

    @staticmethod
    def impedance():
        explore = Explore()

        parser = argparse.ArgumentParser(
            description='Impedance measurement in a browser-based dashboard')

        parser.add_argument("-a", "--address",
                            dest="address", type=str, default=None,
                            help="Explore device's MAC address.")

        parser.add_argument("-n", "--name",
                            dest="name", type=str, default=None,
                            help="Name of the device.")

        parser.add_argument("-nf", "--notchfreq",
                            dest="notchfreq", type=int, default=50,
                            help="Frequency of notch filter.")

        args = parser.parse_args(sys.argv[2:])

        if args.name is None:
            explore.connect(device_addr=args.address)
        else:
            explore.connect(device_name=args.name)

        explore.measure_imp(notch_freq=args.notchfreq)

    @staticmethod
    def format_memory():
        explore = Explore()
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
            explore.connect(device_addr=args.address)
        elif args.address is None:
            explore.connect(device_name=args.name)

        from explorepy import command
        memory_format_cmd = command.MemoryFormat()
        explore.change_settings(memory_format_cmd)

    @staticmethod
    def set_sampling_rate():
        explore = Explore()
        parser = argparse.ArgumentParser(
            description='format the memory of selected explore device (yet in alpha state)')

        parser.add_argument("-a", "--address",
                            dest="address", type=str, default=None,
                            help="Explore device's MAC address.")

        parser.add_argument("-n", "--name",
                            dest="name", type=str, default=None,
                            help="Name of the device.")

        parser.add_argument("-sr", "--sampling_rate",
                            dest="sampling_rate", type=str, default=None,
                            help="Sampling rate of ExG channels, it can be 250 or 500")

        args = parser.parse_args(sys.argv[2:])

        if args.name is None:
            explore.connect(device_addr=args.address)
        elif args.address is None:
            explore.connect(device_name=args.name)

        from explorepy import command
        if args.sampling_rate is None:
            raise ValueError("Please specify the sampling rate")
        elif args.sampling_rate == '250':
            explore.change_settings(command.SetSPS(250))
        elif args.sampling_rate == '500':
            explore.change_settings(command.SetSPS(500))
        else:
            raise ValueError("The only acceptable values are 250 or 500")

    @staticmethod
    def soft_reset():
        explore = Explore()
        parser = argparse.ArgumentParser(
            description='Terminate the recording session and reset the selected explore device')

        parser.add_argument("-a", "--address",
                            dest="address", type=str, default=None,
                            help="Explore device's MAC address.")

        parser.add_argument("-n", "--name",
                            dest="name", type=str, default=None,
                            help="Name of the device.")

        args = parser.parse_args(sys.argv[2:])

        if args.name is None:
            explore.connect(device_addr=args.address)
        elif args.address is None:
            explore.connect(device_name=args.name)

        from explorepy import command
        soft_reset_cmd = command.SoftReset()
        explore.change_settings(soft_reset_cmd)

    @staticmethod
    def set_channels():
        explore = Explore()
        parser = argparse.ArgumentParser(
            description='Mask the channels of selected explore device (yet in alpha state)')

        parser.add_argument("-a", "--address",
                            dest="address", type=str, default=None,
                            help="Explore device's MAC address.")

        parser.add_argument("-n", "--name",
                            dest="name", type=str, default=None,
                            help="Name of the device.")

        parser.add_argument("-m", "--channel_mask",
                            dest="channel_mask", type=str, default=None,
                            help="Channel mask, it should be an integer between 1 and 255, the binarry representation "
                                 "will be interpreted as mask.")

        args = parser.parse_args(sys.argv[2:])

        if args.name is None:
            explore.connect(device_addr=args.address)
        elif args.address is None:
            explore.connect(device_name=args.name)

        from explorepy import command
        if args.channel_mask is None:
            raise ValueError("Please specify the mask")
        elif 1 <= int(args.channel_mask) <= 255:
            explore.change_settings(command.SetCh(int(args.channel_mask)))
        else:
            raise ValueError("Acceptable values are integers between 1 to 255.")

    @staticmethod
    def calibrate_orn():
        explore = Explore()
        parser = argparse.ArgumentParser(
            description='Calibrate the orientation module of the specified device')

        parser.add_argument("-a", "--address",
                            dest="address", type=str, default=None,
                            help="Explore device's MAC address.")

        parser.add_argument("-n", "--name",
                            dest="name", type=str, default=None,
                            help="Name of the device.")

        parser.add_argument("-cf", "--calibration_file",
                            dest="filename", type=str, default=None,
                            help="name of the calibration file starts with this string.")

        parser.add_argument("-ow", "--overwrite", action='store_true', default=False,
                            help="Overwrite files with same name.")

        args = parser.parse_args(sys.argv[2:])

        if args.name is None:
            explore.connect(device_addr=args.address)
        else:
            explore.connect(device_name=args.name)

        assert (args.filename is not None), "Missing Filename"
        explore.calibrate_orn(file_name=args.filename, do_overwrite=args.overwrite)


