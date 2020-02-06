# -*- coding: utf-8 -*-
import sys
import argparse
from explorepy.cli import CLI


def main():
    parser = argparse.ArgumentParser(
        description='Python package for the Mentalab Explore',
        usage='''explorepy <command> [args]

    Available Commands

    find_device:            Scans for nearby explore-devices. Prints out Name and MAC address of the found devices

    acquire:                Connects to device and streams data. needs either MAC or Name of the desired device as input
                            -a --address    Device MAC address (Form XX:XX:XX:XX:XX:XX).
                            -n --name       Device name (e.g. "Explore_12AB").


    record_data:            Connects to a device and records ExG and orientation data live to separate CSV files
                            Inputs: Name or Address, filename, overwrite flag
                            -a --address    Device MAC address (Form XX:XX:XX:XX:XX:XX).
                            -n --name       Device name (e.g. Explore_12AB). Either device name or MAC address is needed.
                            -f --filename   The prefix of the files.
                            -t --type       File type (either edf or csv
                            -o --overwrite  Overwrite already existing files with the same name.
                            -d --duration   Recording duration in seconds

    push2lsl                Streams Data to Lab stream layer. Inputs: Name or Address and Channel number (either 4 or 8)
                            -a --address    Device MAC address (Form XX:XX:XX:XX:XX:XX).
                            -n --name       Device name (e.g. Explore_12AB). Either device name or MAC address is needed.


    visualize               Visualizes real-time data in a browser-based dashboard
                            -a --address    Device MAC address (Form XX:XX:XX:XX:XX:XX).
                            -n --name       Device name (e.g. Explore_12AB). Either device name or MAC address is needed.
                            -nf --notchfreq Frequency of applied notch filter (By default, no notch filter is applied)
                            -lf --lowfreq   Low cutoff frequency of bandpass filter
                            -hf --highfreq  High cutoff frequency of bandpass filter


    impedance               Show electrode impedances
                            -a --address    Device MAC address (Form XX:XX:XX:XX:XX:XX).
                            -n --name       Device name (e.g. Explore_12AB). Either device name or MAC address is needed.
                            -nf --notchfreq Frequency of applied notch filter (By default, no notch filter is applied)


    bin2csv                Takes a Binary file and converts it to 3 CSV files (ExG, orientation and marker files)
                            -i --inputfile  Name of the input file
                            -o --overwrite  Overwrite already existing files with the same name.


    bin2edf                Takes a Binary file and converts it to 2 EDF+ files (Exg and orientation)
                            -i --inputfile  Name of the input file
                            -o --overwrite  Overwrite already existing files with the same name.



    format_memory           This command formats the memory
                            -a --address    Device MAC address (Form XX:XX:XX:XX:XX:XX).
                            -n --name       Device name (e.g. Explore_12AB).


    set_sampling_rate       This command sets the sampling rate of ExG input (yet in alpha state)
                            -a --address        Device MAC address (Form XX:XX:XX:XX:XX:XX).
                            -n --name           Device name (e.g. Explore_12AB).
                            -r --sampling_rate  Sampling rate of ExG channels, it can be 250, 500 or 1000.
    ''')

    parser.add_argument('command', help='Command to run.')
    args = parser.parse_args(sys.argv[1:2])

    if not hasattr(CLI, args.command):
        print('Incorrect usage. See help below.')
        parser.print_help()
        exit(1)

    cli = CLI(args.command)


if __name__ == "__main__":
    main()
