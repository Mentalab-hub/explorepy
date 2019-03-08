"""
Entrypoint module, in case you use `python -mexplorepy`.


Why does this file exist, and why __main__? For more info, read:

- https://www.python.org/dev/peps/pep-0338/
- https://docs.python.org/2/using/cmdline.html#cmdoption-m
- https://docs.python.org/3/using/cmdline.html#cmdoption-m
"""
import sys
import argparse
import explorepy
from explorepy.cli import CLI


def main():
    parser = argparse.ArgumentParser(
        description='Python package for the Mentalab Explore',
        usage='''explorepy <command> [args]
    
    Available Commands
    
    find_device:            Scans for nearby explore-devices. Prints out Name and MAC address of the found devices
    
    acquire:                Connects to device, needs either MAC or Name of the desired device as input
                            -a --address    Device MAC address (Form XX:XX:XX:XX:XX:XX). 
                            -n --name       Device name (e.g. "Explore_12AB").
                                            
    
    record_data:             Connects to a device and records Orientation and Body data live to 2 separate CSV files
                            Inputs: Name or Address, filename, overwrite flag
                            
                            -a --address    Device MAC address (Form XX:XX:XX:XX:XX:XX). 
                            -n --name       Device name (e.g. Explore_12AB).
                            -f --filename   The name of the new CSV Files. 
                            -o --overwrite  Overwrite already existing files with the same name.
                            
    push2lsl                Streams Data to Lab stream layer. Inputs: Name or Address and Channel number (either 4 or 8)
    
                            -a --address    Device MAC address (Form XX:XX:XX:XX:XX:XX). 
                            -n --name       Device name (e.g. Explore_12AB).
                            -c --channels   Number of channels. This is necessary for push2lsl
    
    bin2csv                Takes a Binary file and converts it to 2 CSV files (orientation and Body)
    
                            -i --inputfile  Name of the input file
                            -o --overwrite  Overwrite already existing files with the same name.
    
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
