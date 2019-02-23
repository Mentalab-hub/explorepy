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
    explorer = explorepy.Explore()
    parser = argparse.ArgumentParser(
        description='Python package for the Mentalab Explore',
        usage="explorepy <command> [args]")
    """
    Available Commands
    
    find_device:            Scans for nearby explore-devices
    
    acquire:                Connects to device, needs MAC or Name of the desired device as input
    
    
    """

    args = parser.parse_args(sys.argv[1:2])

    parser.add_argument('command', help='Command to run.')

    if not hasattr(CLI, args.command):
        print('Incorrect usage. See help below.')
        parser.print_help()
        exit(1)

    cli = CLI(args.command)


if __name__ == "__main__":
    main()
