"""
Entrypoint module, in case you use `python -mexplorepy`.


Why does this file exist, and why __main__? For more info, read:

- https://www.python.org/dev/peps/pep-0338/
- https://docs.python.org/2/using/cmdline.html#cmdoption-m
- https://docs.python.org/3/using/cmdline.html#cmdoption-m
"""
import sys
import argparse
from explorepy.cli import CLI


def main():
    parser = argparse.ArgumentParser(
        description='Python package for the Mentalab Explore',
        usage="Usage")

    args = parser.parse_args(sys.argv)

    cli = CLI(args.command)


if __name__ == "__main__":
    sys.exit(main())
