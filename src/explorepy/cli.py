# -*- coding: utf-8 -*-
"""Command Line Interface module for explorepy"""
import logging
import sys
from functools import update_wrapper

import click

import explorepy


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
logger = logging.getLogger(__name__)

default_bt_backend = explorepy.get_bt_interface()


@click.group(context_settings=CONTEXT_SETTINGS, invoke_without_command=True)
@click.option("--version", "-V", help="Print explorepy version", is_flag=True)
@click.pass_context
def cli(ctx, version, args=None):
    """Python API for Mentalab biosignal aquisition devices"""
    logger.debug(sys.argv)
    if ctx.invoked_subcommand is None:
        if version:
            click.echo(explorepy.__version__)
        else:
            click.echo(ctx.get_help())


def verify_inputs(func):
    @click.pass_context
    def wrapper(ctx, *args, **kwargs):
        if kwargs['name'] is None and kwargs['address'] is None:
            logger.warning("Either name or address must be given!")
            sys.exit()
        if kwargs['name'] is not None:
            if kwargs['name'][:8] != "Explore_":
                logger.warning("Invalid device name! Please check the device name and try again.")
                sys.exit()
        if kwargs['address'] is not None:
            if ':' in kwargs['address']:
                separator = ':'
            elif '-' in kwargs['address']:
                separator = '-'
            else:
                separator = ''

            if separator == '-' or separator == ':':
                for unit in kwargs['address'].split(separator):
                    if len(unit) != 2:
                        logger.warning("Invalid device mac address! Please check the MAC address and try again.")
                        sys.exit()
        return ctx.invoke(func, *args, **kwargs)

    return update_wrapper(wrapper, func)


@cli.command()
@click.option("--address", "-a", type=str, help="Explore device's MAC address")
@click.option("--name", "-n", type=str, help="Name of the device")
@click.option("-d", "--duration", type=int, help="Duration in seconds", metavar="<integer>")
@verify_inputs
def acquire(name, address, duration):
    """Connect to a device and print the ExG stream in the console"""
    explore = explorepy.explore.Explore()
    explore.connect(mac_address=address, device_name=name)
    explore.acquire(duration)


@cli.command()
@click.option("--address", "-a", type=str, help="Explore device's MAC address")
@click.option("--name", "-n", type=str, help="Name of the device")
@click.option("-f", "--filename", help="Name of the file.", required=True,
              type=click.Path(file_okay=True, dir_okay=True, resolve_path=True))
@click.option("-ow", "--overwrite", is_flag=True, help="Overwrite existing file")
@click.option("-d", "--duration", type=int, help="Recording duration in seconds", metavar="<integer>")
@click.option("--edf", 'file_type', flag_value='edf', help="Write in EDF file")
@click.option("--csv", 'file_type', flag_value='csv', help="Write in csv file (default type)", default=True)
@verify_inputs
def record_data(address, name, filename, overwrite, duration, file_type):
    """Record data from Explore to a file"""
    explore = explorepy.explore.Explore()
    explore.connect(mac_address=address, device_name=name)
    explore.record_data(file_name=filename, file_type=file_type,
                        do_overwrite=overwrite, duration=duration, block=True)


@cli.command()
@click.option("--address", "-a", type=str, help="Explore device's MAC address")
@click.option("--name", "-n", type=str, help="Name of the device")
@click.option("-d", "--duration", type=int, help="Streaming duration in seconds", metavar="<integer>")
@verify_inputs
def push2lsl(address, name, duration):
    """Push data to lsl"""
    explore = explorepy.explore.Explore()
    explore.connect(mac_address=address, device_name=name)
    explore.push2lsl(duration, block=True)


@cli.command()
@click.option("--address", "-a", type=str, help="Explore device's MAC address")
@click.option("--name", "-n", type=str, help="Name of the device")
@click.option("-d", "--duration", type=int, help="Recording duration in seconds", metavar="<integer>")
@click.option("-f", "--filename", help="Name of the output file", default="exg_data_imp_mode")
@click.option("-nf", "--notch-freq", help="Notch frequency for impedance mode initialization", type=float, default=50.0)
@click.option("-ow", "--overwrite", is_flag=True, help="Overwrite existing file")
@verify_inputs
def live_impedance_mode(address, name, duration, filename, notch_freq, overwrite):
    """Record data in impedance mode with live impedance monitoring"""
    explore = explorepy.explore.Explore()
    explore.connect(mac_address=address, device_name=name)
    explore.live_impedance_mode(file_name=filename, duration=duration,
                                notch_freq=notch_freq, do_overwrite=overwrite, block=True)


@cli.command()
@click.option("-f", "--filename", help="Name of (and path to) the binary file.", required=True,
              type=click.Path(exists=True, file_okay=True, dir_okay=True, resolve_path=True))
@click.option("-ow", "--overwrite", is_flag=True, help="Overwrite existing file")
def bin2csv(filename, overwrite):
    """Convert a binary file to CSV"""
    explore = explorepy.explore.Explore()
    explore.convert_bin(bin_file=filename, do_overwrite=overwrite, file_type='csv')


@cli.command()
@click.option("-f", "--filename", help="Name of (and path to) the binary file.", required=True,
              type=click.Path(exists=True, file_okay=True, dir_okay=True, resolve_path=True))
@click.option("-ow", "--overwrite", is_flag=True, help="Overwrite existing file")
def bin2edf(filename, overwrite):
    """Convert a binary file to EDF (BDF+)"""
    explore = explorepy.explore.Explore()
    explore.convert_bin(bin_file=filename, do_overwrite=overwrite, file_type='edf')


@cli.command()
@click.option("--address", "-a", type=str, help="Explore device's MAC address")
@click.option("--name", "-n", type=str, help="Name of the device")
@verify_inputs
def format_memory(address, name):
    """Format the memory of Explore device"""
    explore = explorepy.explore.Explore()
    explore.connect(mac_address=address, device_name=name)
    explore.format_memory()


@cli.command()
@click.option("--address", "-a", type=str, help="Explore device's MAC address")
@click.option("--name", "-n", type=str, help="Name of the device")
@click.option("-sr", "--sampling-rate", help="Sampling rate of ExG channels, it can be 250 or 500",
              type=click.Choice(['250', '500', '1000']), required=True)
@verify_inputs
def set_sampling_rate(address, name, sampling_rate):
    """Change sampling rate of the Explore device"""
    explore = explorepy.explore.Explore()
    explore.connect(mac_address=address, device_name=name)
    explore.set_sampling_rate(int(sampling_rate))


@cli.command()
@click.option("--address", "-a", type=str, help="Explore device's MAC address")
@click.option("--name", "-n", type=str, help="Name of the device")
@verify_inputs
def soft_reset(address, name):
    """Software reset of Explore device

    Reset the selected explore device (current session will be terminated)."""
    explore = explorepy.explore.Explore()
    explore.connect(mac_address=address, device_name=name)
    explore.reset_soft()


@cli.command()
@click.option("--address", "-a", type=str, help="Explore device's MAC address")
@click.option("--name", "-n", type=str, help="Name of the device")
@click.option("-m", "--channel-mask", type=str, required=True,
              help="Channel mask, it should be a binary string containing 1 and 0, "
                   "representing the mask (LSB is channel 1).")
@verify_inputs
def set_channels(address, name, channel_mask):
    """Mask the channels of the Explore device"""
    explore = explorepy.explore.Explore()
    explore.connect(mac_address=address, device_name=name)
    explore.set_channels(channel_mask)
