# -*- coding: utf-8 -*-
from explorepy.tools import bin2csv, bt_scan, bin2edf
from explorepy.explore import Explore
import click

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(context_settings=CONTEXT_SETTINGS, invoke_without_command=True)
@click.option("--version", "-V", help="Print explorepy version", is_flag=True)
@click.pass_context
def cli(ctx, version, args=None):
    """Python API for Mentalab biosignal aquisition devices"""
    if ctx.invoked_subcommand is None:
        if version:
            import explorepy
            click.echo(explorepy.__version__)
        else:
            click.echo(ctx.get_help())


@cli.command()
def find_device():
    """List available Explore devices."""
    bt_scan()


@cli.command()
@click.option("--address", "-a", type=str, help="Explore device's MAC address")
@click.option("--name", "-n", type=str, help="Name of the device")
def acquire(name, address):
    """Connect to a device with selected name or address. Only one input is necessary"""
    explore = Explore()
    explore.connect(device_addr=address, device_name=name)
    explore.acquire()


@cli.command()
@click.option("--address", "-a", type=str, help="Explore device's MAC address")
@click.option("--name", "-n", type=str, help="Name of the device")
@click.option("-f", "--filename", help="Name of the file.", required=True,
              type=click.Path(file_okay=True, dir_okay=True, resolve_path=True))
@click.option("-ow", "--overwrite", is_flag=True, help="Overwrite existing file")
@click.option("-d", "--duration", type=int, help="Recording duration in seconds", metavar="<integer>")
@click.option("--edf", 'file_type', flag_value='edf', help="Write in EDF file (default type)", default=True)
@click.option("--csv", 'file_type', flag_value='csv', help="Write in csv file")
def record_data(address, name, filename, overwrite, duration, file_type):
    """Record data from Explore to a file """
    explore = Explore()
    explore.connect(device_addr=address, device_name=name)
    explore.record_data(file_name=filename, file_type=file_type,
                        do_overwrite=overwrite, duration=duration)


@cli.command()
@click.option("--address", "-a", type=str, help="Explore device's MAC address")
@click.option("--name", "-n", type=str, help="Name of the device")
def push2lsl(address, name):
    """Push data to lsl"""
    explore = Explore()
    explore.connect(device_addr=address, device_name=name)
    explore.push2lsl()


@cli.command()
@click.option("-f", "--filename", help="Name of (and path to) the binary file.", required=True,
              type=click.Path(exists=True, file_okay=True, dir_okay=True, resolve_path=True))
@click.option("-ow", "--overwrite", is_flag=True, help="Overwrite existing file")
def bin2csv(filename, overwrite):
    """Convert a binary file to CSV"""
    bin2csv(bin_file=filename, do_overwrite=overwrite)


@cli.command()
@click.option("-f", "--filename", help="Name of (and path to) the binary file.", required=True,
              type=click.Path(exists=True, file_okay=True, dir_okay=True, resolve_path=True))
@click.option("-ow", "--overwrite", is_flag=True, help="Overwrite existing file")
def bin2edf(filename, overwrite):
    """Convert a binary file to EDF (BDF+)"""
    bin2edf(bin_file=filename, do_overwrite=overwrite)


@cli.command()
@click.option("--address", "-a", type=str, help="Explore device's MAC address")
@click.option("--name", "-n", type=str, help="Name of the device")
@click.option("-nf", "--notchfreq", type=click.Choice(['50', '60']), help="Frequency of notch filter.", default='50')
@click.option("-lf", "--lowfreq", type=float, help="Low cutoff frequency of bandpass filter.")
@click.option("-hf", "--highfreq", type=float, help="High cutoff frequency of bandpass filter.")
@click.option("-cf", "--calib-file", help="Calibration file name", type=click.Path(exists=True))
def visualize(address, name, notchfreq, lowfreq, highfreq, calib_file):
    """Visualizing signal in a browser-based dashboard"""
    explore = Explore()
    explore.connect(device_addr=address, device_name=name)

    if (lowfreq is not None) and (highfreq is not None):
        explore.visualize(notch_freq=int(notchfreq), bp_freq=(lowfreq, highfreq), calibre_file=calib_file)
    else:
        explore.visualize(notch_freq=int(notchfreq), bp_freq=None, calibre_file=calib_file)


@cli.command()
@click.option("--address", "-a", type=str, help="Explore device's MAC address")
@click.option("--name", "-n", type=str, help="Name of the device")
@click.option("-nf", "--notchfreq", type=click.Choice(['50', '60']), help="Frequency of notch filter.", default='50')
def impedance(address, name, notchfreq):
    """Impedance measurement in a browser-based dashboard"""
    explore = Explore()
    explore.connect(device_addr=address, device_name=name)

    explore.measure_imp(notch_freq=int(notchfreq))


@cli.command()
@click.option("--address", "-a", type=str, help="Explore device's MAC address")
@click.option("--name", "-n", type=str, help="Name of the device")
def format_memory(address, name):
    """format the memory of Explore device"""
    explore = Explore()
    explore.connect(device_addr=address, device_name=name)

    from explorepy import command
    memory_format_cmd = command.MemoryFormat()
    explore.change_settings(memory_format_cmd)


@cli.command()
@click.option("--address", "-a", type=str, help="Explore device's MAC address")
@click.option("--name", "-n", type=str, help="Name of the device")
@click.option("-sr", "--sampling-rate", help="Sampling rate of ExG channels, it can be 250 or 500",
              type=click.Choice(['250', '500']), required=True)
def set_sampling_rate(address, name, sampling_rate):
    """Change sampling rate of the Explore device"""
    explore = Explore()
    explore.connect(device_addr=address, device_name=name)

    from explorepy import command
    explore.change_settings(command.SetSPS(int(sampling_rate)))


@cli.command()
@click.option("--address", "-a", type=str, help="Explore device's MAC address")
@click.option("--name", "-n", type=str, help="Name of the device")
def soft_reset(address, name):
    """Reset the selected explore device (current session will be terminated)."""
    explore = Explore()
    explore.connect(device_addr=address, device_name=name)

    from explorepy import command
    soft_reset_cmd = command.SoftReset()
    explore.change_settings(soft_reset_cmd)


@cli.command()
@click.option("--address", "-a", type=str, help="Explore device's MAC address")
@click.option("--name", "-n", type=str, help="Name of the device")
@click.option("-m", "--channel-mask", type=click.IntRange(min=1, max=255), required=True,
              help="Channel mask, it should be an integer between 1 and 255, the binary representation will be "
                   "interpreted as mask.")
def set_channels(address, name, channel_mask):
    """Mask the channels of selected explore device (yet in alpha state)"""
    explore = Explore()
    explore.connect(device_addr=address, device_name=name)

    from explorepy import command
    explore.change_settings(command.SetCh(channel_mask))


@cli.command()
@click.option("--address", "-a", type=str, help="Explore device's MAC address")
@click.option("--name", "-n", type=str, help="Name of the device")
@click.option("-f", "--filename", help="Name of the file.", required=True,
              type=click.Path(file_okay=True, dir_okay=True, resolve_path=True))
@click.option("-ow", "--overwrite", is_flag=True, help="Overwrite existing file")
def calibrate_orn(address, name, filename, overwrite):
    """Calibrate the orientation module of the specified device"""
    explore = Explore()
    explore.connect(device_addr=address, device_name=name)
    explore.calibrate_orn(file_name=filename, do_overwrite=overwrite)


if __name__ == "__main__":
    cli()
