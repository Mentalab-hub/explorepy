# -*- coding: utf-8 -*-
"""Command Line Interface module for explorepy"""
import click
import explorepy


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(context_settings=CONTEXT_SETTINGS, invoke_without_command=True)
@click.option("--version", "-V", help="Print explorepy version", is_flag=True)
@click.pass_context
def cli(ctx, version, args=None):
    """Python API for Mentalab biosignal aquisition devices"""
    if ctx.invoked_subcommand is None:
        if version:
            click.echo(explorepy.__version__)
        else:
            click.echo(ctx.get_help())


@cli.command()
def find_device():
    """List available Explore devices."""
    explorepy.tools.bt_scan()


@cli.command()
@click.option("--address", "-a", type=str, help="Explore device's MAC address")
@click.option("--name", "-n", type=str, help="Name of the device")
@click.option("--pybluez", is_flag=True, help="Use pybluez as the bluetooth interface")
def acquire(name, address, pybluez):
    """Connect to a device with selected name or address. Only one input is necessary"""
    if pybluez:
        explorepy.set_bt_interface('pybluez')
    explore = explorepy.explore.Explore()
    explore.connect(mac_address=address, device_name=name)
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
@click.option("--pybluez", is_flag=True, help="Use pybluez as the bluetooth interface")
def record_data(address, name, filename, overwrite, duration, file_type, pybluez):
    """Record data from Explore to a file """
    if pybluez:
        explorepy.set_bt_interface('pybluez')
    explore = explorepy.explore.Explore()
    explore.connect(mac_address=address, device_name=name)
    explore.record_data(file_name=filename, file_type=file_type,
                        do_overwrite=overwrite, duration=duration)


@cli.command()
@click.option("--address", "-a", type=str, help="Explore device's MAC address")
@click.option("--name", "-n", type=str, help="Name of the device")
@click.option("-d", "--duration", type=int, help="Streaming duration in seconds", metavar="<integer>")
@click.option("--pybluez", is_flag=True, help="Use pybluez as the bluetooth interface")
def push2lsl(address, name, duration, pybluez):
    """Push data to lsl"""
    if pybluez:
        explorepy.set_bt_interface('pybluez')
    explore = explorepy.explore.Explore()
    explore.connect(mac_address=address, device_name=name)
    explore.push2lsl(duration)


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
@click.option("-nf", "--notchfreq", type=click.Choice(['50', '60']), help="Frequency of notch filter.", default='50')
@click.option("-lf", "--lowfreq", type=float, help="Low cutoff frequency of bandpass/highpass filter.")
@click.option("-hf", "--highfreq", type=float, help="High cutoff frequency of bandpass/lowpass filter.")
@click.option("-cf", "--calib-file",
              help="Calibration file name. If you pass this parameter, ORN module should be ACTIVE! "
                   "To obtain this file refer to Explore.calibrate_orn module.",
              type=click.Path(exists=True))
@click.option("--pybluez", is_flag=True, help="Use pybluez as the bluetooth interface")
def visualize(address, name, notchfreq, lowfreq, highfreq, calib_file, pybluez):
    """Visualizing signal in a browser-based dashboard"""
    if pybluez:
        explorepy.set_bt_interface('pybluez')
    explore = explorepy.explore.Explore()
    explore.connect(mac_address=address, device_name=name)
    explore.visualize(notch_freq=int(notchfreq), bp_freq=(lowfreq, highfreq), calibre_file=calib_file)


@cli.command()
@click.option("--address", "-a", type=str, help="Explore device's MAC address")
@click.option("--name", "-n", type=str, help="Name of the device")
@click.option("-nf", "--notchfreq", type=click.Choice(['50', '60']), help="Frequency of notch filter.", default='50')
@click.option("--pybluez", is_flag=True, help="Use pybluez as the bluetooth interface")
def impedance(address, name, notchfreq, pybluez):
    """Impedance measurement in a browser-based dashboard"""
    if pybluez:
        explorepy.set_bt_interface('pybluez')
    explore = explorepy.explore.Explore()
    explore.connect(mac_address=address, device_name=name)
    explore.measure_imp(notch_freq=int(notchfreq))


@cli.command()
@click.option("--address", "-a", type=str, help="Explore device's MAC address")
@click.option("--name", "-n", type=str, help="Name of the device")
@click.option("--pybluez", is_flag=True, help="Use pybluez as the bluetooth interface")
def format_memory(address, name, pybluez):
    """format the memory of Explore device"""
    if pybluez:
        explorepy.set_bt_interface('pybluez')
    explore = explorepy.explore.Explore()
    explore.connect(mac_address=address, device_name=name)
    explore.format_memory()


@cli.command()
@click.option("--address", "-a", type=str, help="Explore device's MAC address")
@click.option("--name", "-n", type=str, help="Name of the device")
@click.option("-sr", "--sampling-rate", help="Sampling rate of ExG channels, it can be 250 or 500",
              type=click.Choice(['250', '500', '1000']), required=True)
@click.option("--pybluez", is_flag=True, help="Use pybluez as the bluetooth interface")
def set_sampling_rate(address, name, sampling_rate, pybluez):
    """Change sampling rate of the Explore device"""
    if pybluez:
        explorepy.set_bt_interface('pybluez')
    explore = explorepy.explore.Explore()
    explore.connect(mac_address=address, device_name=name)
    explore.set_sampling_rate(int(sampling_rate))


@cli.command()
@click.option("--address", "-a", type=str, help="Explore device's MAC address")
@click.option("--name", "-n", type=str, help="Name of the device")
@click.option("--pybluez", is_flag=True, help="Use pybluez as the bluetooth interface")
def soft_reset(address, name, pybluez):
    """Software reset of Explore device"""
    if pybluez:
        explorepy.set_bt_interface('pybluez')
    """Reset the selected explore device (current session will be terminated)."""
    explore = explorepy.explore.Explore()
    explore.connect(mac_address=address, device_name=name)
    explore.reset_soft()


@cli.command()
@click.option("--address", "-a", type=str, help="Explore device's MAC address")
@click.option("--name", "-n", type=str, help="Name of the device")
@click.option("-m", "--channel-mask", type=click.IntRange(min=1, max=255), required=True,
              help="Channel mask, it should be an integer between 1 and 255, the binary representation will be "
                   "interpreted as mask.")
@click.option("--pybluez", is_flag=True, help="Use pybluez as the bluetooth interface")
def set_channels(address, name, channel_mask, pybluez):
    """Mask the channels of selected explore device"""
    if pybluez:
        explorepy.set_bt_interface('pybluez')
    explore = explorepy.explore.Explore()
    explore.connect(mac_address=address, device_name=name)
    explore.set_channels(channel_mask)


@cli.command()
@click.option("--address", "-a", type=str, help="Explore device's MAC address")
@click.option("--name", "-n", type=str, help="Name of the device")
@click.option("-m", "--module", required=True, type=str, help="Module name to be disabled, options: ORN, ENV, EXG")
@click.option("--pybluez", is_flag=True, help="Use pybluez as the bluetooth interface")
def disable_module(address, name, module, pybluez):
    """Disable a module of Explore device"""
    if pybluez:
        explorepy.set_bt_interface('pybluez')
    explore = explorepy.explore.Explore()
    explore.connect(mac_address=address, device_name=name)
    explore.disable_module(module)


@cli.command()
@click.option("--address", "-a", type=str, help="Explore device's MAC address")
@click.option("--name", "-n", type=str, help="Name of the device")
@click.option("-m", "--module", required=True, type=str, help="Module name to be enabled, options: ORN, ENV, EXG")
@click.option("--pybluez", is_flag=True, help="Use pybluez as the bluetooth interface")
def enable_module(address, name, module, pybluez):
    """Enable a module of Explore device"""
    if pybluez:
        explorepy.set_bt_interface('pybluez')
    explore = explorepy.explore.Explore()
    explore.connect(mac_address=address, device_name=name)
    explore.enable_module(module)


@cli.command()
@click.option("--address", "-a", type=str, help="Explore device's MAC address")
@click.option("--name", "-n", type=str, help="Name of the device")
@click.option("-f", "--filename", help="Name of the file.", required=True,
              type=click.Path(file_okay=True, dir_okay=True, resolve_path=True))
@click.option("-ow", "--overwrite", is_flag=True, help="Overwrite existing file")
@click.option("--pybluez", is_flag=True, help="Use pybluez as the bluetooth interface")
def calibrate_orn(address, name, filename, overwrite, pybluez):
    """Calibrate the orientation module of the specified device"""
    if pybluez:
        explorepy.set_bt_interface('pybluez')
    explore = explorepy.explore.Explore()
    explore.connect(mac_address=address, device_name=name)
    explore.calibrate_orn(file_name=filename, do_overwrite=overwrite)
