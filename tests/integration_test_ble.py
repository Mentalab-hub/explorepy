"""Integration test script for explorepy

Examples:
    $ python integration_test_ble.py -n Explore_1438
"""
import time

import explorepy
import argparse
from explorepy.settings_manager import SettingsManager


def main():
    parser = argparse.ArgumentParser(description="BLE Integration test at different sapling rates")
    parser.add_argument("-n", "--name", dest="name", type=str, help="Name of the device.")
    parser.add_argument("-d", "--duration", dest="duration", type=str, help="Duration of recording")

    args = parser.parse_args()
    # Create an Explore object
    exp_device = explorepy.Explore()
    # Connect to the Explore device using device bluetooth name or mac address
    try:
        exp_device.connect(device_name=args.name)
        duration = args.duration or 120
        sps_dict = {8: 1000, 16: 500, 32: 250}
        dev_ch = SettingsManager(args.name).get_channel_count()
        current_sps = sps_dict[dev_ch]
        exp_device.disconnect()
    except Exception as e:
        print('Got exception in setup of type {} and message {}'.format(type(e), e))

    while 250 <= current_sps <= 1000:
        try:
            exp_device.connect(device_name=args.name)
            exp_device.set_sampling_rate(sampling_rate=current_sps)
            # Push data to lsl. Note that this function creates three lsl streams, ExG, ORN and marker.
            exp_device.push2lsl()
            exp_device.record_data(f'{args.name + '_' + str(current_sps)}.csv',
                                   do_overwrite=True, duration=duration, block=True)
            exp_device.stop_lsl()
            exp_device.disconnect()
            time.sleep(5)
            current_sps = int(current_sps / 2)
        except Exception as e:
            print('Got exception {}', type(e))


if __name__ == "__main__":
    main()
