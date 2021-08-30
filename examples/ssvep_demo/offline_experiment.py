# -*- coding: utf-8 -*-
"""
A script to run a simple offline SSVEP Experiment with Mentalab's Explore device
"""
import argparse
import explorepy
from ssvep import SSVEPExperiment


def main():
    parser = argparse.ArgumentParser(description="A script to run a simple SSVEP Experiment")
    parser.add_argument("-n", "--name", dest="name", type=str, help="Name of the device.")
    parser.add_argument("-f", "--filename", dest="filename", type=str, help="Record file name")
    args = parser.parse_args()

    n_blocks = 20           # Number of blocks
    trials_per_block = 5    # Number of total trials = n_blocks * trials_per_block
    target_pos = [(-.6, 0), (.6, 0)]  # Positions of left and right targets
    hints = [u'\u2190', u'\u2192']  # Left arrow, Right arrow
    fr_rates = [6, 8]  # 10Hz, 7.5Hz

    exp_device = explorepy.Explore()
    exp_device.connect(device_name=args.name)
    exp_device.record_data(file_name=args.filename, file_type='csv')

    def send_marker(idx):
        """Maps index to marker code and sends it to the Explore object"""
        code = idx + 10
        exp_device.set_marker(code)

    experiment = SSVEPExperiment(frame_rates=fr_rates, positions=target_pos, hints=hints, marker_callback=send_marker,
                                 trial_len=6, trials_per_block=trials_per_block, n_blocks=n_blocks,
                                 screen_refresh_rate=60
                                 )
    experiment.run()
    exp_device.stop_recording()
    exp_device.disconnect()


if __name__ == '__main__':
    main()
