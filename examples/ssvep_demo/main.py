# -*- coding: utf-8 -*-
"""
A script to run a simple SSVEP Experiment with Mentalab's Explore device
"""
import argparse
from explorepy.explore import Explore
from explorepy.stream_processor import TOPICS
from ssvep import SSVEPExperiment


def main():
    parser = argparse.ArgumentParser(description="A script to run a simple SSVEP Experiment")
    parser.add_argument("-n", "--name", dest="name", type=str, help="Name of the device.")
    parser.add_argument("-d", "--duration", dest="duration", type=int, help="Duration of the experiment")
    args = parser.parse_args()

    explore = Explore()
    explore.connect(device_name=args.name)

    target_pos = [(-14, -10), (-14, 10), (14, 10), (14, -10)]
    fr_rates = [5, 6, 7, 8]  # 12hz - 10hz - 8.5hz - 7.5hz
    experiment = SSVEPExperiment(frame_rates=fr_rates, positions=target_pos,
                                 signal_len=3, eeg_s_rate=250,
                                 overlap=.2, screen_refresh_rate=60)

    # subscribe the experiment buffer to the EEG data stream
    explore.stream_processor.subscribe(callback=experiment.update_buffer, topic=TOPICS.raw_ExG)

    experiment.run(args.duration)
    experiment.show_statistics()


if __name__ == '__main__':
    main()

