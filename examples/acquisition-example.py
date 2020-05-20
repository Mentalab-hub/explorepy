"""An example code for data acquisition from Explore device"""

import time
import explorepy
from explorepy.stream_processor import TOPICS
import argparse


def my_exg_function(packet):
    """A function that receives ExG packets and does some operations on the data"""
    t_vector, exg_data = packet.get_data()
    print("Received an ExG packet with data shape: ", exg_data.shape)
    #############
    # YOUR CODE #
    #############


def my_orn_function(packet):
    """A function that receives orientation packets and does some operations on the data"""
    timestamp, orn_data = packet.get_data()
    print("Received an orientation packet: ", orn_data)
    #############
    # YOUR CODE #
    #############


def main():
    parser = argparse.ArgumentParser(description="Example code for data acquisition")
    parser.add_argument("-n", "--name", dest="name", type=str, help="Name of the device.")
    args = parser.parse_args()

    # Create an Explore object
    exp_device = explorepy.Explore()

    # Connect to the Explore device using device bluetooth name or mac address
    exp_device.connect(device_name=args.name)

    # Subscribe your function to the stream publisher
    exp_device.stream_processor.subscribe(callback=my_exg_function, topic=TOPICS.raw_ExG)
    exp_device.stream_processor.subscribe(callback=my_orn_function, topic=TOPICS.raw_orn)
    try:
        while True:
            time.sleep(.5)
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
