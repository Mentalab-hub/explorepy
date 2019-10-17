"""An example code for data acquisition from Explore device"""

import explorepy
import argparse


def streaming(parser):
    """A generator function for parsed packets"""
    while True:
        yield parser.parse_packet()


def main():
    parser = argparse.ArgumentParser(description="Example code for data acquisition")
    parser.add_argument("-n", "--name", dest="name", type=str, help="Name of the device.")
    args = parser.parse_args()

    # Create an Explore object
    exp_device = explorepy.Explore()

    # Connect to the Explore device using device bluetooth name or mac address
    exp_device.connect(device_name=args.name)

    # Instantiate a generator object
    gen_func = streaming(exp_device.parser)

    for packet in gen_func:
        print(packet)
        # YOUR CODE


if __name__ == "__main__":
    main()
