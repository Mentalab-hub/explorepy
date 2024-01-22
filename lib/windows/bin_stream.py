import sys

import explorepy
from explorepy.stream_processor import TOPICS
import time


def my_exg_function(packet):
    """A function that receives ExG packets and does some operations on the data"""
    t_vector, exg_data = packet.get_data()
    print("Received an ExG packet with data shape: ", exg_data.shape)
    #############
    # YOUR CODE #
    #############




explorepy.Explore().convert_bin(bin_file='test.BIN', file_type='csv', do_overwrite=True, custom_callback=my_exg_function)

# Connect to the Explore device using device bluetooth name or mac address


try:
    while True:
        time.sleep(.1)
except KeyboardInterrupt:
    sys.exit(0)
