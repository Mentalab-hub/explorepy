import time

import explorepy
from explorepy.stream_processor import TOPICS

def my_exg_function(packet):
    """A function that receives ExG packets and does some operations on the data"""
    exg_data = packet.get_impedances()
    print("Received an ExG packet with data shape: ", exg_data)
    #############
    # YOUR CODE #
    #############

explorepy.set_bt_interface('ble')
i = 0

exp_device = explorepy.Explore()

# Connect to the Explore device using device bluetooth name or mac address
exp_device.connect(device_name='Explore_AAAP')
exp_device.stream_processor.imp_initialize(notch_freq=50)
exp_device.stream_processor.subscribe(callback=my_exg_function, topic=TOPICS.imp)

try:
    while True:
        time.sleep(.5)
except KeyboardInterrupt:
    print('stopped!')




