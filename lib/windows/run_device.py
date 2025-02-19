import explorepy
import time
import binascii
import serial
from explorepy import exploresdk

exp_dev = explorepy.Explore()
exp_dev.connect('Explore_84DF')
exp_dev.record_data(file_name='hello', file_type='csv', duration=30, do_overwrite=True, block=True)

exp_dev.disconnect()