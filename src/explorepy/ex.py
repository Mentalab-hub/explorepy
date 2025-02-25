import time

from pylsl import local_clock

import explorepy


explorepy.set_bt_interface('usb')
# Before starting the USB based trigger, make sure the device is connected to the USB port
exp_device = explorepy.Explore()
exp_device.connect('Explore_AAAF')
exp_device.set_sampling_rate(8000)
# measure time
start = local_clock()
exp_device.record_data(file_type='csv', file_name='test', do_overwrite=True, block=True, duration=7200)
print('Duration: {}'.format(local_clock() - start))
