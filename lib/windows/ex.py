import explorepy
import time


# Before starting the USB based trigger, make sure the device is connected to the USB port
exp_device = explorepy.Explore()
exp_device.connect('Explore_AAAM')

exp_device.record_data( 'file_name', do_overwrite=True, duration=60, file_type='csv')
n_markers = 20
interval = 2
for i in range(n_markers):
        exp_device.send_8_bit_trigger(i)
        time.sleep(2)
