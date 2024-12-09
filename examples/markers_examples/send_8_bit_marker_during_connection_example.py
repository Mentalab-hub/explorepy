import explorepy
import time

# at first, set correct bt interface! Possible interface options are 'ble' and 'usb'
explorepy.set_bt_interface('ble')

exp_device = explorepy.Explore()
exp_device.connect('Explore_AAAC')
exp_device.record_data(file_name='AAAC', duration=30, file_type='csv')
n_markers = 20
interval = 2
for i in range(n_markers):
        exp_device.send_8_bit_trigger(i)
        time.sleep(2)

