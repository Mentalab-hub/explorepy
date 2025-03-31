import explorepy
import time

# for independent markers, we only need to instantiate Explore class!
exp_device = explorepy.Explore()

n_markers = 20
interval = 2
for i in range(n_markers):
        exp_device.send_8_bit_trigger(i)
        time.sleep(interval)

