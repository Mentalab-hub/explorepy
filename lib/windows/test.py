import numpy as np

import explorepy
from explorepy import settings_manager
e = explorepy.Explore()
name = "Explore_844A"
e.connect(name)
#s = settings_manager.SettingsManager(name)
#s.set_adc_mask([1] + [0] * 31)
#e.push2lsl()
#e.set_channels("1010")
e.record_data(do_overwrite=True, duration=10, file_type="csv", file_name="test1")
import time
import os
#time.sleep(15)
#e.acquire(30)


