# import numpy as np
# import time
import explorepy
# from explorepy import settings_manager
# e = explorepy.Explore()
# name = "Explore_8443"
# e.connect(name)
# e.format_memory()
# # # #s = settings_manager.SettingsManager(name)
# # # #s.set_adc_mask([1] + [0] * 31)
# # #e.push2lsl()
# # # #e.set_channels("1010")
# e.set_sampling_rate(500)
# time.sleep(10)
# # time.sleep(20)
# e.set_sampling_rate(250)
# # #e.set_sampling_rate(250)
# e.record_data(do_overwrite=True, duration=60, file_type="csv", file_name="test1")
# # # import time
# # # import os
# # # #time.sleep(15)
# # # #e.acquire(30)
# # #e.convert_bin(file_type='csv', bin_file='DATA000.BIN')

explorepy.tools.compare_recover_from_bin('test1', 'DATA000_36.0')
