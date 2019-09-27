from explorepy.explore import Explore
import numpy as np
import os.path
import csv
from explorepy import tools
from explorepy import command

filename = "test13"
filename2 = "test13_calibre_coef.csv"
filename3 = "/home/lilac/Desktop/DATA130.BIN"
filename4 = "/home/lilac/Desktop/DATA022_eeg.csv"

myexplore = Explore()
myexplore.connect(device_name="Explore_330B")
myexplore.pass_msg(msg2send=command.Command.FORMAT_MEMORY.value)
myexplore.acquire()
#myexplore.record_data(filename, do_overwrite=False)
#myexplore.push2lsl(n_chan=2)
#myexplore.calibrate(file_name= filename)
#myexplore.visualize(n_chan = 3, notch_freq=50, bp_freq=(1,100), calibre_file=filename2)  # Give number of channels (4 or 8)

'''
tools.bin2csv(filename3)
from collections import deque
with open(filename4, 'r') as f:
    q = deque(f, 1)  # replace 2 with n (lines read at the end)
print(q)
'''
