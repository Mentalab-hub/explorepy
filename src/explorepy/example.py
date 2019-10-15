from explorepy.explore import Explore
import numpy as np
import os.path
import csv
from explorepy import tools
import matplotlib.pyplot as plt
import csv
from threading import Thread, Timer
from explorepy.command import Command
filename = "DATA025_PC"
filename2 = "test13_calibre_coef.csv"
filename3 = "/home/lilac/Desktop/27Sep/DATA027.BIN"
filename4 = "/home/lilac/Desktop/27Sep/DATA026_eeg.csv"
#filename4 = "/home/lilac/py-workspace/Explorepy/explorepy/tests/DATA025_PC_ExG.csv"

myexplore = Explore()
myexplore.connect(device_name= "Explore_E02B", device_addr="00:13:43:86:E0:2B")
#00:13:43:86:E0:2B,00:13:43:6E:2F:AA,00:13:43:86:E0:28,00:13:43:86:E0:29,00:13:43:2B:33:0B
#myexplore.pass_msg(msg2send=Command.ENV_DISABLE.value)
#myexplore.pass_msg(msg2send=Command.MOTION_DISABLE.value)
#myexplore.acquire()
#myexplore.record_data(filename, do_overwrite=True)
#myexplore.push2lsl(n_chan=8)
#myexplore.calibrate(file_name= filename)
myexplore.visualize(n_chan = 8, notch_freq=50, bp_freq=(1,30))  # Give number of channels (4 or 8)



#tools.bin2csv(bin_file= filename3,do_overwrite=True)
"""
from collections import deque
with open(filename4, 'r') as f:
    q = deque(f, 1)  # replace 2 with n (lines read at the end)

print(q)

def sum1forline(filename):
    with open(filename) as f:
        return sum(1 for line in f)

print(sum1forline(filename4))


with open(filename4, 'r') as f:
    data = list(csv.reader(f, delimiter=','))
    data = np.array(data[1:], dtype=np.float)
    data = data[:, 0]
print(data.shape)
plt.plot(np.diff(data))
plt.show()
"""

