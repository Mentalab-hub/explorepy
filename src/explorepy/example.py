from explorepy.explore import Explore
import numpy as np
import os.path
import csv
from explorepy import tools
from explorepy import command
import matplotlib.pyplot as plt
import csv

filename = "DATA025_PC"
filename2 = "test13_calibre_coef.csv"
filename3 = "/home/lilac/Desktop/27Sep/DATA027.BIN"
filename4 = "/home/lilac/Desktop/27Sep/DATA026_eeg.csv"
#filename4 = "/home/lilac/py-workspace/Explorepy/explorepy/tests/DATA025_PC_ExG.csv"

myexplore = Explore()
myexplore.connect(device_name="Explore_2FA4")
myexplore.pass_msg(msg2send=command.Command.FORMAT_MEMORY.value)
myexplore.acquire()
#myexplore.record_data(filename, do_overwrite=True)
#myexplore.push2lsl(n_chan=2)
#myexplore.calibrate(file_name= filename)
#myexplore.visualize(n_chan = 3, notch_freq=50, bp_freq=(1,100), calibre_file=filename2)  # Give number of channels (4 or 8)




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

