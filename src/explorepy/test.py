import numpy as np
from explore import Explore
f = open("demofile.txt", "w")

explorer = Explore()

explorer.connect(device_id = 0)
explorer.recordData("testfile", device_id=0)
