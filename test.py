import numpy as np
from explorepy import explore
file_name = "demofile"

explorer = explore.Explore()

explorer.connect(device_id = 0)
explorer.acquire(device_id = 0)
