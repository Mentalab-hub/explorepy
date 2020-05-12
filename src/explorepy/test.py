from explorepy import Explore

device_name = "Explore_143A"

explore = Explore()
explore.connect(device_name=device_name)
#explore.set_channels(7)
explore.visualize(bp_freq=(1, 40), notch_freq=50)