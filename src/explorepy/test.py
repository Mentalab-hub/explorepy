from explorepy.explore import Explore
import time
myexplore = Explore()
myexplore.connect(device_name="Explore_1C3B")
myexplore.record_data(file_name='test', duration=10, file_type='csv', do_overwrite= True)
#wait until recording is done
time.sleep(12)
data_last = myexplore.get_last_recorded_data()

# now it is possible to access exg and orientation by keys: data_last['exg'] and data_last['marker'] will return separate lists
