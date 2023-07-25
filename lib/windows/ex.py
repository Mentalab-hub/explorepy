import explorepy
from explorepy.stream_processor import TOPICS
exp_device = explorepy.Explore()

# Connect to the Explore device using device bluetooth name or mac address
exp_device.connect('Explore_8531')
exp_device.set_channels(channel_mask="00110011001100110011001100110011")
exp_device.record_data( 'file_name', do_overwrite=True, duration=15, file_type='csv')