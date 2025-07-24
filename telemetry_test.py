import explorepy
from explorepy.BLEClient import BLEClient
from explorepy import settings_manager

# Set the Bluetooth interface to 'mock' to use the mock server
explorepy.set_bt_interface('mock') #changes made to bt_mock_client.py to get this to work

# Create an Explore instance and connect (device_name can be any string)
explore = explorepy.Explore()
explore.connect(device_name='Explore_1C33')

explore.acquire(duration=2)

print(explore.get_firmware_version())
print(explore.get_battery_level()) #device must be acquiring data to get battery level
#print(explore.get_rssi()) #cant get rssi if using mock server
print(explore.get_rtt()) #cant get latency if using mock server

