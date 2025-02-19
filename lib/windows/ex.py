import explorepy
import time
import binascii
import serial
from explorepy import exploresdk

#exploresdk.BTSerialPortBinding.Create('00:13:43:A1:84:DF', 5).Connect()
myusb = serial.Serial('/dev/tty.Explore_84DF', timeout=5)

#myusb.reset_input_buffer()

count = 0
while count < 10:
        print(binascii.hexlify(myusb.read(4)))
        count+= 1
