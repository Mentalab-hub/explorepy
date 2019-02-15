=====
Usage
=====

To use explorepy in a project::

	import explorepy

Before starting a session, make sure your device is paired to your computer. The device will be shown under the following name: ExploreXXXX,
with the last 4 characters being the last 4 hex numbers of the devices MAC adress

Make sure to initialize the bluetooth connection before starting a recording session or a push to lsl using the following lines::

    explorer = explore.Explore()
    explorer.connect(device_id = 0)

If only one device is paired and on,the connection will be made automatically.
If multiple Explore-devices are paired with your computer, a short dialogue will let you
choose the device you want to connect to.

Afterwards you are free to start recording to CSV using the following line::

    explorer.record_data(device_id = 0)

or to push data to LSL using the following line::

    explorer.push2lsl(device_id = 0)


The function record_data will create 2 CSV files, one containing the ExG files and one containing environmental data (Temperature, Orientation, Battery)
In case of a disconnect (device loses connection), the program will try to reconnect automatically.
