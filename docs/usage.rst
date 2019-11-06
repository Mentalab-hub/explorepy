=====
Usage
=====

Command Line Interface
^^^^^^^^^^^^^^^^^^^^^^
**Command structure:**
``explorepy <command> [args]``


Available Commands
""""""""""""""""""

**find_device**
Scans for nearby explore-devices. Prints out Name and MAC address of the found devices


**acquire**

Connects to device, needs either MAC or Name of the desired device as input
* ``-a`` or ``--address``    Device MAC address (Form XX:XX:XX:XX:XX:XX).
* ``-n`` or ``--name``       Device name (e.g. "Explore_12AB").



**record_data**
Connects to a device and records Orientation and Body data live to 2 separate CSV files

* ``-a`` or ``--address``    Device MAC address (Form XX:XX:XX:XX:XX:XX).
* ``-n`` or ``--name``       Device name (e.g. Explore_12AB).
* ``-f`` or ``--filename``   The name of the new CSV Files.
* ``-o`` or ``--overwrite``  Overwrite already existing files with the same name.
* ``-d`` or ``--duration``   Recording duration in seconds



**push2lsl**
Streams Data to Lab stream layer. Inputs: Name or Address and Channel number

* ``-a`` or ``--address``    Device MAC address (Form XX:XX:XX:XX:XX:XX).
* ``-n`` or ``--name``       Device name (e.g. Explore_12AB).
* ``-c`` or ``--channels``   Number of channels. This is necessary for push2lsl



**bin2csv**
Takes a Binary file and converts it to 2 CSV files (orientation and Body)

* ``-i`` or ``--inputfile``  Name of the input file
* ``-o`` or ``--overwrite``  Overwrite already existing files with the same name.



**visualize**
Visualizes real-time data in a browser-based dashboard. Currently, Chrome is the supported browser. The visualization in IE and Edge might be very slow.

* ``-a`` or ``--address``    Device MAC address (Form XX:XX:XX:XX:XX:XX).
* ``-n`` or ``--name``       Device name (e.g. Explore_12AB).
* ``-c`` or ``--channels``   Number of channels.
* ``-nf`` or ``--notchfreq`` Frequency of applied notch filter (By default, no notch filter is applied)

**pass_msg**
sends a byte array to the Bluetooth device. the byte array corresponding to specific commands has been declared in command class definition. when the message field is empty, default message which is host timestamp will be sent to the device. When a message is received by Explore, it will send an acknowledge message and also another status message after processing if received command message.

* ``-a`` or ``--address``    Device MAC address (Form XX:XX:XX:XX:XX:XX).
* ``-n`` or ``--name``       Device name (e.g. Explore_12AB).
* ``-m`` or ``--message``    The message to be sent, the input format is byte array. If not used, host timestamp will be sent.

**format_memory**
This command formats the memory of the specified Explore device.

* ``-a`` or ``--address``    Device MAC address (Form XX:XX:XX:XX:XX:XX).
* ``-n`` or ``--name``       Device name (e.g. Explore_12AB).

**set_sampling_rate**
This command sets the sampling rate of ExG input on the specified Explore device. The only acceptable values for sampling rates are 250, 500 or 1000.

* ``-a`` or ``--address``        Device MAC address (Form XX:XX:XX:XX:XX:XX).
* ``-n`` or ``--name``           Device name (e.g. Explore_12AB).
* ``-r`` or ``--sampling_rate``  Sampling rate of ExG channels, it can be 250, 500 or 1000.


Example commands:
"""""""""""""""""
Data acquisition: ``explorepy acquire -n Explore_XXXX  #Put your device Bluetooth name``

Record data: ``explorepy record_data -n Explore_XXXX -f file_name``

Push data to lsl: ``explorepy push2lsl -n Explore_XXXX -c 4 #-c number of channels (4 or 8)``

Convert a binary file to csv: ``explorepy bin2csv -i input_file``

Visualize in real-time: ``explorepy visualize -n Explore_XXXX -c 4``

send a message: ``explorepy pass_msg -n Explore_XXXX``

format the memory: ``explorepy format_memory -n Explore_XXXX``

set the sampling rate: ``explorepy set_sampling_rate -n Explore_XXXX -r 500``

To see the full list of commands ``explorepy -h``.

Python project
^^^^^^^^^^^^^^
To use explorepy in a python project::

	import explorepy


Initialization
^^^^^^^^^^^^^^
Before starting a session, make sure your device is paired to your computer. The device will be shown under the following name: Explore_XXXX,
with the last 4 characters being the last 4 hex numbers of the devices MAC adress

Make sure to initialize the bluetooth connection before starting a recording session or a push to lsl using the following lines::

    explorer = explorepy.Explore()
    explorer.connect(device_name="Explore_XXXX") #Put your device Bluetooth name

Alternatively you can use the device's MAC address::

    explorer.connect(device_addr="XX:XX:XX:XX:XX:XX")

If the device is not found it will raise an error.

Streaming
^^^^^^^^^
After connecting to the device you are able to stream data and print the data in the console.::

    explorer.acquire()


Recording
^^^^^^^^^
Afterwards you are free to start recording to CSV using the following line::

    explorer.record_data(file_name='test', duration=120)

This will record data in three separate files "test_ExG.csv", "test_ORN.csv" and "test_marker.csv" which contain ExG, orientation data (accelerometer, gyroscope, magnetometer) and event markers respectively. The duration of the recording can be specified (in seconds).
The program will usually stop if files with the same name are detected. If you want to overwrite already existing files, change the line above::

    explorer.record_data(file_name='test', do_overwrite=True, duration=120)


Visualization
^^^^^^^^^^^^^
It is possible to visualize real-time signal in a browser-based dashboard by the following code. Currently, Chrome is the supported browser. The visualization in IE and Edge might be very slow.::


    explorer.visualize(n_chan=4, bp_freq=(1, 30), notch_freq=50)

Where `n_chan`, `bp_freq` and `notch_freq` determine the number of channels, cut-off frequencies of bandpass filter and frequency of notch filter (either 50 or 60) respectively.


In the dashboard, you can set signal mode to EEG or ECG. EEG mode provides the spectral analysis plot of the signal. In ECG mode, the heart beats are detected and heart rate is estimated from RR-intervals.

EEG:

.. image:: /images/Dashboard_EEG.jpg
  :width: 800
  :alt: EEG Dashboard

ECG with heart beat detection:

.. image:: /images/Dashboard_ECG.jpg
  :width: 800
  :alt: ECG Dashboard

Labstreaminglayer (lsl)
^^^^^^^^^^^^^^^^^^^^^^^
You can push data directly to LSL using the following line::

    explorer.push2lsl(n_chan=4)


It is important that you state the number of channels your device has.
After that you can stream data from other software such as OpenVibe or other programming languages such as MATLAB, Java, C++ and so on. (See `labstreaminglayer <https://github.com/sccn/labstreaminglayer>`_, `OpenVibe <http://openvibe.inria.fr/how-to-use-labstreaminglayer-in-openvibe/>`_ documentations for details).
This function creates three LSL streams for ExG, Orientation and markers.
In case of a disconnect (device loses connection), the program will try to reconnect automatically.


Converter
^^^^^^^^^
It is also possible to extract BIN files from the device via USB. To convert these to CSV, you can use the function bin2csv, which takes your desired BIN file
and converts it to 2 CSV files (one for orientation, the other one for ExG data). Bluetooth connection is not necessary for conversion. ::

    from explorepy.tools import bin2csv
    bin2csv(bin_file)

If you want to overwrite existing files, use::

    bin2csv(bin_file, do_overwrite=True)


Passmessage
^^^^^^^^^^^^
Once the Explore device is connected. you can send message to the device these messages can be commands or host timestamp::
the current set of commands are::

    FORMAT_MEMORY   #formats memory and resume the run
    SPS_250         #sets sampling rate to 250 samples per second
    SPS_500         #sets sampling rate to 500 samples per second
    SPS_1000        #sets sampling rate to 1000 samples per second

and this is how to use it::

    from explorepy import command
    myexplore.connect(device_name="Explore_XXXX")
    explorer.pass_msg(msg2send=Command.FORMAT_MEMORY.value)


If you want to send a host timestamp, use::

    explorer.pass_msg()



