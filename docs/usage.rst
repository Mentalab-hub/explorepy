=====
Usage
=====

Command Line Interface
^^^^^^^^^^^^^^^^^^^^^^
**Command structure:**
``explorepy <command> [args]``

You can get help for a specific command by ``explorepy <command> -h``, for example to get help about visualize command, ``explorepy visualize -h`` will result to::

    Usage: explorepy visualize [OPTIONS]

      Visualizing signal in a browser-based dashboard

    Options:
      -a, --address TEXT        Explore device's MAC address
      -n, --name TEXT           Name of the device
      -nf, --notchfreq [50|60]  Frequency of notch filter.
      -lf, --lowfreq FLOAT      Low cutoff frequency of bandpass/highpass filter.
      -hf, --highfreq FLOAT     High cutoff frequency of bandpass/lowpass filter.
      -cf, --calib-file PATH    Calibration file name
      --pybluez                 Use pybluez as the bluetooth interface
      -h, --help                Show this message and exit.



Available Commands
""""""""""""""""""

**find-device**
Scans for nearby explore-devices. Prints out Name and MAC address of the found devices.


**acquire**::

    Connect to a device with selected name or address. Only one input is necessary.

    Options:
      -a, --address TEXT  Explore device's MAC address
      -n, --name TEXT     Name of the device
      --pybluez                 Use pybluez as the bluetooth interface
      -h, --help          Show this message and exit.



**record-data**

Connects to a device and records ExG and orientation data into 2 separate files. Note that in CSV mode there will be an extra
file for the marker events. In EDF mode, the data is actually recorded in BDF+ format (in 24-bit resolution).::

    Options:
      -a, --address TEXT        Explore device's MAC address
      -n, --name TEXT           Name of the device
      -f, --filename PATH       Name of the file.  [required]
      -ow, --overwrite          Overwrite existing file
      -d, --duration <integer>  Recording duration in seconds
      --edf                     Write in EDF file (default type)
      --csv                     Write in csv file
      --pybluez                 Use pybluez as the bluetooth interface
      -h, --help                Show this message and exit.

**push2lsl**
Streams data to Lab Streaming Layer (LSL).::

    Options:
      -a, --address TEXT        Explore device's MAC address
      -n, --name TEXT           Name of the device
      -d, --duration <integer>  Streaming duration in seconds
      --pybluez                 Use pybluez as the bluetooth interface
      -h, --help                Show this message and exit.





**bin2csv**
Takes a Binary file and converts it to 3 CSV files (ExG, orientation and marker files)::

    Options:
      -f, --filename PATH  Name of (and path to) the binary file.  [required]
      -ow, --overwrite     Overwrite existing file
      -h, --help           Show this message and exit.



.. note:: For devices with firmware version 2.1.1 and lower, explorepy v0.5.0 has to be used to convert binary files.

**bin2edf**
Takes a Binary file and converts it to 2 EDF files (ExG and orientation - markers will be written in ExG file).
The data is actually recorded in BDF+ format (in 24-bit resolution).::

    Options:
      -f, --filename PATH  Name of (and path to) the binary file.  [required]
      -ow, --overwrite     Overwrite existing file
      -h, --help           Show this message and exit.

.. note:: For devices with firmware version 2.1.1 and lower, explorepy v0.5.0 has to be used to convert binary files.

**visualize**
Visualizes real-time data in a browser-based dashboard. Currently, Chrome is the supported browser. The visualization in IE and Edge might be very slow.::

    Options:
      -a, --address TEXT        Explore device's MAC address
      -n, --name TEXT           Name of the device
      -nf, --notchfreq [50|60]  Frequency of notch filter.
      -lf, --lowfreq FLOAT      Low cutoff frequency of bandpass/highpass filter.
      -hf, --highfreq FLOAT     High cutoff frequency of bandpass/lowpass filter.
      -cf, --calib-file PATH    Calibration file name
      --pybluez                 Use pybluez as the bluetooth interface
      -h, --help                Show this message and exit.


**impedance**
Visualizes electrodes impedances in a browser-based dashboard. Currently, Chrome is the supported browser.::

    Options:
      -a, --address TEXT        Explore device's MAC address
      -n, --name TEXT           Name of the device
      -nf, --notchfreq [50|60]  Frequency of notch filter.
      --pybluez                 Use pybluez as the bluetooth interface
      -h, --help                Show this message and exit.



**calibrate_orn**
Calibrate the orientation module of the specified device. After running this module, a file containing
calibration data will be generated. Using this file, an extra computation block can be activated in the visualize
to compute the physical orientation of the device from raw sensor data.::

    Options:
      -a, --address TEXT   Explore device's MAC address
      -n, --name TEXT      Name of the device
      -f, --filename PATH  Name of the file.  [required]
      -ow, --overwrite     Overwrite existing file
      --pybluez                 Use pybluez as the bluetooth interface
      -h, --help           Show this message and exit.


**format-memory**
This command formats the memory of the specified Explore device.::

    Options:
      -a, --address TEXT  Explore device's MAC address
      -n, --name TEXT     Name of the device
      --pybluez                 Use pybluez as the bluetooth interface
      -h, --help          Show this message and exit.


**set-sampling-rate**
This command sets the sampling rate of ExG on the specified Explore device. The only acceptable values for
sampling rates are 250, 500 or 1000.::

    Options:
      -a, --address TEXT              Explore device's MAC address
      -n, --name TEXT                 Name of the device
      -sr, --sampling-rate [250|500|1000]
                                      Sampling rate of ExG channels, it can be 250
                                      or 500  [required]
      --pybluez                 Use pybluez as the bluetooth interface
      -h, --help                      Show this message and exit.


**soft-reset**
This command does a soft reset of the device. All the settings (e.g. sampling rate, channel mask)
return to the default values.::

    Options:
      -a, --address TEXT  Explore device's MAC address
      -n, --name TEXT     Name of the device
      --pybluez                 Use pybluez as the bluetooth interface
      -h, --help          Show this message and exit.


Example commands:
"""""""""""""""""
Data acquisition: ``explorepy acquire -n Explore_XXXX  # Put your device Bluetooth name``

Record data: ``explorepy record-data -n Explore_XXXX -f test_file --edf -ow``

Push data to lsl: ``explorepy push2lsl -n Explore_XXXX``

Convert a binary file to csv: ``explorepy bin2csv -f input_file.BIN``

Convert a binary file to EDF and overwrite if files exist already: ``explorepy bin2edf -f input_file.BIN -ow``

Visualize in real-time: ``explorepy visualize -n Explore_XXXX``

Impedance measurement: ``explorepy impedance -n Explore_XXXX``

Format the memory: ``explorepy format-memory -n Explore_XXXX``

Set the sampling rate: ``explorepy set-sampling-rate -n Explore_XXXX -sr 500``

To see the full list of commands ``explorepy -h``.

Python project
^^^^^^^^^^^^^^
To use explorepy in a python project::

	import explorepy


Initialization
^^^^^^^^^^^^^^
Before starting a session, make sure your device is paired to your computer. The device will be shown under the following name: Explore_XXXX,
with the last 4 characters being the last 4 hex numbers of the devices MAC adress

**Make sure to initialize the Bluetooth connection before streaming using the following lines**::

    explore = explorepy.Explore()
    explore.connect(device_name="Explore_XXXX") # Put your device Bluetooth name

Alternatively you can use the device's MAC address::

    explore.connect(mac_address="XX:XX:XX:XX:XX:XX")

If the device is not found it will raise an error.

By defalut, Explorepy uses its own SDK for bluetooth interface. However, you can use Pybluez as the BT interface. To change the
BT interface, use the following code. ::

    explorepy.set_bt_interface('pybluez')

To return it to the SDK: ::

    explorepy.set_bt_interface('sdk')


Streaming
^^^^^^^^^
After connecting to the device you are able to stream data and print the data in the console.::

    explore.acquire()


Recording
^^^^^^^^^
You can record data in realtime to EDF (BDF+) or CSV files::

    explore.record_data(file_name='test', duration=120, file_type='csv')

This will record data in three separate files "test_ExG.csv", "test_ORN.csv" and "test_marker.csv" which contain ExG, orientation data (accelerometer, gyroscope, magnetometer) and event markers respectively. The duration of the recording can be specified (in seconds).
If you want to overwrite already existing files, change the line above::

    explore.record_data(file_name='test', do_overwrite=True, file_type='csv', duration=120)


Visualization
^^^^^^^^^^^^^
It is possible to visualize real-time signal in a browser-based dashboard by the following code. Currently, Chrome is the supported browser. The visualization in IE and Edge might be very slow.::


    explore.visualize(bp_freq=(1, 30), notch_freq=50)

Where `bp_freq` and `notch_freq` determine cut-off frequencies of bandpass/lowpass/highpass filter and frequency of notch filter (either 50 or 60) respectively.


In the dashboard, you can set signal mode to EEG or ECG. EEG mode provides the spectral analysis plot of the signal. In ECG mode, the heart beats are detected and heart rate is estimated from RR-intervals.

EEG:

.. image:: /images/Dashboard_EEG.jpg
  :width: 800
  :alt: EEG Dashboard

ECG with heart beat detection:

.. image:: /images/Dashboard_ECG.jpg
  :width: 800
  :alt: ECG Dashboard


Impedance measurement
^^^^^^^^^^^^^^^^^^^^^
To measure electrodes impedances::


    explore.impedance(notch_freq=50)


.. image:: /images/Dashboard_imp.jpg
  :width: 800
  :alt: Impedance Dashboard

.. note::  The accuracy of measured impedances are subject to environmental conditions such as noise and temperature.

Labstreaminglayer (lsl)
^^^^^^^^^^^^^^^^^^^^^^^
You can push data directly to LSL using the following line::

    explore.push2lsl()


After that you can stream data from other software such as OpenVibe or other programming languages such as MATLAB, Java, C++ and so on. (See `labstreaminglayer <https://github.com/sccn/labstreaminglayer>`_, `OpenVibe <http://openvibe.inria.fr/how-to-use-labstreaminglayer-in-openvibe/>`_ documentations for details).
This function creates three LSL streams for ExG, Orientation and markers.
In case of a disconnect (device loses connection), the program will try to reconnect automatically.


Converter
^^^^^^^^^
It is also possible to extract BIN files from the device via USB. To convert these to CSV, you can use the function bin2csv, which takes your desired BIN file
and converts it to 2 CSV files (one for orientation, the other one for ExG data). Bluetooth connection is not necessary for conversion. ::

    explore.convert_bin(bin_file='Data001.BIN', file_type='csv', do_overwrite=False)

If you want to overwrite existing files, use::

    bin2csv(bin_file, do_overwrite=True)


Event markers
^^^^^^^^^^^^^
In addition to the marker event generated by pressing the button on Explore device, you can set markers in your code using `explorepy.Explore.set_marker` function. However, this function must be called from a different thread than the parsing thread.
Please not that marker codes between 0 and 7 are reserved for hardware related markers. You can use any other (integer) code for your marker from 8 to 65535.
To see an example usage of this function look at `this script <https://github.com/Mentalab-hub/explorepy/tree/master/examples/marker_example.py>`_
