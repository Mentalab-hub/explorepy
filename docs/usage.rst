=====
Usage
=====

Command Line Interface
^^^^^^^^^^^^^^^^^^^^^^
**Command structure:**
``explorepy <command> [args]``

Get help for a specific command using
::
    explorepy <command> -h
For example to get help about the visualize command, run: ``explorepy push2lsl -h``
::

    Usage: explorepy push2lsl [OPTIONS]

        Push data to lsl

    Options:
        -a, --address TEXT        Explore device's MAC address
        -n, --name TEXT           Name of the device
        -d, --duration <integer>  Streaming duration in seconds
        -h, --help                Show this message and exit.


Available Commands
""""""""""""""""""

acquire
%%%%
Connects to a device with selected name or address. Only one input is necessary.
::

    Options:
      -a, --address TEXT  Explore device's MAC address
      -n, --name TEXT     Name of the device
      -h, --help          Show this message and exit.

Example:
::
    explorepy acquire -n Explore_XXXX  # Put your device Bluetooth name

record-data
%%%%

Connects to a device and records ExG and orientation data into two separate files. In EDF mode, the data is actually recorded in BDF+ format (in 24-bit resolution). Note that in CSV mode there will be two extra files. One for the marker events, and one for the metadata.
::

    Options:
      -a, --address TEXT              Explore device's MAC address
      -n, --name TEXT                 Name of the device
      -f, --filename PATH             Name of the file.  [required]
      -ow, --overwrite                Overwrite existing file
      -d, --duration <integer>        Recording duration in seconds
      --edf                           Write in EDF file
      --csv                           Write in csv file (default type)
      --imp-mode                      Enable impedance mode with live monitoring
      -h, --help                      Show this message and exit.


.. note:: If you change your device's sampling rate or channel mask during recording, ``explorepy`` will create a new CSV file for ExG data with the given file name plus the time the setting changed.

.. note:: You can Explore desktop to generate .set files for EEGLAB export

.. note:: Because environmental factors, like temperature, can affect your device's sampling rate, we recommend computing the sampling rate of recorded data. If you find a deviation between the recorded sampling rate and ``explorepy``'s sampling rate, resample your signal to correct for drifts. The timestamps in the CSV/EDF file can be used to compute the resampling factor.

           If you are setting markers, use CSV. Alternatively, push data to LSL and record with `LabRecorder <https://github.com/labstreaminglayer/App-labrecorder/tree/master>`_. Avoid EDF files here, as they cannot guarantee precise timing.


Example:
::
    explorepy record-data -n Explore_XXXX -f test_file --edf -ow

push2lsl
%%%%

Streams data to Lab Streaming Layer (LSL).
::

    Options:
      -a, --address TEXT        Explore device's MAC address
      -n, --name TEXT           Name of the device
      -d, --duration <integer>  Streaming duration in seconds
      -h, --help                Show this message and exit.

Example:
::
    explorepy push2lsl -n Explore_XXXX

bin2csv
%%%%

Takes a binary file and converts it to four CSV files (ExG, orientation, marker files and metadata).
::

    Options:
      -f, --filename PATH  Name of (and path to) the binary file.  [required]
      -ow, --overwrite     Overwrite existing file
      -h, --help           Show this message and exit.

.. note:: If you change your device's sampling rate or channel mask during recording, ``explorepy`` will create a new CSV file for ExG data with the given file name plus the time the setting changed.

Example:
::
    explorepy bin2csv -f input_file.BIN

bin2edf
%%%%

Takes a binary file and converts it to two EDF files (ExG and orientation - markers will be written in ExG file). The data is actually recorded in BDF+ format (in 24-bit resolution).
::

    Options:
      -f, --filename PATH  Name of (and path to) the binary file.  [required]
      -ow, --overwrite     Overwrite existing file
      -h, --help           Show this message and exit.

.. note:: To load EDF files, you can use `pyedflib <https://github.com/holgern/pyedflib>`_ or `mne <https://github.com/mne-tools/mne-python>`_ (for mne, you may need to change the file extension to ``bdf`` manually) in Python.

          EEGLAB's BIOSIG plugin has problems with some EDF files (see this `issue <https://github.com/sccn/eeglab/issues/103>`_). To resolve this, download a precompiled MATLAB file (mexSLOAD.mex) from BIOSIG `here <https://pub.ist.ac.at/~schloegl/src/mexbiosig/>`_. Documentation is `here <http://biosig.sourceforge.net/help/biosig/t200/sload.html>`_.

.. note:: Because environmental factors, like temperature, can affect your device's sampling rate, we recommend computing the sampling rate of recorded data. If you find a deviation between the recorded sampling rate and ``explorepy``'s sampling rate, resample your signal to correct for drifts. The timestamps in the CSV/EDF file can be used to compute the resampling factor.

           If you are setting markers, use CSV. Alternatively, push data to LSL and record with `LabRecorder <https://github.com/labstreaminglayer/App-labrecorder/tree/master>`_. Avoid EDF files here, as they cannot guarantee precise timing.

Example (overwrite):
::
    explorepy bin2edf -f input_file.BIN -ow


format-memory
%%%%

Formats device memory.
::

    Options:
      -a, --address TEXT  Explore device's MAC address
      -n, --name TEXT     Name of the device
      -h, --help          Show this message and exit.

Example:
::
    explorepy format-memory -n Explore_XXXX

set-sampling-rate
%%%%

Sets a device's ExG sampling rate. Acceptable values: 250, 500 or 1000 (beta). The default sampling rate is 250 Hz.
::

    Options:
      -a, --address TEXT              Explore device's MAC address
      -n, --name TEXT                 Name of the device
      -sr, --sampling-rate [250 | 500 | 1000]
                                      Sampling rate of ExG channels, it can be 250,
                                      500 or 1000 [required]
      -h, --help                      Show this message and exit.

Example:
::
    explorepy set-sampling-rate -n Explore_XXXX -sr 500


soft-reset
%%%%

Soft resets a device. All settings (e.g. sampling rate, channel mask) return to default.
::

    Options:
      -a, --address TEXT  Explore device's MAC address
      -n, --name TEXT     Name of the device
      -h, --help          Show this message and exit.


All commands:
"""""""""""""""""
To see the full list of commands
::
    explorepy -h


Creating a Python project
^^^^^^^^^^^^^^
To use ``explorepy`` in a Python project:
::
	import explorepy

.. note:: Because ``explorepy`` uses multithreading, running Python scripts in some consoles, such as Ipython's or Spyder's, can cause strange behaviours.

.. note:: For an example project using ``explorepy``, see this `folder on GitHub <https://github.com/Mentalab-hub/explorepy/tree/master/examples>`_.


Initialization
""""""""""""""
Before starting a session, ensure your device is paired to your computer. The device will display under the following: ``Explore_XXXX``.

**Be sure to initialize the Bluetooth connection before streaming:**
::

    explore = explorepy.Explore()
    explore.connect(device_name="Explore_XXXX") # Put your device Bluetooth name

Alternatively, use your device's MAC address.
::
    explore.connect(mac_address="XX:XX:XX:XX:XX:XX")

If the device cannot be found, you will receive an error.

Streaming
"""""""""

After connecting to the device, you will be able to stream and print data to the console.
::
    explore.acquire()

Recording
"""""""""

You can record data in realtime to EDF (BDF+) or CSV files using:
::
    explore.record_data(file_name='test', duration=120, file_type='csv')

To also record impedance data, use:
::
    explore.record_data(file_name='test', duration=120, file_type='csv', imp_mode=True)

This will record data in three separate files: "``test_ExG.csv``", "``test_ORN.csv``" and "``test_marker.csv``", which contain ExG data, orientation data (accelerometer, gyroscope, magnetometer) and event markers respectively. Add command arguments to overwrite files and set the duration of the recording (in seconds).
::
    explore.record_data(file_name='test', do_overwrite=True, file_type='csv', duration=120)

.. note:: To load EDF files, you can use `pyedflib <https://github.com/holgern/pyedflib>`_ or `mne <https://github.com/mne-tools/mne-python>`_ (for mne, you may need to change the file extension to ``bdf`` manually) in Python.

          EEGLAB's BIOSIG plugin has problems with some EDF files (see this `issue <https://github.com/sccn/eeglab/issues/103>`_). To resolve this, download a precompiled MATLAB file (mexSLOAD.mex) from BIOSIG `here <https://pub.ist.ac.at/~schloegl/src/mexbiosig/>`_. Documentation is `here <http://biosig.sourceforge.net/help/biosig/t200/sload.html>`_.

.. note:: Because environmental factors, like temperature, can affect your device's sampling rate, we recommend computing the sampling rate of recorded data. If you find a deviation between the recorded sampling rate and ``explorepy``'s sampling rate, resample your signal to correct for drifts. The timestamps in the CSV/EDF file can be used to compute the resampling factor.

           If you are setting markers, use CSV. Alternatively, push data to LSL and record with `LabRecorder <https://github.com/labstreaminglayer/App-labrecorder/tree/master>`_. Avoid EDF, as it cannot guarantee precise timing.


Lab Streaming Layer (lsl)
"""""""""""""""""""""""

You can push data to LSL using:
::
    explore.push2lsl()

LSL allows you to stream data from your Explore device and third-parties, like OpenVibe or MATLAB, simultaneously. (See the `LabStreaming Layer docs <https://github.com/sccn/labstreaminglayer>`_ and `OpenVibe docs <http://openvibe.inria.fr/how-to-use-labstreaminglayer-in-openvibe/>`_ for more).

``push2lsl`` creates three LSL streams; one for each of ExG data, orientation data and marker events. If your device loses connection, ``explorepy`` will try to reconnect automatically.

Converter
"""""""""

It is possible to extract BIN files from a device via USB. To convert these binary files to CSV, use ``bin2csv``. This function will create two CSV files (one for orientation, the other one for ExG data). A Bluetooth connection is not needed for this.
::
    explore.convert_bin(bin_file='DATA001.BIN', file_type='csv', do_overwrite=False)


.. note:: If you change your device's sampling rate or channel mask during recording, ``explorepy`` will create a new CSV file for ExG data with the given file name plus the time the setting changed.

.. note:: Because environmental factors, like temperature, can affect your device's sampling rate, we recommend computing the sampling rate of recorded data. If you find a deviation between the recorded sampling rate and ``explorepy``'s sampling rate, resample your signal to correct for drifts. The timestamps in the CSV/EDF file can be used to compute the resampling factor.

           If you are setting markers, use CSV. Alternatively, push data to LSL and record with `LabRecorder <https://github.com/labstreaminglayer/App-labrecorder/tree/master>`_. Avoid EDF, as it cannot guarantee precise timing.

Event markers
"""""""""""""
Event markers can be used to time synch data. The following table describes all types of event markers
available for Explore device.

.. list-table:: Event markers table
    :widths: 25 25 50
    :header-rows: 1

    * - Type
      - Code range
      - Label in recordings
    * - Push button
      - 0-7
      - pb_<CODE>
    * - Software marker
      - Any string between 1 and 7 characters
      - sw_<marker_text>
    * - Trigger-in
      - Only 0
      - in_0

In order to set markers programmatically, use:
::
    explore.set_marker(code='marker_1')

A simple example of software markers used in a script can be found `here <https://github.com/Mentalab-hub/explorepy/tree/master/examples/marker_example.py>`_.

Device configuration
""""""""""""""""""""

You can programmatically change a device's settings.

To change a device's sampling rate:
::
    explore.set_sampling_rate(sampling_rate=500)


To format a device's memory:
::
    explore.format_memory()


To reset a device's settings:
::
    explore.reset_soft()


Sensor data acquisition in real-time
""""""""""""""""""""""""""""""""""""
::

    """An example code for data acquisition from Explore device

    import time
    import explorepy
    from explorepy.stream_processor import TOPICS
    import argparse


    def my_exg_function(packet):
        """A function that receives ExG packets and does some operations on the data"""
        t_vector, exg_data = packet.get_data()
        print("Received an ExG packet with data shape: ", exg_data.shape)
        #############
        # YOUR CODE #
        #############


    def my_env_function(packet):
        """A function that receives env packets(temperature, light, battery) and does some operations on the data"""
        print("Received an environment packet: ", packet)
        #############
        # YOUR CODE #
        #############


    def my_orn_function(packet):
        """A function that receives orientation packets and does some operations on the data"""
        timestamp, orn_data = packet.get_data()
        print("Received an orientation packet: ", orn_data)
        #############
        # YOUR CODE #
        #############


    def main():
        parser = argparse.ArgumentParser(description="Example code for data acquisition")
        parser.add_argument("-n", "--name", dest="name", type=str, help="Name of the device.")
        args = parser.parse_args()

        # Create an Explore object
        exp_device = explorepy.Explore()

        # Connect to the Explore device using device bluetooth name or mac address
        exp_device.connect(device_name=args.name)

        # Subscribe your function to the stream publisher
        exp_device.stream_processor.subscribe(callback=my_exg_function, topic=TOPICS.raw_ExG)
        exp_device.stream_processor.subscribe(callback=my_orn_function, topic=TOPICS.raw_orn)
        exp_device.stream_processor.subscribe(callback=my_env_function, topic=TOPICS.env)
        try:
            while True:
                time.sleep(.5)
        except KeyboardInterrupt:
            return


    if __name__ == "__main__":
        main()
    """

Impedance data acquisition in real-time
"""""""""""""""""""""""""""""""""""""""
::

    import argparse
    import csv
    import time

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy import signal

    import explorepy
    from explorepy.stream_processor import TOPICS


    # ----------------------------- Argument Parsing ----------------------------- #
    parser = argparse.ArgumentParser(description="Acquire and filter impedance-mode ExG data from an Explore device.")
    parser.add_argument("--device-name", required=True, help="Name of the Explore device (e.g., Explore_AAXX)")
    args = parser.parse_args()


    # ----------------------------- Configuration ----------------------------- #
    FS = 250  # Sampling rate in Hz
    CHANNEL_LABELS = [f"ch{i}" for i in range(1, 33)]
    OUTPUT_FILENAME = "exg_data_imp_mode.csv"
    RECORD_SECONDS = 40


    # ----------------------------- CSV Setup ----------------------------- #
    csv_file = open(OUTPUT_FILENAME, 'w', newline='\n')
    csv_writer = csv.writer(csv_file, delimiter=",")
    csv_writer.writerow(['Timestamp'] + CHANNEL_LABELS[:8])  # Log only first 8 channels


    # ----------------------------- Packet Handlers ----------------------------- #
    def handle_exg_packet(packet):
        """Callback to handle incoming ExG data packets."""
        timestamps, signals = packet.get_data(FS)
        data = np.concatenate((np.array(timestamps)[:, np.newaxis].T, np.array(signals)), axis=0)
        np.savetxt(csv_file, np.round(data.T, 4), fmt='%4f', delimiter=',')


    def handle_impedance_packet(packet):
        """Callback to handle incoming impedance packets."""
        impedance_values = packet.get_impedances()
        print("Impedance:", impedance_values)


    # ----------------------------- Device Initialization ----------------------------- #
    device = explorepy.Explore()
    device.connect(args.device_name)
    device.stream_processor.subscribe(callback=handle_impedance_packet, topic=TOPICS.imp)
    device.stream_processor.subscribe(callback=handle_exg_packet, topic=TOPICS.raw_ExG)
    device.stream_processor.imp_initialize(notch_freq=50)


    # ----------------------------- Data Acquisition Loop ----------------------------- #
    for _ in range(RECORD_SECONDS):
        time.sleep(1)


    # ----------------------------- Cleanup ----------------------------- #
    device.stream_processor.disable_imp()
    device.stream_processor.unsubscribe(callback=handle_impedance_packet, topic=TOPICS.imp)
    device.stream_processor.unsubscribe(callback=handle_exg_packet, topic=TOPICS.raw_ExG)
    csv_file.close()


    # ----------------------------- Signal Processing ----------------------------- #
    def apply_bandpass_filter(signal_data, low_freq, high_freq, fs, order=3):
        """Apply a Butterworth bandpass filter to the signal."""
        b, a = signal.butter(order, [low_freq / fs, high_freq / fs], btype='bandpass')
        return signal.filtfilt(b, a, signal_data)


    def apply_notch_filter(signal_data, fs, freq=50, quality_factor=30.0):
        """Apply a notch filter at the specified frequency."""
        b, a = signal.iirnotch(freq, quality_factor, fs)
        return signal.filtfilt(b, a, signal_data)


    # ----------------------------- Load and Filter Data ----------------------------- #
    df = pd.read_csv(OUTPUT_FILENAME, delimiter=',', dtype=np.float64)
    raw_ch1 = df['ch1']
    filtered = apply_notch_filter(raw_ch1, FS, freq=62.5)
    filtered = apply_notch_filter(filtered, FS, freq=50)
    filtered = apply_bandpass_filter(filtered, low_freq=0.5, high_freq=30, fs=FS)


    # ----------------------------- Plot Filtered Signal ----------------------------- #
    plt.figure(figsize=(10, 4))
    plt.plot(df['Timestamp'], filtered, label='Filtered ch1')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (µV)")
    plt.title("Filtered EEG Signal - Channel 1")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

