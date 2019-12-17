# -*- coding: utf-8 -*-

from explorepy.bt_client import BtClient
from explorepy.parser import Parser
from explorepy.dashboard.dashboard import Dashboard
import bluetooth
import csv
import os
import time
import signal
import sys
from pylsl import StreamInfo, StreamOutlet
from threading import Thread, Timer
from explorepy.packet import CommandRCV, CommandStatus, CalibrationInfo, MarkerEvent


class Explore:
    r"""Mentalab Explore device"""
    def __init__(self, n_device=1):
        r"""
        Args:
            n_device (int): Number of devices to be connected
        """
        self.device = []
        self.socket = None
        self.parser = None
        self.m_dashboard = None
        for i in range(n_device):
            self.device.append(BtClient())
        self.is_connected = False
        self.is_acquiring = None

    def connect(self, device_name=None, device_addr=None, device_id=0):
        r"""
        Connects to the nearby device. If there are more than one device, the user is asked to choose one of them.

        Args:
            device_name (str): Device name in the format of "Explore_XXXX"
            device_addr (str): The MAC address in format "XX:XX:XX:XX:XX:XX" Either Address or name should be in the input
            device_id (int): device id (not needed in the current version)

        """

        self.device[device_id].init_bt(device_name=device_name, device_addr=device_addr)
        if self.socket is None:
            self.socket = self.device[device_id].bt_connect()

        if self.parser is None:
            self.parser = Parser(socket=self.socket)
        self.is_connected = True

    def disconnect(self, device_id=None):
        r"""Disconnects from the device

        Args:
            device_id (int): device id (not needed in the current version)
        """
        self.device[device_id].socket.close()
        self.is_connected = False

    def acquire(self, device_id=0, duration=None):
        r"""Start getting data from the device

        Args:
            device_id (int): device id (not needed in the current version)
            duration (float): duration of acquiring data (if None it streams data endlessly)
        """

        assert self.is_connected, "Explore device is not connected. Please connect the device first."

        is_acquiring = [True]

        def stop_acquiring(flag):
            flag[0] = False

        if duration is not None:
            Timer(duration, stop_acquiring, [is_acquiring]).start()
            print("Start acquisition for ", duration, " seconds...")

        while is_acquiring[0]:
            try:
                self.parser.parse_packet(mode="print")
            except ValueError:
                # If value error happens, scan again for devices and try to reconnect (see reconnect function)
                print("Disconnected, scanning for last connected device")
                socket = self.device[device_id].bt_connect()
                self.parser.socket = socket
            except bluetooth.BluetoothError as error:
                print("Bluetooth Error: attempting reconnect. Error: ", error)
                self.parser.socket = self.device[device_id].bt_connect()

        print("Data acquisition stopped after ", duration, " seconds.")

    def record_data(self, file_name, do_overwrite=False, device_id=0, duration=None):
        r"""Records the data in real-time

        Args:
            file_name (str): output file name
            device_id (int): device id (not needed in the current version)
            do_overwrite (bool): Overwrite if files exist already
            duration (float): Duration of recording in seconds (if None records endlessly).
        """
        assert self.is_connected, "Explore device is not connected. Please connect the device first."

        # Check invalid characters
        if set(r'[<>/{}[\]~`]*%').intersection(file_name):
            raise ValueError("Invalid character in file name")

        time_offset = None
        exg_out_file = file_name + "_ExG.csv"
        orn_out_file = file_name + "_ORN.csv"
        marker_out_file = file_name + "_Marker.csv"
        meta_data_file = file_name + "_Metadata.csv"

        if not do_overwrite:
            assert not os.path.isfile(exg_out_file), exg_out_file + " already exists!"
            assert not os.path.isfile(orn_out_file), orn_out_file + " already exists!"
            assert not os.path.isfile(marker_out_file), marker_out_file + " already exists!"
            assert not os.path.isfile(meta_data_file), meta_data_file + " already exists!"

        with open(exg_out_file, "w") as f_exg, open(orn_out_file, "w") as f_orn, \
                open(marker_out_file, "w") as f_marker, open(meta_data_file, "w") as f_metadata:
            f_orn.write("TimeStamp,ax,ay,az,gx,gy,gz,mx,my,mz\n")
            f_exg.write("TimeStamp,ch1,ch2,ch3,ch4,ch5,ch6,ch7,ch8\n")
            f_marker.write("TimeStamp,Marker_code\n")
            f_metadata.write("TimeStamp,firmware_version, data_rate_info, adc_mask\n")

            csv_exg = csv.writer(f_exg, delimiter=",")
            csv_orn = csv.writer(f_orn, delimiter=",")
            csv_marker = csv.writer(f_marker, delimiter=",")
            csv_metadata = csv.writer(f_metadata, delimiter=",")

            is_acquiring = [True]

            def stop_acquiring(flag):
                flag[0] = False

            if duration is not None:
                Timer(duration, stop_acquiring, [is_acquiring]).start()
                print("Start recording for ", duration, " seconds...")
            else:
                print("Recording...")

            while is_acquiring[0]:
                try:
                    packet = self.parser.parse_packet(mode="record", csv_files=(csv_exg, csv_orn, csv_marker, csv_metadata))
                    if time_offset is not None:
                        packet.timestamp = packet.timestamp-time_offset
                    else:
                        time_offset = packet.timestamp

                except ValueError:
                    # If value error happens, scan again for devices and try to reconnect (see reconnect function)
                    print("Disconnected, scanning for last connected device")
                    self.parser.socket = self.device[device_id].bt_connect()
                except bluetooth.BluetoothError as error:
                    print("Bluetooth Error: Timeout, attempting reconnect. Error: ", error)
                    self.parser.socket = self.device[device_id].bt_connect()
            print("Recording finished after ", duration, " seconds.")
            f_marker.close()
            f_exg.close()
            f_orn.close()

    def push2lsl(self, n_chan, device_id=0, duration=None, sampling_rate=250):
        r"""Push samples to two lsl streams

        Args:
            device_id (int): device id (not needed in the current version)
            n_chan (int): Number of channels (4 or 8)
            duration (float): duration of data acquiring (if None it streams endlessly).
            sampling_rate : sampling_rate of ExG data stream
        """

        assert (n_chan is not None), "Number of channels missing"
        assert self.is_connected, "Explore device is not connected. Please connect the device first."

        info_orn = StreamInfo('Explore', 'Orientation', 9, 20, 'float32', 'ORN')
        info_exg = StreamInfo('Explore', 'ExG', n_chan, sampling_rate, 'float32', 'ExG')
        info_marker = StreamInfo('Explore', 'Markers', 1, 0, 'int32', 'Marker')

        orn_outlet = StreamOutlet(info_orn)
        exg_outlet = StreamOutlet(info_exg)
        marker_outlet = StreamOutlet(info_marker)

        is_acquiring = [True]

        def stop_acquiring(flag):
            flag[0] = False

        if duration is not None:
            Timer(duration, stop_acquiring, [is_acquiring]).start()
            print("Start pushing to lsl for ", duration, " seconds...")
        else:
            print("Pushing to lsl...")

        while is_acquiring[0]:

            try:
                self.parser.parse_packet(mode="lsl", outlets=(orn_outlet, exg_outlet, marker_outlet))
            except ValueError:
                # If value error happens, scan again for devices and try to reconnect (see reconnect function)
                print("Disconnected, scanning for last connected device")
                self.socket = self.device[device_id].bt_connect()
                time.sleep(1)
                self.parser = Parser(self.socket)

            except bluetooth.BluetoothError as error:
                print("Bluetooth Error: Timeout, attempting reconnect. Error: ", error)
                self.socket = self.device[device_id].bt_connect()
                time.sleep(1)
                self.parser = Parser(self.socket)
        print("Data acquisition finished after ", duration, " seconds.")

    def visualize(self, n_chan, device_id=0, bp_freq=(1, 30), notch_freq=50, sampling_rate=250):
        r"""Visualization of the signal in the dashboard
        Args:
            n_chan (int): Number of channels device_id (int): Device ID (in case of multiple device connection)
            device_id (int): Device ID (not needed in the current version)
            bp_freq (tuple): Bandpass filter cut-off frequencies (low_cutoff_freq, high_cutoff_freq), No bandpass filter
            if it is None.
            notch_freq (int): Line frequency for notch filter (50 or 60 Hz), No notch filter if it is None
            sampling_rate : sampling_rate of ExG data stream
        """
        assert self.is_connected, "Explore device is not connected. Please connect the device first."

        self.m_dashboard = Dashboard(n_chan=n_chan, sampling_rate=sampling_rate)
        self.m_dashboard.start_server()

        thread = Thread(target=self._io_loop)
        thread.setDaemon(True)
        thread.start()

        self.parser = Parser(socket=self.socket, bp_freq=bp_freq, notch_freq=notch_freq, sampling_rate=sampling_rate, \
                             n_chan=n_chan)
        self.m_dashboard.start_loop()

    def _io_loop(self, device_id=0, mode="visualize"):
        self.is_acquiring = [True]
        # Wait until dashboard is initialized.
        while not hasattr(self.m_dashboard, 'doc'):
            print('wait...')
            time.sleep(.5)

        while self.is_acquiring[0]:
            try:
                packet = self.parser.parse_packet(mode=mode, dashboard=self.m_dashboard)
            except ValueError:
                # If value error happens, scan again for devices and try to reconnect (see reconnect function)
                print("Disconnected, scanning for last connected device")
                socket = self.device[device_id].bt_connect()
                self.parser.socket = socket
            except bluetooth.BluetoothError as error:
                print("Bluetooth Error: attempting reconnect. Error: ", error)
                self.parser.socket = self.device[device_id].bt_connect()
        sys.exit(0)

    def signal_handler(self, signal, frame):
        # Safe handler of keyboardInterrupt
        self.is_acquiring = [False]
        print("Program is exiting...")
        sys.exit(0)

    def measure_imp(self, n_chan, device_id=0, notch_freq=50, sampling_rate=250):
        """
        Visualization of the electrode impedances

        Args:
            n_chan (int): Number of channels
            device_id (int): Device ID
            notch_freq (int): Notch frequency for filtering the line noise (50 or 60 Hz)
            sampling_rate (int): Sampling rate of the device

        Returns:

        """
        assert self.is_connected, "Explore device is not connected. Please connect the device first."
        self.is_acquiring = [True]

        signal.signal(signal.SIGINT, self.signal_handler)

        try:
            thread = Thread(target=self._io_loop, args=(device_id, "impedance",))
            thread.setDaemon(True)
            self.parser = Parser(socket=self.socket, bp_freq=(61, 64), notch_freq=notch_freq, sampling_rate=sampling_rate)
            thread.start()

            # Activate impedance measurement mode in the device
            from explorepy import command
            imp_activate_cmd = command.ZmeasurementEnable()
            if self.change_settings(imp_activate_cmd):
                self.m_dashboard = Dashboard(n_chan=n_chan, mode="impedance", sampling_rate=sampling_rate,
                                             firmware_version=self.parser.firmware_version)
                self.m_dashboard.start_server()
                self.m_dashboard.start_loop()
            else:
                os._exit(0)
        finally:
            print("Disabling impedance mode...")
            from explorepy import command
            imp_deactivate_cmd = command.ZmeasurementDisable()
            self.change_settings(imp_deactivate_cmd)
            sys.exit(0)

    def change_settings(self, command, device_id=0):
        """
        sends a message to the device
        Args:
            device_id (int): Device ID
            command (explorepy.command.Command): Command object

        Returns:

        """
        from explorepy.command import send_command

        assert self.is_connected, "Explore device is not connected. Please connect the device first."

        sending_attempt = 5
        while sending_attempt:
            try:
                sending_attempt = sending_attempt-1
                time.sleep(0.1)
                send_command(command, self.socket)
                sending_attempt = 0
            except ValueError:
                # If value error happens, scan again for devices and try to reconnect (see reconnect function)
                print("Disconnected, scanning for last connected device")
                socket = self.device[device_id].bt_connect()
                self.parser.socket = socket
            except bluetooth.BluetoothError as error:
                print("Bluetooth Error: attempting reconnect. Error: ", error)
                self.parser.socket = self.device[device_id].bt_connect()

        is_listening = [True]
        command_processed = False

        def stop_listening(flag):
            flag[0] = False

        waiting_time = 10
        command_timer = Timer(waiting_time, stop_listening, [is_listening])
        command_timer.start()
        print("waiting for ack and status messages...")
        while is_listening[0]:
            try:
                packet = self.parser.parse_packet(mode="listen")

                if isinstance(packet, CommandRCV):
                    temp = command.int2bytearray(packet.opcode, 1)
                    if command.int2bytearray(packet.opcode, 1) == command.opcode.value:
                        print("The opcode matches the sent command, Explore has received the command")
                if isinstance(packet, CalibrationInfo):
                    self.parser.imp_calib_info['slope'] = packet.slope
                    self.parser.imp_calib_info['offset'] = packet.offset
                    
                if isinstance(packet, CommandStatus):
                    if command.int2bytearray(packet.opcode, 1) == command.opcode.value:
                        command_processed = True
                        is_listening = [False]
                        command_timer.cancel()
                        print("The opcode matches the sent command, Explore has processed the command")
                        return True

            except ValueError:
                # If value error happens, scan again for devices and try to reconnect (see reconnect function)
                print("Disconnected, scanning for last connected device")
                socket = self.device[device_id].bt_connect()
                self.parser.socket = socket
            except bluetooth.BluetoothError as error:
                print("Bluetooth Error: attempting reconnect. Error: ", error)
                self.parser.socket = self.device[device_id].bt_connect()
        if not command_processed:
            print("No status message has been received after ", waiting_time, " seconds. Please restart the device and "
                                                                              "send the command again.")
            return False


if __name__ == '__main__':
    pass
