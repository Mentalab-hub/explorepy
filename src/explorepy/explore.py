# -*- coding: utf-8 -*-
from explorepy.bt_client import BtClient
from explorepy.parser import Parser
from explorepy.dashboard.dashboard import Dashboard
from explorepy._exceptions import *
from explorepy.packet import CommandRCV, CommandStatus, CalibrationInfo, DeviceInfo
from explorepy.tools import FileRecorder
import csv
import os
import time
import signal
import sys
from pylsl import StreamInfo, StreamOutlet
from threading import Thread, Timer


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
        packet = None

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
            except ConnectionAbortedError:
                print("Device has been disconnected! Scanning for last connected device...")
                try:
                    self.parser.socket = self.device[device_id].bt_connect()
                except DeviceNotFoundError as e:
                    print(e)
                    return 0

        print("Data acquisition stopped after ", duration, " seconds.")

    def record_data(self, file_name, do_overwrite=False, device_id=0, duration=None, file_type='csv'):
        r"""Records the data in real-time

        Args:
            file_name (str): Output file name
            device_id (int): Device id (not needed in the current version)
            do_overwrite (bool): Overwrite if files exist already
            duration (float): Duration of recording in seconds (if None records endlessly).
            file_type (str): File type of the recorded file. Supported file types: 'csv', 'edf'
        """
        assert self.is_connected, "Explore device is not connected. Please connect the device first."

        # Check invalid characters
        if set(r'<>{}[]~`*%').intersection(file_name):
            raise ValueError("Invalid character in file name")
        n_chan = self.parser.n_chan
        if file_type not in ['edf', 'csv']:
            raise ValueError('{} is not a supported file extension!'.format(file_type))
        time_offset = None
        exg_out_file = file_name + "_ExG"
        orn_out_file = file_name + "_ORN"
        marker_out_file = file_name + "_Marker"

        exg_ch = ['TimeStamp', 'ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8'][0:n_chan+1]
        exg_unit = ['s', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V'][0:n_chan+1]
        exg_max = [86400, 1, 1, 1, 1, 1, 1, 1, 1][0:n_chan + 1]
        exg_min = [0, -1, -1, -1, -1, -1, -1, -1, -1][0:n_chan + 1]
        exg_recorder = FileRecorder(file_name=exg_out_file, ch_label=exg_ch, fs=self.parser.fs, ch_unit=exg_unit,
                                    file_type=file_type, do_overwrite=do_overwrite, ch_min=exg_min, ch_max=exg_max)

        orn_ch = ['TimeStamp', 'ax', 'ay', 'az', 'gx', 'gy', 'gz', 'mx', 'my', 'mz']
        orn_unit = ['s', 'mg', 'mg', 'mg', 'mdps', 'mdps', 'mdps', 'mgauss', 'mgauss', 'mgauss']
        orn_max = [86400, 2000, 2000, 2000, 287000, 287000, 287000, 50000, 50000, 50000]
        orn_min = [0, -2000, -2000, -2000, -287000, -287000, -287000, -50000, -50000, -50000]
        orn_recorder = FileRecorder(file_name=orn_out_file, ch_label=orn_ch, fs=20,
                                    ch_unit=orn_unit, file_type=file_type, do_overwrite=do_overwrite,
                                    ch_min=orn_min, ch_max=orn_max)
        if file_type == 'csv':
            marker_ch = ['TimeStamp', 'Code']
            marker_unit = ['s', '-']
            marker_recorder = FileRecorder(file_name=marker_out_file, ch_label=marker_ch, fs=None,
                                           ch_unit=marker_unit, file_type=file_type, do_overwrite=do_overwrite)
        elif file_type == 'edf':
            marker_recorder = exg_recorder

        is_acquiring = [True]

        def stop_acquiring(flag):
            flag[0] = False

        if duration is not None:
            if duration <= 0:
                raise ValueError("Recording time must be a positive number!")
            rec_timer = Timer(duration, stop_acquiring, [is_acquiring])
            rec_timer.start()
            print("Start recording for ", duration, " seconds...")
        else:
            print("Recording...")
        is_disconnect_occurred = False
        while is_acquiring[0]:
            try:
                packet = self.parser.parse_packet(mode="record", recorders=(exg_recorder, orn_recorder, marker_recorder))
                if time_offset is not None:
                    packet.timestamp = packet.timestamp-time_offset
                else:
                    time_offset = packet.timestamp
            except ConnectionAbortedError:
                print("Device has been disconnected! Scanning for last connected device...")
                try:
                    self.parser.socket = self.device[device_id].bt_connect()
                except DeviceNotFoundError as e:
                    print(e)
                    rec_timer.cancel()
                    return 0

        if is_disconnect_occurred:
            print("Error: Recording finished before ", duration, "seconds.")
            rec_timer.cancel()
        else:
            print("Recording finished after ", duration, " seconds.")
        exg_recorder.stop()
        orn_recorder.stop()
        if file_type == 'csv':
            marker_recorder.stop()

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

            except ConnectionAbortedError:
                print("Device has been disconnected! Scanning for last connected device...")
                try:
                    self.parser.socket = self.device[device_id].bt_connect()
                except DeviceNotFoundError as e:
                    print(e)
                    return 0
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
            except ConnectionAbortedError:
                print("Device has been disconnected! Scanning for last connected device...")
                try:
                    self.parser.socket = self.device[device_id].bt_connect()
                except DeviceNotFoundError as e:
                    print(e)
                    self.is_acquiring[0] = False
                    if mode == "visualize":
                        os._exit(0)
        os.exit(0)

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
            except ConnectionAbortedError:
                print("Device has been disconnected! Scanning for last connected device...")
                try:
                    self.parser.socket = self.device[device_id].bt_connect()
                except DeviceNotFoundError as e:
                    print(e)
                    return 0

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

            except ConnectionAbortedError:
                print("Device has been disconnected! Scanning for last connected device...")
                try:
                    self.parser.socket = self.device[device_id].bt_connect()
                except DeviceNotFoundError as e:
                    print(e)
                    return 0
        if not command_processed:
            print("No status message has been received after ", waiting_time, " seconds. Please restart the device and "
                                                                              "send the command again.")
            return False


if __name__ == '__main__':
    pass
