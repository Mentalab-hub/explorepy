# -*- coding: utf-8 -*-
from explorepy.bt_client import BtClient
from explorepy.parser import Parser
from explorepy.dashboard.dashboard import Dashboard
from explorepy._exceptions import *
from explorepy.packet import CommandRCV, CommandStatus, CalibrationInfo, DeviceInfo
from explorepy.tools import create_exg_recorder, create_orn_recorder, create_marker_recorder
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

        exg_recorder = create_exg_recorder(filename=exg_out_file,
                                           file_type=file_type,
                                           fs=self.parser.fs,
                                           adc_mask=self.parser.adc_mask,
                                           do_overwrite=do_overwrite)
        orn_recorder = create_orn_recorder(filename=orn_out_file,
                                           file_type=file_type,
                                           do_overwrite=do_overwrite)

        if file_type == 'csv':
            marker_ch = ['TimeStamp', 'Code']
            marker_unit = ['s', '-']
            marker_recorder = create_marker_recorder(filename=marker_out_file, do_overwrite=do_overwrite)
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

    def push2lsl(self, device_id=0, duration=None):
        r"""Push samples to two lsl streams

        Args:
            device_id (int): device id (not needed in the current version)
            duration (float): duration of data acquiring (if None it streams endlessly).
        """

        assert self.is_connected, "Explore device is not connected. Please connect the device first."

        info_orn = StreamInfo('Explore', 'Orientation', 9, 20, 'float32', 'ORN')
        info_exg = StreamInfo('Explore', 'ExG', self.parser.n_chan, self.parser.fs, 'float32', 'ExG')
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

    def visualize(self, device_id=0, bp_freq=(1, 30), notch_freq=50, calibre_file=None):
        r"""Visualization of the signal in the dashboard
        Args:
            device_id (int): Device ID (not needed in the current version)
            bp_freq (tuple): Bandpass filter cut-off frequencies (low_cutoff_freq, high_cutoff_freq), No bandpass filter
            if it is None.
            notch_freq (int): Line frequency for notch filter (50 or 60 Hz), No notch filter if it is None
            calibre_file (str): Calibration data file name
        """
        import numpy as np
        assert self.is_connected, "Explore device is not connected. Please connect the device first."
        if calibre_file is not None:
            with open(calibre_file, "r") as f_calibre:
                csv_reader_calibre = csv.reader(f_calibre, delimiter=",")
                calibre_set = list(csv_reader_calibre)
                self.parser.calibre_set = np.asarray(calibre_set[1], dtype=np.float64)
        self.parser.notch_freq = notch_freq
        if bp_freq is not None:
            self.parser.apply_bp_filter = True
            self.parser.bp_freq = bp_freq

        self.m_dashboard = Dashboard(n_chan=self.parser.n_chan,
                                     exg_fs=self.parser.fs,
                                     firmware_version=self.parser.firmware_version)
        self.m_dashboard.start_server()

        thread = Thread(target=self._io_loop)
        thread.setDaemon(True)
        thread.start()
        self.m_dashboard.start_loop()

    def _io_loop(self, device_id=0, mode="visualize"):
        self.is_acquiring = [True]
        if self.parser.calibre_set is not None:
            is_initialized = False
        else:
            is_initialized = True # flag as True since it doesn't matter and we skip orientation calculation process
        # Wait until dashboard is initialized.
        while not hasattr(self.m_dashboard, 'doc'):
            print('wait...')
            time.sleep(.5)

        while self.is_acquiring[0]:
            if is_initialized:
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
            else:
                try:
                    packet = self.parser.parse_packet(mode="initialize", dashboard=self.m_dashboard)
                    if hasattr(packet, 'acc'):
                        if self.parser.init_set is not None:
                            is_initialized = True
                except ConnectionAbortedError:
                    print("Device has been disconnected! Scanning for last connected device...")
                    try:
                        self.parser.socket = self.device[device_id].bt_connect()
                    except DeviceNotFoundError as e:
                        print(e)
                        self.is_acquiring[0] = False
                        if mode == "visualize":
                            os._exit(0)

    def signal_handler(self, signal, frame):
        # Safe handler of keyboardInterrupt
        self.is_acquiring = [False]
        print("Program is exiting...")
        sys.exit(0)

    def measure_imp(self, device_id=0, notch_freq=50):
        """
        Visualization of the electrode impedances

        Args:
            device_id (int): Device ID
            notch_freq (int): Notch frequency for filtering the line noise (50 or 60 Hz)
        """
        assert self.is_connected, "Explore device is not connected. Please connect the device first."
        assert self.parser.fs == 250, "Impedance mode only works in 250 Hz sampling rate!"
        self.is_acquiring = [True]

        signal.signal(signal.SIGINT, self.signal_handler)

        try:
            thread = Thread(target=self._io_loop, args=(device_id, "impedance",))
            thread.setDaemon(True)
            self.parser.apply_bp_filter = True
            self.parser.bp_freq = (61, 64)
            self.parser.notch_freq = notch_freq
            thread.start()

            # Activate impedance measurement mode in the device
            from explorepy import command
            imp_activate_cmd = command.ZmeasurementEnable()
            if self.change_settings(imp_activate_cmd):
                self.m_dashboard = Dashboard(n_chan=self.parser.n_chan, mode="impedance", exg_fs=self.parser.fs,
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

    def set_marker(self, code):
        """Sets an event marker during the recording

        Args:
            code (int): Marker code. It must be an integer larger than 7 (codes from 0 to 7 are reserved for hardware markers).

        """
        assert self.is_connected, "Explore device is not connected. Please connect the device first."
        self.parser.set_marker(marker_code=code)

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

    def calibrate_orn(self, file_name, device_id=0, do_overwrite=False):
        r"""Calibrate the orientation module of the specified device

        Args:
            device_id (int): device id
            file_name (str): filename to be used for calibration. If you pass this parameter, ORN module should be ACTIVE!
            do_overwrite (bool): Overwrite if files exist already
        """
        print("Start recording for 100 seconds, please move the device around during this time, in all directions")
        self.record_data(file_name, do_overwrite=do_overwrite, device_id=device_id, duration=100, file_type='csv')
        calibre_out_file = file_name + "_calibre_coef.csv"
        assert not (os.path.isfile(calibre_out_file) and do_overwrite), calibre_out_file + " already exists!"
        import numpy as np
        with open((file_name + "_ORN.csv"), "r") as f_set, open(calibre_out_file, "w") as f_coef:
            f_coef.write("kx, ky, kz, mx_offset, my_offset, mz_offset\n")
            csv_reader = csv.reader(f_set, delimiter=",")
            csv_coef = csv.writer(f_coef, delimiter=",")
            np_set = list(csv_reader)
            np_set = np.array(np_set[1:], dtype=np.float)
            mag_set_x = np.sort(np_set[:, -3])
            mag_set_y = np.sort(np_set[:, -2])
            mag_set_z = np.sort(np_set[:, -1])
            mx_offset = 0.5 * (mag_set_x[0] + mag_set_x[-1])
            my_offset = 0.5 * (mag_set_y[0] + mag_set_y[-1])
            mz_offset = 0.5 * (mag_set_z[0] + mag_set_z[-1])
            kx = 0.5 * (mag_set_x[-1] - mag_set_x[0])
            ky = 0.5 * (mag_set_y[-1] - mag_set_y[0])
            kz = 0.5 * (mag_set_z[-1] - mag_set_z[0])
            k = np.sort(np.array([kx, ky, kz]))
            kx = 1 / kx
            ky = 1 / ky
            kz = 1 / kz
            calibre_set = np.array([kx, ky, kz, mx_offset, my_offset, mz_offset])
            csv_coef.writerow(calibre_set)
            f_set.close()
            f_coef.close()
        os.remove((file_name + "_ORN.csv"))
        os.remove((file_name + "_ExG.csv"))
        os.remove((file_name + "_Marker.csv"))


if __name__ == '__main__':
    pass
