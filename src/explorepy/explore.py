# -*- coding: utf-8 -*-

from explorepy.bt_client import BtClient
from explorepy.parser import Parser
from explorepy.dashboard.dashboard import Dashboard
import bluetooth
import csv
import os
import time
from pylsl import StreamInfo, StreamOutlet
from threading import Thread, Timer
from datetime import datetime
from explorepy.packet import Orientation, Environment, TimeStamp, Disconnect, DeviceInfo, EEG, EEG94, EEG98, EEG99s, \
    CommandRCV, CommandStatus


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

    def disconnect(self, device_id=None):
        r"""Disconnects from the device

        Args:
            device_id (int): device id (not needed in the current version)
        """
        self.device[device_id].socket.close()

    def acquire(self, device_id=0, duration=None):
        r"""Start getting data from the device

        Args:
            device_id (int): device id (not needed in the current version)
            duration (float): duration of acquiring data (if None it streams data endlessly)
        """

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
        # Check invalid characters
        if set(r'[<>/{}[\]~`]*%').intersection(file_name):
            raise ValueError("Invalid character in file name")

        time_offset = None
        exg_out_file = file_name + "_ExG.csv"
        orn_out_file = file_name + "_ORN.csv"
        marker_out_file = file_name + "_Marker.csv"

        if not do_overwrite:
            assert not os.path.isfile(exg_out_file), exg_out_file + " already exists!"
            assert not os.path.isfile(orn_out_file), orn_out_file + " already exists!"
            assert not os.path.isfile(marker_out_file), marker_out_file + " already exists!"

        with open(exg_out_file, "w") as f_exg, open(orn_out_file, "w") as f_orn, open(marker_out_file, "w") as f_marker:
            f_orn.write("TimeStamp,ax,ay,az,gx,gy,gz,mx,my,mz\n")
            # f_orn.write(
            #     "hh:mm:ss,mg/LSB,mg/LSB,mg/LSB,mdps/LSB,mdps/LSB,mdps/LSB,mgauss/LSB,mgauss/LSB,mgauss/LSB\n")
            f_exg.write("TimeStamp,ch1,ch2,ch3,ch4,ch5,ch6,ch7,ch8\n")
            f_marker.write("TimeStamp,Marker_code\n")

            csv_exg = csv.writer(f_exg, delimiter=",")
            csv_orn = csv.writer(f_orn, delimiter=",")
            csv_marker = csv.writer(f_marker, delimiter=",")

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
                    # self.parser.parse_packet()
                    packet = self.parser.parse_packet(mode="record", csv_files=(csv_exg, csv_orn, csv_marker))
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

    def push2lsl(self, n_chan, device_id=0, duration=None):
        r"""Push samples to two lsl streams

        Args:
            device_id (int): device id (not needed in the current version)
            n_chan (int): Number of channels (4 or 8)
            duration (float): duration of data acquiring (if None it streams endlessly).
        """

        assert (n_chan is not None), "Number of channels missing"

        info_orn = StreamInfo('Explore', 'Orientation', 9, 20, 'float32', 'ORN')
        info_exg = StreamInfo('Explore', 'ExG', n_chan, 250, 'float32', 'ExG')
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

    def visualize(self, n_chan, device_id=0, bp_freq=(1, 30), notch_freq=50):
        r"""Visualization of the signal in the dashboard

        Args:
            n_chan (int): Number of channels device_id (int): Device ID (in case of multiple device connection)
            device_id (int): Device ID (not needed in the current version)
            bp_freq (tuple): Bandpass filter cut-off frequencies (low_cutoff_freq, high_cutoff_freq), No bandpass filter
            if it is None.
            notch_freq (int): Line frequency for notch filter (50 or 60 Hz), No notch filter if it is None
        """
        self.m_dashboard = Dashboard(n_chan=n_chan)
        self.m_dashboard.start_server()

        thread = Thread(target=self._io_loop)
        thread.setDaemon(True)
        thread.start()

        self.parser = Parser(socket=self.socket, bp_freq=bp_freq, notch_freq=notch_freq)

        self.m_dashboard.start_loop()

    def _io_loop(self, device_id=0, mode="visualize"):
        is_acquiring = True

        # Wait until dashboard is initialized.
        while not hasattr(self.m_dashboard, 'doc'):
            print('wait')
            time.sleep(.2)
        while is_acquiring:
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

    def measure_imp(self, n_chan, device_id=0, notch_freq=50):
        self.m_dashboard = Dashboard(n_chan=n_chan, mode="impedance")
        self.m_dashboard.start_server()

        thread = Thread(target=self._io_loop, args=(device_id, "impedance",))
        thread.setDaemon(True)
        thread.start()

        self.parser = Parser(socket=self.socket, bp_freq=(61, 64), notch_freq=notch_freq)

        self.m_dashboard.start_loop()
        #
        # while True:
        #     try:
        #         self.parser.parse_packet(mode="impedance")
        #     except ValueError:
        #         # If value error happens, scan again for devices and try to reconnect (see reconnect function)
        #         print("Disconnected, scanning for last connected device")
        #         socket = self.device[device_id].bt_connect()
        #         self.parser.socket = socket
        #     except bluetooth.BluetoothError as error:
        #         print("Bluetooth Error: attempting reconnect. Error: ", error)
        #         self.parser.socket = self.device[device_id].bt_connect()

    def pass_msg(self, device_id=0, msg2send=None):
        r"""
        sends a set of parameters to the device
        Returns:
        sample commands and messages:
        msg = Host time stamp: default message if the msg2send field is empty
        msg = b'\xA0\x00\x0A\x00\xda\xba\xad\xde\xA1\x02\xaf\xbe\xad\xde' it is the command to switch to 500sps mode
        msg = b'\xA0\x00\x0A\x00\xda\xba\xad\xde\xA3\x00\xaf\xbe\xad\xde' it is the command to format memory

        example:
        myexplore.pass_msg(msg2send=command.Command.FORMAT_MEMORY.value)
        """

        if self.socket is None:
            self.socket = self.device[device_id].bt_connect()

        if self.parser is None:
            self.parser = Parser(socket=self.socket)

        if msg2send is None:
            # current date and time
            msg_is_command = 0
            now = datetime.now()
            print(now)
            timestamp = int(1000000000 * datetime.timestamp(now))  # time stamp in nanosecond
            ts_str = hex(timestamp)
            ts_str = ts_str[2:18]
            host_ts = bytes.fromhex(ts_str)
            ID = b'\x1B'
            CNT = b'\x01'
            Payload = b'\x10\x00' # i.e. 0x0010
            device_ts = b'\x00\x00\x00\x00'
            Fletcher = b'\xFF\xFF\xFF\xFF'
            msg2send = ID + CNT + Payload + device_ts + host_ts + Fletcher
        else:
            msg_is_command = msg2send[-6]
        is_sending = True

        while is_sending:
            try:
                time.sleep(0.1)
                self.parser.send_msg(msg2send)
                print(" Message Sent :)")
                is_sending = False
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

        Timer(100, stop_listening, [is_listening]).start()
        print("waiting for ack and status messages...")
        while is_listening[0]:
            try:
                packet = self.parser.parse_packet(mode="listen")
                if isinstance(packet, CommandRCV):
                    if packet.opcode == msg_is_command:
                        print ("the opcode matches the sent command, Explore has received the command")
                if isinstance(packet, CommandStatus):
                    if packet.opcode == msg_is_command:
                        print ("the opcode matches the sent command, Explore has processed the command")
                        is_listening = [False]
                        command_processed = True

            except ValueError:
                # If value error happens, scan again for devices and try to reconnect (see reconnect function)
                print("Disconnected, scanning for last connected device")
                socket = self.device[device_id].bt_connect()
                self.parser.socket = socket
            except bluetooth.BluetoothError as error:
                print("Bluetooth Error: attempting reconnect. Error: ", error)
                self.parser.socket = self.device[device_id].bt_connect()
        if not command_processed:
            print("No status message has been received after ", 100, " seconds. Please send the command again")


if __name__ == '__main__':
    pass
