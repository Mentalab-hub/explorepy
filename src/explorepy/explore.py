import numpy as np
from .bt_client import BtClient
from .parser import Parser
import bluetooth
import csv
import os
import time
from pylsl import StreamInfo, StreamOutlet


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
        for i in range(n_device):
            self.device.append(BtClient())

    def connect(self, device_id=0):
        r"""
        Connects to the nearby device. If there are more than one device, the user is asked to choose one of them
        Args:
            device_id (int): device id

        """
        self.device[device_id].initBT()

    def disconnect(self, device_id=None):
        r"""
        Disconnects from the device
        Args:
            device_id (int): device id (id=None for disconnecting all devices)

        Returns:

        """
        self.device[device_id].socket.close()

    def acquire(self, device_id=0):
        r"""
        Start getting data from the device
        Args:
            device_id (int): device id (id=None for disconnecting all devices)
        """

        self.socket = self.device[device_id].bt_connect()

        if self.parser is None:
            self.parser = Parser(self.socket)

        is_acquiring = True
        while is_acquiring:
            try:
                packet = self.parser.parse_packet(mode="print")
            except ValueError:
                # If value error happens, scan again for devices and try to reconnect (see reconnect function)
                print("Disconnected, scanning for last connected device")
                socket = self.device[device_id].bt_connect()
                self.parser.socket = socket
            except bluetooth.BluetoothError as error:
                print("Bluetooth Error: attempting reconnect. Error: ", error)
                self.parser.socket = self.device[device_id].bt_connect()

    def record_data(self, file_name, device_id=0):
        r"""
        Records the data in real-time
        Args:
            file_name (str): output file name
            device_id (int): device id

        Returns:

        """
        self.socket = self.device[device_id].bt_connect()

        if self.parser is None:
            self.parser = Parser(self.socket)

        eeg_out_file = file_name + "_eeg.csv"
        orn_out_file = file_name + "_orn.csv"
        # TODO: If there is already a file with the same name, ask the user if he/she wants to replace the file

        c = None
        if os.path.isfile(eeg_out_file):
            c = input("A file with this name already exist, are you sure you want to proceed? [Enter y/n]")
            while True:
                if c == 'n':
                    exit()
                elif c == 'y':
                    break
                else:
                    c = input("A file with this name already exist, are you sure you want to proceed? [Enter y/n]")
        while True:
            with open(eeg_out_file, "w") as f_eeg, open(orn_out_file, "w") as f_orn:
                f_orn.write("TimeStamp, ax, ay, az, gx, gy, gz, mx, my, mz \n")
                f_orn.write(
                    "hh:mm:ss, mg/LSB, mg/LSB, mg/LSB, mdps/LSB, mdps/LSB, mdps/LSB, mgauss/LSB, mgauss/LSB, mgauss/LSB\n")
                f_eeg.write("TimeStamp, ch1, ch2, ch3, ch4, ch5, ch6, ch7, ch8\n")
                csv_eeg = csv.writer(f_eeg, delimiter=",")
                csv_orn = csv.writer(f_orn, delimiter=",")

                is_acquiring = True
                print("Recording...")
                while is_acquiring:
                    try:
                        packet = self.parser.parse_packet(mode="record", csv_files=(csv_eeg, csv_orn))
                    except ValueError:
                        # If value error happens, scan again for devices and try to reconnect (see reconnect function)
                        print("Disconnected, scanning for last connected device")
                        self.parser.socket = self.device[device_id].bt_connect()
                    except bluetooth.BluetoothError as error:
                        print("Bluetooth Error: Probably timeout, attempting reconnect. Error: ", error)
                        self.parser.socket = self.device[device_id].bt_connect()

    def push2lsl(self, device_id=0):
        r"""
        push the stream to lsl

        Returns:

        """



        self.Socket=self.device[device_id].bt_connect()

        if self.parser is None:
            self.parser = Parser(self.Socket)

        #Create 2 stream infos,
        info_orn = StreamInfo('Mentalab', 'Orientation', 8, 100, 'float32', 'myuid34234')
        info_eeg =StreamInfo('Mentalab', 'EEG', 8, 100, 'float32', 'myuid34234')
        r"""Start getting data from the device """

        print("now sending data...")

        # Create Outlets and push data to LSL
        is_acquiring = True
        while is_acquiring:

            try:
                packet = self.parser.parse_packet(mode="lsl", outlets=(StreamOutlet(info_orn),StreamOutlet(info_eeg)))
            except ValueError:
                # If value error happens, scan again for devices and try to reconnect (see reconnect function)
                print("Disconnected, scanning for last connected device")
                self.Socket = self.device[device_id].bt_connect()
                time.sleep(1)
                self.parser = Parser(self.Socket)

                pass

            except bluetooth.BluetoothError as error:
                print("Bluetooth Error: Probably timeout, attempting reconnect. Error: ", error)
                self.Socket = self.device[device_id].bt_connect()
                time.sleep(1)
                self.parser = Parser(self.Socket)

                pass





    def visualize(self):
        r"""
        Start visualization of the data in the viewer
        Returns:

        """
        pass


if __name__ == '__main__':
    pass
