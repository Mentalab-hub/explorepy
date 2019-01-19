import numpy as np
from .bt_client import BtClient
from .parser import Parser
import csv


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
        self.device[device_id].connect()

    def disconnect(self, device_id=None):
        r"""
        Disconnects from the device
        Args:
            device_id (int): device id (id=None for disconnecting all devices)

        Returns:

        """
        self.device[device_id].socket.close()

    def acquire(self, device_id=0):
        r"""Start getting data from the device """

        if self.parser is None:
            self.parser = Parser(socket=self.device[device_id].socket)
        is_acquiring = True
        while is_acquiring:
            try:
                packet = self.parser.parse_packet(mode="print")
            except ValueError:
                # If value error happens, scan again for devices and try to reconnect (see reconnect function)
                print("Disconnected, scanning for last connected device")
                self.device[device_id].is_connected = False
                is_acquiring = self.device[device_id].reconnect()

    def log_data(self):
        r"""
        Print the data in the terminal/console

        Returns:

        """
        pass

    def record_data(self, file_name, device_id=0):
        r"""
        Records the data in real-time
        Args:
            file_name (str): output file name
            device_id (int): device id

        Returns:

        """
        if self.parser is None:
            self.parser = Parser(socket=self.device[device_id].socket)

        eeg_out_file = file_name + "_eeg.csv"
        orn_out_file = file_name + "_orn.csv"
        # TODO: If there is already a file with the same name, ask the user if he/she wants to replace the file
        with open(eeg_out_file, "w") as f_eeg, open(orn_out_file, "w") as f_orn:
            f_orn.write("TimeStamp, ax, ay, az, gx, gy, gz, mx, my, mz \n")
            f_orn.write(
                "hh:mm:ss, mg/LSB, mg/LSB, mg/LSB, mdps/LSB, mdps/LSB, mdps/LSB, mgauss/LSB, mgauss/LSB, mgauss/LSB\n")
            f_eeg.write("TimeStamp, ch1, ch2, ch3, ch4, ch5, ch6, ch7, ch8\n")
            csv_eeg = csv.writer(f_eeg, delimiter=",")
            csv_orn = csv.writer(f_orn, delimiter=",")

            is_acquiring = True
            while is_acquiring:
                try:
                    packet = self.parser.parse_packet(mode="record", csv_files=(csv_eeg, csv_orn))
                except ValueError:
                    print("Disconnected, scanning for last connected device")
                    self.device[device_id].is_connected = False
                    is_acquiring = self.device[device_id].reconnect()

    def push2lsl(self):
        r"""
        push the stream to lsl

        Returns:

        """
        pass

    def visualize(self):
        r"""
        Start visualization of the data in the viewer
        Returns:

        """
        pass


if __name__ == '__main__':
    pass
