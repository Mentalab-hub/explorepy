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
        for i in range(n_device):
            self.device.append(BtClient())

    def connect(self, device_id=0):
        r"""

        Args:
            id (int): device id

        Returns:

        """
        self.device[device_id].connect()

    def disconnect(self, device_id=None):
        r"""

        Args:
            id: device id (id=None for disconnecting all devices)

        Returns:

        """
        self.device[device_id].socket.close()

    def acquire(self, device_id=0):
        r"""
        Start getting data from the device

        """
        exp_parser = Parser(socket=self.device[device_id].socket)
        try:
            while True:
                pid, timestamp, data = exp_parser.parse_packet()
                print("packet ID: [%i]" % pid)
        except ValueError:
            # If value error happens, scan again for devices and try to reconnect (see reconnect function)
            print("Disconnected, scanning for last connected device")
            self.device[device_id].is_connected = False
            self.device[device_id].reconnect()
            self.acquire(device_id)

    def logData(self):
        r"""
        Print the data in the terminal/console

        Returns:

        """
        pass

    def recordData(self, fileName, device_id=0):
        exp_parser = Parser(socket=self.device[device_id].socket)

        eeg_out_file = fileName + '_eeg.csv'
        orn_out_file = fileName + '_orn.csv'

        with open(eeg_out_file, "w") as f_eeg, open(orn_out_file, "w") as f_orn:
            f_orn.write('TimeStamp, ax, ay, az, gx, gy, gz, mx, my, mz \n')
            f_orn.write(
                'hh:mm:ss, mg/LSB, mg/LSB, mg/LSB, mdps/LSB, mdps/LSB, mdps/LSB, mgauss/LSB, mgauss/LSB, mgauss/LSB\n')
            f_eeg.write('TimeStamp, ch1, ch2, ch3, ch4, ch5, ch6, ch7, ch8\n')
            csv_eeg = csv.writer(f_eeg, delimiter=',')
            csv_orn = csv.writer(f_orn, delimiter=',')
            pid, timestamp, data = exp_parser.parse_packet(mode='read')
            while pid:
                if pid == 144:
                   data[0, :] = timestamp
                    csv_eeg.writerows(data.T.tolist())
                if pid == 146:
                   data[0, :] = timestamp
                    csv_eeg.writerows([timestamp] + data.T.tolist())
                if pid == 30:
                    data[0, :] = timestamp
                    csv_eeg.writerows(data.T.tolist())
                if pid == 62:
                    csv_eeg.writerows(data.T.tolist())
                if pid == 13:
                    csv_orn.writerow([timestamp] + data.tolist())

                pid, timestamp, data = exp_parser.parse_packet(mode='read')
        try:
            while True:
                pid, timestamp, data = exp_parser.parse_packet(fileName=fileName, mode='record')
                print("package ID: [%i]" % pid)
        except ValueError:
            # If value error happens, scan again for devices and try to reconnect (see reconnect function)
            print("Disconnected, scanning for last connected device")
            self.device[device_id].is_connected = False
            self.device[device_id].reconnect()
            self.recordData(fileName, device_id)

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
