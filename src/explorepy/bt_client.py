# -*- coding: utf-8 -*-
"""A module for bluetooth connection"""
import time
import bluetooth

from explorepy._exceptions import DeviceNotFoundError, InputError

SPP_UUID = "1101"  # Serial Port Profile (SPP) service


class BtClient:
    """ Responsible for Connecting and reconnecting explore devices via bluetooth"""
    def __init__(self, device_name=None, mac_address=None):
        """Initialize Bluetooth connection

        Args:
            device_name(str): Name of the device (either device_name or device address should be given)
            mac_address(str): Devices MAC address
        """
        if (mac_address is None) and (device_name is None):
            raise InputError("Either name or address options must be provided!")
        self.is_connected = False
        self.mac_address = mac_address
        self.device_name = device_name
        self.socket = None
        self.host = None
        self.port = None

    def connect(self):
        """Connect to the device and return the socket

        Returns:
            socket
        """
        if self.mac_address is None:
            self._find_mac_address()
        else:
            self.device_name = "Explore_" + str(self.mac_address[-5:-3]) + str(self.mac_address[-2:])

        self._find_service()

        for _ in range(5):
            try:
                self.socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
                print("Connecting to {}".format(self.device_name))
                self.socket.connect((self.host, self.port))
                return self.socket
            except bluetooth.BluetoothError as error:
                self.socket.close()
                print(error, "\nCould not connect; Retrying in 2s...")
                time.sleep(2)

        raise DeviceNotFoundError("Could not find the device! Please make sure"
                                  " the device is on and in advertising mode.")

    def reconnect(self):
        """Reconnect to the last used bluetooth socket.

        This function reconnects to the the last bluetooth socket. If after 1 minute the connection doesn't succeed,
        program will end.
        """
        for _ in range(5):
            try:
                self.socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
                self.socket.connect((self.host, self.port))
                return self.socket
            except bluetooth.BluetoothError as error:
                print("Bluetooth Error: {}, attempting to reconnect...".format(error))
                time.sleep(2)

        self.socket.close()
        raise DeviceNotFoundError("Could not find the device! Please make sure the device is on and in"
                                  "advertising mode.")

    def _find_mac_address(self):
        for _ in range(5):
            nearby_devices = bluetooth.discover_devices(lookup_names=True, flush_cache=True)
            for address, name in nearby_devices:
                if name == self.device_name:
                    self.mac_address = address
                    return
            print("No device found with the name: {}, searching again...".format(self.device_name))
            time.sleep(0.1)
        raise DeviceNotFoundError("No device found with the name: {}".format(self.device_name))

    def _find_service(self):
        for _ in range(5):
            service = bluetooth.find_service(uuid=SPP_UUID, address=self.mac_address)
            if service:
                self.port = service[0]["port"]
                self.host = service[0]["host"]
                return 1
        raise DeviceNotFoundError("SSP service for the device {}. Please restart the device and try "
                                  "again".format(self.device_name))

    @staticmethod
    def _check_mac_address(device_name, mac_address):
        return (device_name[-4:-2] == mac_address[-5:-3]) and (device_name[-2:] == mac_address[-2:])
