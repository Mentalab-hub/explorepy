# -*- coding: utf-8 -*-
import bluetooth
import time
import sys


class BtClient:
    """ Responsible for Connecting and reconnecting explore devices via bluetooth"""

    def __init__(self):
        self.is_connected = False
        self.lastUsedAddress = None
        self.socket = None
        self.host = None
        self.port = None
        self.name = None

    def init_bt(self, device_name=None, device_addr=None):
        """
        Initialize Bluetooth connection

        Args:
            device_name(str): Name of the device (either device_name or device address should be given)
            device_addr(str): Devices MAC address
        """
        assert (device_addr is not None) or (device_name is not None), "Missing name or address"

        if device_name is not None:
            if device_addr is None:
                assert self.find_mac_addr(device_name), "Error: Couldn't find the device! Restart your device and " \
                                                        "run the code again and check if MAC address/name is entered" \
                                                        " correctly."
                assert ((device_name[-4:-2] == self.lastUsedAddress[-5:-3]) and
                        (device_name[-2:] == self.lastUsedAddress[-2:])), \
                    "MAC address does not match the expected value!"
            else:
                self.lastUsedAddress = device_addr
                assert ((device_name[-4:-2] == self.lastUsedAddress[-5:-3]) and
                        (device_name[-2:] == self.lastUsedAddress[-2:])), \
                    "MAC address does not match the expected value!"
        else:
            # No need to scan if we have the address
            self.lastUsedAddress = device_addr
            device_name = "Explore_"+str(device_addr[-5:-3])+str(device_addr[-2:])
            address_known = True

        service_matches = self.find_explore_service()

        assert service_matches, "SSP service for the device %s, with MAC address %s could not be found." \
            "restart the device and try again" %(device_name, self.lastUsedAddress)

        for services in service_matches:
            self.port = services["port"]
            self.name = services["name"]
            self.host = services["host"]
            # Checking if "Explore_ABCD" matches "XX:XX:XX:XX:AB:CD"
            if (device_name[-4:-2] == self.host[-5:-3])and(device_name[-2:] == self.host[-2:]):
                break

        assert((device_name[-4:-2] == self.host[-5:-3])and(device_name[-2:] == self.host[-2:])), \
            "MAC address does not match the expected value on the SSP service!!"

        print("Connecting to %s with address %s" % (self.name, self.host))

    def bt_connect(self):
        """Creates the socket
        """
        while True:
            try:
                self.socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
                self.socket.connect((self.host, self.port))
                break
            except bluetooth.BluetoothError as error:
                self.socket.close()
                print("Could not connect: ", error, "; Retrying in 2s...")
                time.sleep(2)
        return self.socket

    def reconnect(self):
        """
        tries to open the last bt socket, uses the last port and host. if after 1 minute the connection doesnt succeed,
        program will end
        """

        timeout = 1
        while timeout < 5:
            try:
                self.socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
                self.socket.connect((self.host, self.port))
                break
            except bluetooth.BluetoothError as error:
                print("Bluetooth Error: Probably timeout, attempting reconnect. Error: ", error)
                time.sleep(5)

            timeout += 1

        if timeout == 5:
            print("Device not found!")
            self.socket.close()
            return False

    def find_mac_addr(self, device_name):
        i = 0
        while i < 5:
            nearby_devices = bluetooth.discover_devices(lookup_names=True, flush_cache=True )
            for address, name in nearby_devices:
                if name == device_name:
                    self.lastUsedAddress = address
                    return True
            i += 1
            print("No device found with name: %s, searching again in 0,1 seconds" % device_name)
            time.sleep(0.1)
        return False

    def find_explore_service(self):
        uuid = "1101"  # Serial Port Profile (SPP) service
        i = 0
        while i < 5:
            service_matches = bluetooth.find_service(uuid=uuid, address=self.lastUsedAddress)
            if len(service_matches) > 0:
                return service_matches
            i += 1
        return False
