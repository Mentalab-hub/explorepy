# -*- coding: utf-8 -*-
"""A module for bluetooth connection"""
import time
import exploresdk
import logging

from explorepy._exceptions import DeviceNotFoundError, InputError


class SDKBtClient:
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

    def connect(self):
        """Connect to the device and return the socket

        Returns:
            socket (bluetooth.socket)
        """

        if self.mac_address is None:
            self._find_mac_address()
        else:
            self.device_name = "Explore_" + str(self.mac_address[-5:-3]) + str(self.mac_address[-2:])

      #  self.bt_serial_port_manager = exploresdk.BTSerialPortBinding_Create(self.mac_address, 5)
       # self.bt_serial_port_manager.Connect()

        for _ in range(5):
            try:
                self.bt_serial_port_manager = exploresdk.BTSerialPortBinding_Create(self.mac_address, 5)
                return_code = self.bt_serial_port_manager.Connect()
                if return_code == 0:
                    self.is_connected = True
                    return
                else:
                    self.is_connected = False
                    print(return_code, "\nCould not connect; Retrying in 2s...")
                    time.sleep(2)

            except:
                self.is_connected = False
                print(return_code, "\nCould not connect; Retrying in 2s...")
                time.sleep(2)


##        for _ in range(5):
#
 #           return_code = self.bt_serial_port_manager.Connect()
#
       #     if return_code == 0:
      #          self.is_connected = True
     #           return
    #        else:
   #             self.is_connected = False
  #              print(return_code, "\nCould not connect; Retrying in 2s...")
 #               time.sleep(2)
#
    #    self.is_connected = False
   #     raise DeviceNotFoundError("Could not find the device! Please make sure"
  #                                " the device is on and in advertising mode.")
##

    def reconnect(self):
        """Reconnect to the last used bluetooth socket.

        This function reconnects to the the last bluetooth socket. If after 1 minute the connection doesn't succeed,
        program will end.
        """
        connection_error_code = self.bt_serial_port_manager.Connect()
        if(connection_error_code != 0):
            print("Connection failed")
        else:
            print("connection success")

        self.bt_serial_port_manager.Close()
        self.is_connected = False
        raise DeviceNotFoundError("Could not find the device! Please make sure the device is on and in"
                                  "advertising mode.")

    def disconnect(self):
        """Disconnect from the device"""
        self.bt_serial_port_manager.Close()
        self.is_connected = False

    def _find_mac_address(self):

        self.device_manager = exploresdk.ExploreSDK_Create()
        for _ in range(5):

            available_list = self.device_manager.PerformDeviceSearch()

            for bluetooth_device in available_list:
                if bluetooth_device.name == self.device_name:
                    self.mac_address = bluetooth_device.address
                    return

            print("No device found with the name: {}, searching again...".format(self.device_name))
            time.sleep(0.1)
        raise DeviceNotFoundError("No device found with the name: {}".format(self.device_name))


    def read(self, n_bytes):
        """Read n_bytes from the socket

            Args:
                n_bytes (int): number of bytes to be read

            Returns:
                list of bytes
        """

        time.sleep(.002)
        read_return_code = -100

        try:
            size, read_output = self.bt_serial_port_manager.Read(n_bytes)

            if size < n_bytes:
                raise ConnectionAbortedError('No data in bluetooth buffer.')

            current_length = len(read_output)

            actual_byte_data = read_output.encode('utf-8', errors='surrogateescape')

            return actual_byte_data

        except RuntimeError as error:
            print('Runtime Error occured while reading device data, please make sure that the device is on and in advertising mode')
        except OverflowError as error:
            print('Error Reading data from Bluetooth buffer, please make sure that the device is on and in advertising mode')
        except AssertionError as error:
            print('Error Reading data from Bluetooth buffer, please make sure that the device is on and in advertising mode')




    def send(self, data):
        """Send data to the device

        Args:
            data (bytearray): Data to be sent
        """
        print('-------------------------------------------------')
        print(type(data))
        string_data = data.decode('utf-8', errors='surrogateescape')

        self.bt_serial_port_manager.Write(data)



    @staticmethod
    def _check_mac_address(device_name, mac_address):
        return (device_name[-4:-2] == mac_address[-5:-3]) and (device_name[-2:] == mac_address[-2:])




