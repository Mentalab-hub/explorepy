import bluetooth, subprocess
import sys
import time


class BtClient:
    """ Responsible for Connecting and reconnecting explore devices via bluetooth"""

    def __init__(self):
        self.is_connected = False
        self.lastUsedAddress = None
        self.socket = None
        self.host = None
        self.port = None
        self.name = None

    def initBT(self, device_name=None, device_addr=None):
        """
        Initialize Bluetooth connection

        Args:
            device_name(str): Name of the device
            device_addr(str): Devices MAC address
        """
        assert (device_addr is not None) or (device_name is not None), "Missing name or address"

        if device_addr is None:
            nearby_devices = bluetooth.discover_devices(lookup_names=True)
            for address, name in nearby_devices:
                if device_name == name:
                    self.lastUsedAddress = address
                    break
                else:
                    print("The Device with the given Name has not been found. Please check if device is turned on and paired"
                          " or check the entered name for typos")
                    sys.exit()
        else:
         #No need to scan if we have the address
            self.lastUsedAddress = device_addr



        uuid = "1101"  # Serial Port Profile (SPP) service
        service_matches = bluetooth.find_service(uuid=uuid, address=self.lastUsedAddress)
        assert len(service_matches) > 0, "Couldn't find the Device! Restart your device and run the code again and check if MAC address" \
                                         "is entered correctly."

        first_match = service_matches[0]
        self.port = first_match["port"]
        self.name = first_match["name"]
        self.host = first_match["host"]

        print("Connecting to serial port on %s" % self.host)

    def bt_connect(self):
        """Creates the socket
        """
        while True:
            try:
                socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
                socket.connect((self.host, self.port))
                break
            except bluetooth.BluetoothError as error:
                socket.close()
                print("Could not connect: ", error, "; Retrying in 2s...")
                time.sleep(2)
        return socket

    def reconnect(self):
        """
        tries to open the last bt socket, uses the last port and host. if after 1 minute the connection doesnt succeed,
        program will end
        """

        timeout = 1
        is_reconnected = False
        while timeout < 5:
            try:
                socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
                socket.connect((self.host, self.port))
                break;
            except bluetooth.BluetoothError as error:
                print("Bluetooth Error: Probably timeout, attempting reconnect. Error: ", error)
                time.sleep(5)
                pass
            if is_reconnected is True:
                timeout = 0
                break
            timeout += 1

        if timeout == 5:
            print("Device not found, exiting the program!")
            self.socket.close()
            return False
