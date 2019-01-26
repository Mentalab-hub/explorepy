import bluetooth, subprocess
import sys


class BtClient:
    """ Responsible for Connecting and reconnecting explore devices via bluetooth"""

    def __init__(self):
        self.is_connected = False
        self.lastUsedAddress = None
        self.socket = None

    def connect(self):
        """
        Checks for available explore devices and establishes connection depending on number of devices
        For now, only one device can be connected to a computer at the same time, however it is possible to choose which
        device should be connected

        Returns:

        """
        explore_devices = []
        print("Searching for nearby devices...")
        nearby_devices = bluetooth.discover_devices(lookup_names=True)
        counter = 0
        for address, name in nearby_devices:
            if "Explore" in name:
                counter += 1
                explore_devices.append([address, name])
            print(name)

        if counter == 1:
            print("Device found: %s - %s" % (explore_devices[0][0], explore_devices[0][1]))
            self.lastUsedAddress = explore_devices[0][0]

        elif counter > 1:
            print("Multiple Devices found: ")
            k = 0
            for address, name in explore_devices:
                print(" [%i]: %s - %s" % (k, address, name))
                k += 1

            selector = input('Please choose a device by entering the number in front of the MAC address: ')
            self.lastUsedAddress = explore_devices[int(selector)][0]

        elif counter == 0:
            print("No devices found. Restart your device and run the code again.")
            sys.exit(0)

        uuid = "1101"   # Serial Port Profile (SPP) service
        service_matches = bluetooth.find_service(uuid=uuid, address=self.lastUsedAddress)

        if len(service_matches) == 0:
            print("Couldn't find the SampleServer service! Restart your device and run the code again")
            sys.exit(0)

        first_match = service_matches[0]
        port = first_match["port"]
        name = first_match["name"]
        host = first_match["host"]

        print("Connecting to \"%s\" on %s" % (name, host))

        # Create the client socket

        self.socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
        self.socket.connect((host, port))
        self.socket.settimeout(10)

    def reconnect(self):
        """
        Scans for the last connected MAC adress. If it is available it tries to reconnect, otherwise the program will s
        stop and give out a timeout error

        Returns:

        """

        timeout = 1
        is_reconnected = False
        while timeout < 5:
            print("scanning")
            nearby_devices = bluetooth.discover_devices(duration=10, lookup_names=True)
            for address, name in nearby_devices:
                if address == self.lastUsedAddress:
                    print("Former device found, attempting to reconnect")
                    is_reconnected = True

            if is_reconnected is True:
                timeout = 0
                break
            timeout += 1

        if timeout == 5:
            print("Device not found, exiting the program!")
            self.socket.close()
            return False

        uuid = "1101"   # Serial Port Profile (SPP) service
        service_matches = bluetooth.find_service(uuid=uuid, address=self.lastUsedAddress)

        if len(service_matches) == 0:
            print("Couldn't find the SampleServer service =(")
            sys.exit(0)

        first_match = service_matches[0]
        port = first_match["port"]
        name = first_match["name"]
        host = first_match["host"]
        print("Connecting to \"%s\" on %s" % (name, host))

        # Create the client socket

        self.socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
        self.socket.connect((host, port))
        self.socket.settimeout(10)

        self.is_connected = True
        return self.is_connected




