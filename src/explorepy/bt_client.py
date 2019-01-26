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

    def initBT(self):
        """
        lets the user choose a device
        For now, only one device can be connected to a computer at the same time, however it is possible to choose which
        device should be connected

        Also reserves a port
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

        uuid = "1101"  # Serial Port Profile (SPP) service
        service_matches = bluetooth.find_service(uuid=uuid, address=self.lastUsedAddress)

        if len(service_matches) == 0:
            print("Couldn't find the SampleServer service! Restart your device and run the code again")
            sys.exit(0)

        first_match = service_matches[0]
        self.port = first_match["port"]
        self.name = first_match["name"]
        self.host = first_match["host"]

        print("Connecting to \"%s\" on %s" % (self.name, self.host))

    def connect(self):
        """
        creates the socket

        Returns:

        """

        # Create the client socket

        time.sleep(10)
        while True:
            try:
                self.socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
                self.socket.connect((self.host, self.port))
                print("socket online!:", self.host, " : ", self.port)
                break
            except bluetooth.BluetoothError as error:
                self.socket.close()
                print("Could not connect: ", error, "; Retrying in 10s...")
                time.sleep(10)


    def reconnect(self):
        """
        tries to open the last bt socket, uses the last port and host. if after 1 minute the connection doesnt succeed,
        program will end
        Returns:

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

                time.sleep(10)
                pass
            if is_reconnected is True:
                timeout = 0
                break
            timeout += 1

        if timeout == 5:
            print("Device not found, exiting the program!")
            self.socket.close()
            return False


        # Create the client socket

        self.socket.connect((self.host, self.port))
        self.socket.settimeout(10)

        self.is_connected = True
        return self.is_connected




