import ExploreParser
import bluetooth
import sys


class Bt_Explore:
    connected = False
    pid = 0
    lastUsedAddress = 0
    sock = 0
    def connect(self):
        explore_devices = []
        nearby_devices = bluetooth.discover_devices(lookup_names=True)
        counter = 0
        for addr, name in nearby_devices:
            if name == "Explore":
                print("  %s - %s" % (addr, name))
                counter += 1
                explore_devices.append([addr, name])

        if counter == 1:
            print("Device found: %s - %s" % (explore_devices[0][0], explore_devices[0][1]))
            Bt_Explore.lastUsedAddress = explore_devices[0][0]

        elif counter > 1:
            print("Multiple Devices found: ")
            k = 0
            for addr, name in explore_devices:
                print(" [%i]: %s - %s" % (k, addr, name))
                k += 1

            selector = input('Please choose a device by entering the number in front of the MAC address: ')
            Bt_Explore.lastUsedAddress = explore_devices[0][int(selector)]

        elif counter == 0:
            print("no devices found. Restart your device and run the code again")
            exit()

        # search for the SampleServer service
        uuid = "1101"
        service_matches = bluetooth.find_service(uuid=uuid, address=Bt_Explore.lastUsedAddress)

        if len(service_matches) == 0:
            print("couldn't find the SampleServer service =(")
            sys.exit(0)

        first_match = service_matches[0]
        port = first_match["port"]
        name = first_match["name"]
        host = first_match["host"]

        print("connecting to \"%s\" on %s" % (name, host))

        # Create the client socket
        Bt_Explore.sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
        Bt_Explore.sock.connect((host, port))



    def reconnect(self):

        timeout = 1
        Reconnect = False
        while timeout < 5:
            print("scanning")
            nearby_devices = bluetooth.discover_devices(duration=15, lookup_names=True)
            for addr, name in nearby_devices:
                if addr == Bt_Explore.lastUsedAddress:
                    print("Former device found, attempting to reconnect")
                    Reconnect = True


            if Reconnect is True:
                timeout = 0
                break
            timeout += 1

        if timeout == 5:
            print("Device not found, shutting down")
            exit()

        # search for the SampleServer service
        uuid = "1101"
        service_matches = bluetooth.find_service(uuid=uuid, address=Bt_Explore.lastUsedAddress)
        if len(service_matches) == 0:
            print("couldn't find the SampleServer service =(")
            sys.exit(0)
        first_match = service_matches[0]
        port = first_match["port"]
        name = first_match["name"]
        host = first_match["host"]
        print("connecting to \"%s\" on %s" % (name, host))

        # Create the client socket
        Bt_Explore.sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
        Bt_Explore.sock.connect((host, port))

        Bt_Explore.connected = True



    def accquire(self):
        exp_parser = ExploreParser.ExploreParser(socket=Bt_Explore.sock)
        try:
            while True:
                Bt_Explore.pid = exp_parser.parse_packet()
                if len(Bt_Explore.pid) == 0:
                    Bt_Explore.connected = False
                    break
                print("package ID: [%i]" % Bt_Explore.pid)
        except ValueError:
            # If value error happens, scan again for devices and try to reconnect (see reconnect function)
            print("Disconnected, scanning for last connected device")
            Bt_Explore.connected = False
            Bt_Explore.sock.close()
