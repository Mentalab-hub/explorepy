import numpy as np
import struct
from .packet import *


def generate_packet(pid, timestamp, bin_data):
    r"""
    Generates the packets according to the pid
    Args:
        pid (int): Packet ID
        timestamp (int): Timestamp
        bin_data: Binary dat

    Returns:
        Packet
    """
    if pid == 13:  # Orientation
        packet = Orientation(timestamp, bin_data)
    elif pid == 19:  # Environment packet
        packet = Environment(timestamp, bin_data)
    elif pid == 27:  # Host timestamp
        packet = TimeStamp(timestamp, bin_data)
    elif pid == 111:  # Disconnect packet
        packet = Disconnect(timestamp, bin_data)
    elif pid == 99:  # Device info packet
        packet = DeviceInfo(timestamp, bin_data)
    elif pid == 144:  # 4 channel device (EEG94)
        packet = EEG94(timestamp, bin_data)
    elif pid == 146:  # 8 channel device + status (ADS1298 - EEG98)
        packet = EEG98(timestamp, bin_data)
    elif pid == 30:  # 8 channel device + status (ADS1299 - EEG99s)
        packet = EEG99s(timestamp, bin_data)
    elif pid == 62:  # 8 channel device (ADS1298 - EEG99)
        packet = EEG99(timestamp, bin_data)

    else:
        print("Unknown Packet ID:" + str(pid))
        print("Length of the binary data:", len(bin_data))
        packet = Reconnect(timestamp, bin_data)

    return packet


class Parser:
    def __init__(self, socket=None, fid=None):
        r"""
        Parser class for explore device
        Args:
            socket (BluetoothSocket): Bluetooth Socket (Should be None if fid is provided)
            fid (file object): File object for reading data (Should be None if socket is provided)
        """
        self.socket = socket
        self.fid = fid
        self.dt_int16 = np.dtype(np.int16).newbyteorder('<')
        self.dt_uint16 = np.dtype(np.uint16).newbyteorder('<')

    def parse_packet(self, mode="print", csv_files=None):
        r"""
        Reads and parses a package from a file or socket
        Args:
            mode (str): logging mode {'print', None}
            csv_files (tuple): Tuple of csv file objects (EEG_csv_file, ORN_csv_file)
        Returns:

        """
        pid = struct.unpack('B', self.read(1))[0]
        cnt = self.read(1)[0]
        payload = struct.unpack('<H', self.read(2))[0]
        timestamp = struct.unpack('<I', self.read(4))[0]
        payload_data = self.read(payload - 4)
        packet = generate_packet(pid, timestamp, payload_data)
        if mode == "print":
            print(packet)
        elif mode == "record":
            if isinstance(packet, Orientation):
                packet.write_to_csv(csv_files[1])

            elif isinstance(packet, EEG94) or isinstance(packet, EEG98) or isinstance(packet, EEG99s) or isinstance(
                packet, EEG99):
                packet.write_to_csv(csv_files[0])

        return packet

    def read(self, n_bytes):
        r"""
        Read n_bytes from socket or file
        Args:
            n_bytes (int): number of bytes to be read

        Returns:
            list of bytes

        """
        if self.socket is not None:
            byte_data = self.socket.recv(n_bytes)
        else:
            byte_data = self.fid.read(n_bytes)
        if len(byte_data) != n_bytes:
            raise ValueError("Number of received bytes is less than expected")
            # TODO: Create a specific exception for this case
        return byte_data
