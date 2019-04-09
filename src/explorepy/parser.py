import numpy as np
import struct
from explorepy.packet import *
from explorepy.filters import Filter

LOW_CUTOFF_FREQ = .5   # Hz
HIGH_CUTOFF_FREQ = 40  # Hz

ORN_ID = 13
ENV_ID = 19
TS_ID = 27
DISCONNECT_ID = 111
INFO_ID = 99
EEG94_ID = 144
EEG98_ID = 146
EEG99S_ID = 30
EEG99_ID = 62
EEG94R_ID = 208
EEG98R_ID = 210

PACKET_CLASS_DICT = {
    ORN_ID: Orientation,
    ENV_ID: Environment,
    TS_ID: TimeStamp,
    DISCONNECT_ID: Disconnect,
    INFO_ID: DeviceInfo,
    EEG94_ID: EEG94,
    EEG98_ID: EEG98,
    EEG99S_ID: EEG99s,
    EEG99_ID: EEG99s,
    EEG94R_ID: EEG94_ID,
    EEG98R_ID: EEG98
}


def generate_packet(pid, timestamp, bin_data):
    r"""Generates the packets according to the pid

    Args:
        pid (int): Packet ID
        timestamp (int): Timestamp
        bin_data: Binary dat

    Returns:
        Packet
    """

    if pid in PACKET_CLASS_DICT:
        packet = PACKET_CLASS_DICT[pid](timestamp, bin_data)
    else:
        print("Unknown Packet ID:" + str(pid))
        print("Length of the binary data:", len(bin_data))
        packet = None
    return packet


class Parser:
    def __init__(self, socket=None, fid=None, apply_filter=False):
        r"""Parser class for explore device

        Args:
            socket (BluetoothSocket): Bluetooth Socket (Should be None if fid is provided)
            fid (file object): File object for reading data (Should be None if socket is provided)
        """
        self.socket = socket
        self.fid = fid
        self.dt_int16 = np.dtype(np.int16).newbyteorder('<')
        self.dt_uint16 = np.dtype(np.uint16).newbyteorder('<')
        self.time_offset = None
        self.apply_filter = apply_filter

        self.firmware_version = None
        self.filter = None
        if apply_filter:
            # Initialize a bandpass filter
            self.filter = Filter(l_freq=LOW_CUTOFF_FREQ, h_freq=HIGH_CUTOFF_FREQ)

    def parse_packet(self, mode="print", csv_files=None, outlets=None, dashboard=None):
        r"""Reads and parses a package from a file or socket

        Args:
            mode (str): logging mode {'print', 'record', 'lsl', 'visualize', None}
            csv_files (tuple): Tuple of csv file objects (EEG_csv_file, ORN_csv_file)
            outlets (tuple): Tuple of lsl StreamOutlet (orientation_outlet, EEG_outlet
            dashboard (Dashboard): Dashboard object for visualization
        Returns:
            packet object
        """
        pid = struct.unpack('B', self.read(1))[0]
        cnt = self.read(1)[0]
        payload = struct.unpack('<H', self.read(2))[0]
        timestamp = struct.unpack('<I', self.read(4))[0]

        # Timestamp conversion
        if self.time_offset is None:
            self.time_offset = timestamp
            timestamp = 0
        else:
            timestamp = (timestamp - self.time_offset) * .0001  # Timestamp unit is .1 ms

        payload_data = self.read(payload - 4)
        packet = generate_packet(pid, timestamp, payload_data)

        if isinstance(packet, DeviceInfo):
            self.firmware_version = packet.firmware_version
        if mode == "print":
            print(packet)

        elif mode == "record":
            assert isinstance(csv_files, tuple), "Invalid csv writer objects!"
            if isinstance(packet, Orientation):
                packet.write_to_csv(csv_files[1])
            elif isinstance(packet, EEG):
                packet.write_to_csv(csv_files[0])

        elif mode == "lsl":
            if isinstance(packet, Orientation):
                packet.push_to_lsl(outlets[0])
            elif isinstance(packet, EEG):
                if self.apply_filter:
                    packet.apply_filter(filter=self.filter)
                packet.push_to_lsl(outlets[1])

        elif mode == "visualize":
            if isinstance(packet, EEG):
                if self.apply_filter:
                    packet.apply_filter(filter=self.filter)
            packet.push_to_dashboard(dashboard)

        return packet

    def read(self, n_bytes):
        r"""Read n_bytes from socket or file

        Args:
            n_bytes (int): number of bytes to be read

        Returns:
            list of bytes
        """
        if self.socket is not None:
            byte_data = self.socket.recv(n_bytes)
        elif not self.fid.closed:
            byte_data = self.fid.read(n_bytes)
        else:
            raise ValueError("File has been closed unexpectedly!")
        if len(byte_data) != n_bytes:
            raise ValueError("Number of received bytes is less than expected")
            # TODO: Create a specific exception for this case
        return byte_data
