# -*- coding: utf-8 -*-
import numpy as np
import struct
from explorepy.packet import PACKET_ID, PACKET_CLASS_DICT, TimeStamp, EEG, Environment, CommandRCV, CommandStatus,\
                                Orientation, DeviceInfo, Disconnect, MarkerEvent, CalibrationInfo
from explorepy.filters import Filter
import copy

def generate_packet(pid, timestamp, bin_data):
    """Generates the packets according to the pid

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
    def __init__(self, bp_freq=None, notch_freq=50, socket=None, fid=None):
        """Parser class for explore device

        Args:
            socket (BluetoothSocket): Bluetooth Socket (Should be None if fid is provided)
            fid (file object): File object for reading data (Should be None if socket is provided)
            bp_freq (tuple): Tuple of cut-off frequencies of bandpass filter (low cut-off frequency, high cut-off frequency)
            notch_freq (int): Notch filter frequency (50 or 60 Hz)
        """
        self.socket = socket
        self.fid = fid
        self.dt_int16 = np.dtype(np.int16).newbyteorder('<')
        self.dt_uint16 = np.dtype(np.uint16).newbyteorder('<')
        self.time_offset = None
        if bp_freq is not None:
            assert bp_freq[0] < bp_freq[1], "High cut-off frequency must be larger than low cut-off frequency"
            self.bp_freq = bp_freq
            self.apply_bp_filter = True
        else:
            self.apply_bp_filter = False
            self.bp_freq = (0, 100)  # dummy values
        self.notch_freq = notch_freq
        self.firmware_version = None
        self.filter = None
        if self.apply_bp_filter or notch_freq:
            # Initialize filters
            self.filter = Filter(l_freq=self.bp_freq[0], h_freq=self.bp_freq[1], line_freq=notch_freq)

        self.imp_calib_info = {}

    def parse_packet(self, mode="print", csv_files=None, outlets=None, dashboard=None):
        """Reads and parses a package from a file or socket

        Args:
            mode (str): logging mode {'print', 'record', 'lsl', 'visualize', None}
            csv_files (tuple): Tuple of csv file objects (EEG_csv_file, ORN_csv_file, Marker_csv_file)
            outlets (tuple): Tuple of lsl StreamOutlet (orientation_outlet, EEG_outlet, marker_outlet)
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
            elif isinstance(packet, MarkerEvent):
                packet.write_to_csv(csv_files[2])

        elif mode == "lsl":
            if isinstance(packet, Orientation):
                packet.push_to_lsl(outlets[0])
            elif isinstance(packet, EEG):
                packet.push_to_lsl(outlets[1])
            elif isinstance(packet, MarkerEvent):
                packet.push_to_lsl(outlets[2])

        elif mode == "visualize":
            if isinstance(packet, EEG):
                if self.notch_freq:
                    packet.apply_notch_filter(exg_filter=self.filter)
                if self.apply_bp_filter:
                    packet.apply_bp_filter(exg_filter=self.filter)
            packet.push_to_dashboard(dashboard)

        elif mode == "listen":
            if isinstance(packet, CommandRCV):
                print(packet)
            elif isinstance(packet, CommandStatus):
                print(packet)
            elif isinstance(packet, CalibrationInfo):
                print(packet)
                
        elif mode == "debug":
            if isinstance(packet, EEG):
                print(packet)
        
        elif mode == "impedance":
            if isinstance(packet, EEG):
                if self.notch_freq:
                    packet.apply_notch_filter(exg_filter=self.filter)
                if self.apply_bp_filter:
                    temp_packet = copy.deepcopy(packet)
                    temp_packet.apply_bp_filter_test(exg_filter=self.filter)
                    mag = np.ptp(temp_packet.data, axis=1)
                    self.imp_calib_info['noise_level'] = mag
                    packet.apply_bp_filter(exg_filter=self.filter)
                packet.push_to_imp_dashboard(dashboard, self.imp_calib_info)
            elif isinstance(packet, Environment):
                packet.push_to_dashboard(dashboard)
    def read(self, n_bytes):
        """Read n_bytes from socket or file

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

    def send_msg(self, msg):
        """
         tries to send a message through socket.
         """
        if msg is None:
            ts_packet = TimeStamp()
            ts_packet.translate()
            msg = ts_packet.raw_data

        if self.socket is not None:
            self.socket.send(msg)
        else:
            raise ValueError("Cannot send the msg")
        return True
