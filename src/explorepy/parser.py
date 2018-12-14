import numpy as np
import struct


class Parser:
    """Parser class for explore device"""
    def __init__(self, socket):
        """

        Args:
            socket:
        """
        self.socket = socket
        self.dt_int16 = np.dtype(np.int16).newbyteorder('<')
        self.dt_uint16 = np.dtype(np.uint16).newbyteorder('<')

    def parse_packet(self):
        pid = struct.unpack('B', self.socket.recv(1))[0]
        cnt = self.socket.recv(1)[0]
        payload = struct.unpack('<H', self.socket.recv(2))[0]
        timestamp = struct.unpack('<I', self.socket.recv(4))[0]
        bin_data = self.socket.recv(payload-8)
        fletcher = self.socket.recv(4)
        data = self._convert(pid, bin_data)
        return pid

    def _convert(self, pid, bin_data):
        if pid == 13:
            data = np.frombuffer(bin_data, dtype=self.dt_int16)
        elif pid == 19:
            temperature = np.frombuffer(bin_data[0], dtype=np.int8)
            light = (1000/4095) * np.frombuffer(bin_data[1:3], dtype=self.dt_uint16)
            battery = (16.8 / 6.8) * (1.8 / 2457) * np.frombuffer(bin_data[3:5], dtype=self.dt_uint16)
        elif pid == 27:
            pass  # TO DO: make sure the timestamp packet doesn't give an error
        elif pid == 144:
            data = np.frombuffer(bin_data, dtype=self.dt_int16)
