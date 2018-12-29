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

    def _convert(self, pid, bin_data, mode='print'):
        data = None
        if pid == 13:
            data = np.frombuffer(bin_data, dtype=self.dt_int16)
            if mode == 'print':
                print("Accelerometer: ", data)

        elif pid == 19:
            temperature = bin_data[0]   # np.frombuffer(bin_data[0], dtype=np.int8)
            light = (1000/4095) * np.frombuffer(bin_data[1:3], dtype=self.dt_uint16)
            battery = (16.8 / 6.8) * (1.8 / 2457) * np.frombuffer(bin_data[3:5], dtype=self.dt_uint16)
            if mode == 'print':
                print(f"Temperature: {temperature}, Light: {light}, Battery:{battery}")
            data = (temperature, light, battery)

        elif pid == 27:
            pass  # TO DO: make sure the timestamp packet doesn't give an error

        elif pid == 144:
            data = np.asarray([int.from_bytes(bin_data[x:x+3],
                                              byteorder='little', signed=True) for x in range(0, len(bin_data), 3)])
            nChan = 5
            vref = 2.4
            nPacket = 33
            data = data.reshape((nChan, nPacket))
            eeg = data[1:, :].astype(np.float) * vref / (2 ^ 23 - 1) * 6/32
            if mode == 'print':
                print("EEG data: ", eeg[:, 32])

        return data
