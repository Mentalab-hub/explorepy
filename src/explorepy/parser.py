import numpy as np
import struct


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

    def parse_packet(self, mode='print'):
        r"""
        Reads and parses a package from a file or socket
        Args:
            mode (str): logging mode {'print', None}

        Returns:

        """
        pid = struct.unpack('B', self.read(1))[0]
        cnt = self.read(1)[0]
        payload = struct.unpack('<H', self.read(2))[0]
        timestamp = struct.unpack('<I', self.read(4))[0]
        bin_data = self.read(payload - 8)
        fletcher = self.read(4)
        # TODO: Check fletcher
        data = self._convert(pid, bin_data, mode)
        return pid, timestamp, data

    def _convert(self, pid, bin_data, mode='print'):
        r"""
        Parses the packet
        Args:
            pid (int): Packet ID
            bin_data: Binary data
            mode: logging mode {'print', None}

        Returns:
            data (

        """
        data = None
        if pid == 13:
            data = np.copy(np.frombuffer(bin_data, dtype=self.dt_int16)).astype(np.float)
            data[0:3] = 0.061 * data[0:3]  # Unit [mg/LSB]
            data[3:6] = 8.750 * data[3:6]  # Unit [mdps/LSB]
            data[6:] = 1.52 * data[6:]  # Unit [mgauss/LSB]
            if mode == 'print':
                print("Accelerometer: ", data)

        elif pid == 19:  # Environment packet
            temperature = bin_data[0]
            light = (1000 / 4095) * np.frombuffer(bin_data[1:3], dtype=self.dt_uint16)  # Unit Lux
            battery = (16.8 / 6.8) * (1.8 / 2457) * np.frombuffer(bin_data[3:5], dtype=self.dt_uint16)  # Unit Volt
            if mode == 'print':
                print("Temperature: ", temperature, ", Light: ", light, ", Battery: ", battery)
            data = (temperature, light, battery)

        elif pid == 27:  # Host timestamp
            data = struct.unpack('<Q', bin_data)[0]
            if mode == 'print':
                print("Host timestamp:", data)

        elif pid == 111:  # Disconnect packet
            data = None

        elif pid == 99:  # Device info packet
            data = struct.unpack('<I', bin_data)[0]
            if mode == 'print':
                print("Firmware version:", data)

        elif pid == 144:  # 4 channel device
            data = self._bit24ToInt(bin_data)
            nChan = 5
            vref = 2.4
            nPacket = 33
            data = data.reshape((nPacket, nChan)).astype(np.float).T
            data[1:, :] = data[1:, :] * vref / ((2 ** 23) - 1) * 6. / 32.
            if mode == 'print':
                print("EEG data: ", data[1:, 32])

        elif pid == 146:  # 8 channel device + status (ADS1298)
            data = self._bit24ToInt(bin_data)
            nChan = 9
            vref = 2.4
            nPacket = -1
            data = data.reshape((nPacket, nChan)).astype(np.float).T
            data[1:, :] = data[0:, :] * vref / ((2 ** 23) - 1) * 6. / 32.
            if mode == 'print':
                print("EEG data: ", data[1:, -1])

        elif pid == 30:  # 8 channel device + status (ADS1299)
            data = self._bit24ToInt(bin_data)
            nChan = 9
            vref = 4.5
            nPacket = -1
            data = data.reshape((nPacket, nChan)).astype(np.float).T
            data[1:, :] = data[1:, :] * vref / ((2 ** 23) - 1) * 6. / 32.
            if mode == 'print':
                print("EEG data: ", data[1:, -1])

        elif pid == 62:  # 8 channel device (ADS1298)
            data = self._bit24ToInt(bin_data)
            nChan = 8
            vref = 4.5
            nPacket = -1
            data = data.reshape((nPacket, nChan)).astype(np.float).T
            data[0:, :] = data[0:, :] * vref / ((2 ** 23) - 1) * 6. / 32.
            if mode == 'print':
                print("EEG data: ", data[0:, -1])
        else:
            print("Unknown Packet ID:", pid)
            print("Length of the binary data:", len(bin_data))

        return data

    def _bit24ToInt(self, bin_data):
        r"""
        converts binary data to int32

        Args:
            bin_data (list): list of bytes with the structure of int24

        Returns:
            np.ndarray of int values
        """
        assert len(bin_data) % 3 == 0, "Packet length error!"
        return np.asarray([int.from_bytes(bin_data[x:x + 3],
                                          byteorder='little',
                                          signed=True) for x in range(0, len(bin_data), 3)])

    def read(self, n_bytes):
        r"""
        Read n_bytes from socket or file
        Args:
            n_bytes (int): number of bytes to be read

        Returns:
            list of bytes

        """
        if self.socket is not None:
            return self.socket.recv(n_bytes)
        else:
            return self.fid.read(n_bytes)
