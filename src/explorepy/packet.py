import numpy as np
import abc
import struct


class Packet:
    """An abstract base class for Explore packet"""
    __metadata__ = abc.ABCMeta

    def __init__(self, timestamp, payload):
        """
        Should get the packet id and the remaining packet bytes and convert it to the
        Args:
            payload (bytearray): a byte array including binary data and fletcher
        """
        self.timestamp = timestamp

    @abc.abstractmethod
    def __str__(self):
        pass

    @abc.abstractmethod
    def _convert(self, bin_data):
        """Read the binary data and convert it to real values"""
        pass

    @abc.abstractmethod
    def _check_fletcher(self, fletcher):
        """Checks if the fletcher is valid"""
        pass

    @abc.abstractmethod
    def __str__(self):
        """Print the data/info"""
        pass

    @staticmethod
    def int24to32(bin_data):
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


class EEG94(Packet):
    """EEG packet for 4 channel device"""
    def __init__(self, timestamp, payload):
        super().__init__(timestamp, payload)
        self._convert(payload[:-4])
        self._check_fletcher(payload[-4:])

    def _convert(self, bin_data):
        data = Packet.int24to32(bin_data)
        n_chan = 5
        v_ref = 2.4
        n_packet = 33
        data = data.reshape((n_packet, n_chan)).astype(np.float).T
        self.data = data[1:, :] * v_ref / ((2 ** 23) - 1) * 6. / 32.
        self.dataStatus = data[0, :]

    def _check_fletcher(self, fletcher):
        assert fletcher == b'\xaf\xbe\xad\xde', "Fletcher error!"

    def __str__(self):
        return "EEG: " + str(self.data[:, -1])

    def write_to_csv(self, csv_writer):
        tmpstmp = np.zeros([self.data.shape[1], 1])
        tmpstmp[:,:] = self.timestamp
        csv_writer.writerows(np.concatenate((tmpstmp, self.data.T), axis=1).tolist())


class EEG98(Packet):
    """EEG packet for 8 channel device"""
    def __init__(self, timestamp, payload):
        super().__init__(timestamp, payload)
        self._convert(payload[:-4])
        self._check_fletcher(payload[-4:])

    def _convert(self, bin_data):
        data = Packet.int24to32(bin_data)
        n_chan = 9
        v_ref = 2.4
        n_packet = -1
        data = data.reshape((n_packet, n_chan)).astype(np.float).T
        self.data = data[1:, :] * v_ref / ((2 ** 23) - 1) * 6. / 32.
        self.status = data[0, :]

    def _check_fletcher(self, fletcher):
        assert fletcher == b'\xaf\xbe\xad\xde', "Fletcher error!"

    def __str__(self):
        return "EEG: " + str(self.data[:, -1])

    def write_to_csv(self, csv_writer):
        tmpstmp = np.zeros([self.data.shape[1], 1])
        tmpstmp[:,:] = self.timestamp
        csv_writer.writerows(np.concatenate((tmpstmp, self.data.T), axis=1).tolist())


class EEG99s(Packet):
    """EEG packet for 8 channel device"""
    def __init__(self, timestamp, payload):
        super().__init__(timestamp, payload)
        self._convert(payload[:-4])
        self._check_fletcher(payload[-4:])

    def _convert(self, bin_data):
        data = Packet.int24to32(bin_data)
        n_chan = 9
        v_ref = 4.5
        n_packet = -1
        data = data.reshape((n_packet, n_chan)).astype(np.float).T
        self.data = data[1:, :] * v_ref / ((2 ** 23) - 1) * 6. / 32.
        self.status = data[0, :]

    def _check_fletcher(self, fletcher):
        assert fletcher == b'\xaf\xbe\xad\xde', "Fletcher error!"

    def __str__(self):
        return "EEG: " + str(self.data[:, -1])

    def write_to_csv(self, csv_writer):
        tmpstmp = np.zeros([self.data.shape[1], 1])
        tmpstmp[:,:] = self.timestamp
        csv_writer.writerows(np.concatenate((tmpstmp, self.data.T), axis=1).tolist())


class EEG99(Packet):
    """EEG packet for 8 channel device"""
    def __init__(self, timestamp, payload):
        super().__init__(timestamp, payload)
        self._convert(payload[:-4])
        self._check_fletcher(payload[-4:])

    def _convert(self, bin_data):
        data = Packet.int24to32(bin_data)
        n_chan = 8
        v_ref = 4.5
        n_packet = -1
        data = data.reshape((n_packet, n_chan)).astype(np.float).T
        self.data = data * v_ref / ((2 ** 23) - 1) * 6. / 32.

    def _check_fletcher(self, fletcher):
        assert fletcher == b'\xaf\xbe\xad\xde', "Fletcher error!"

    def __str__(self):
        return "EEG: " + str(self.data[:, -1])

    def write_to_csv(self, csv_writer):
        tmpstmp = np.zeros([self.data.shape[1], 1])
        tmpstmp[:,:] = self.timestamp
        csv_writer.writerows(np.concatenate((tmpstmp, self.data.T), axis = 1).tolist())


class Orientation(Packet):
    """Orientation data packet"""
    def __init__(self, timestamp, payload):
        super().__init__(timestamp, payload)
        self._convert(payload[:-4])
        self._check_fletcher(payload[-4:])

    def _convert(self, bin_data):
        data = np.copy(np.frombuffer(bin_data, dtype=np.dtype(np.int16).newbyteorder('<'))).astype(np.float)
        self.acc = 0.061 * data[0:3]  # Unit [mg/LSB]
        self.gyro = 8.750 * data[3:6]  # Unit [mdps/LSB]
        self.mag = 1.52 * data[6:]  # Unit [mgauss/LSB]

    def _check_fletcher(self, fletcher):
        assert fletcher == b'\xaf\xbe\xad\xde', "Fletcher error!"

    def __str__(self):
        return "Acc: " + str(self.acc) + "\tGyro: " + str(self.gyro) + "\tMag: " + str(self.mag)

    def write_to_csv(self, csv_writer):
        csv_writer.writerow([self.timestamp] + self.acc.tolist() + self.gyro.tolist() + self.mag.tolist())


class Environment(Packet):
    """Environment data packet"""
    def __init__(self, timestamp, payload):
        super().__init__(timestamp, payload)
        self._convert(payload[:-4])
        self._check_fletcher(payload[-4:])

    def _convert(self, bin_data):
        self.temperature = bin_data[0]
        self.light = (1000 / 4095) * np.frombuffer(bin_data[1:3], dtype=np.dtype(np.uint16).newbyteorder('<'))  # Unit Lux
        self.battery = (16.8 / 6.8) * (1.8 / 2457) * np.frombuffer(bin_data[3:5], dtype=np.dtype(np.uint16).newbyteorder('<'))  # Unit Volt

    def _check_fletcher(self, fletcher):
        assert fletcher == b'\xaf\xbe\xad\xde', "Fletcher error!"

    def __str__(self):
        return "Temperature: " + str(self.temperature) + "\tLight: " + str(self.light) + "\tBattery: " + str(self.battery)


class TimeStamp(Packet):
    """Time stamp data packet"""
    def __init__(self, timestamp, payload):
        super().__init__(timestamp, payload)
        self._convert(payload[:-4])
        self._check_fletcher(payload[-4:])

    def _convert(self, bin_data):
        self.hostTimeStamp = np.frombuffer(bin_data, dtype=np.dtype(np.uint64).newbyteorder('<'))

    def _check_fletcher(self, fletcher):
        assert fletcher == b'\xff\xff\xff\xff', "Fletcher error!"

    def __str__(self):
        return "Host timestamp: " + str(self.hostTimeStamp)


class Disconnect(Packet):
    """Disconnect packet"""
    def __init__(self, timestamp, payload):
        super().__init__(timestamp, payload)
        self._check_fletcher(payload)

    def _convert(self, bin_data):
        """Disconnect packet has no data"""
        pass

    def _check_fletcher(self, fletcher):
        assert fletcher == b'\xaf\xbe\xad\xde', "Fletcher error!"

    def __str__(self):
        return "Device has been disconnected!"


class DeviceInfo(Packet):
    """Device information packet"""
    def __init__(self, timestamp, payload):
        super().__init__(timestamp, payload)
        self._convert(payload[:-4])
        self._check_fletcher(payload[-4:])

    def _convert(self, bin_data):
        self.firmware_version = np.frombuffer(bin_data, dtype=np.dtype(np.uint32).newbyteorder('<'))

    def _check_fletcher(self, fletcher):
        assert fletcher == b'\xaf\xbe\xad\xde', "Fletcher error!"

    def __str__(self):
        return "Firmware version: " + str(self.firmware_version)
