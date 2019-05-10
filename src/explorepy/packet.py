# -*- coding: utf-8 -*-
import numpy as np
import abc
import struct
from functools import partial


class Packet:
    """An abstract base class for Explore packet"""
    __metadata__ = abc.ABCMeta

    def __init__(self, timestamp, payload):
        """
        Gets the timestamp and payload and initializes the packet object

        Args:
            payload (bytearray): a byte array including binary data and fletcher
        """
        self.timestamp = timestamp

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
        """
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

    @abc.abstractmethod
    def push_to_dashboard(self, dashboard):
        pass


class EEG(Packet):

    @abc.abstractmethod
    def write_to_csv(self, csv_writer):
        """
        Write EEG data to csv file

        Args:
            csv_writer(csv_writer): csv writer object

        """
        pass

    def apply_bp_filter(self, exg_filter):
        """Bandpass filtering of ExG data

        Args:
        exg_filter: Filter object
        """
        self.data = exg_filter.apply_bp_filter(self.data)

    def apply_notch_filter(self, exg_filter):
        """Band_stop filtering of ExG data

        Args:
            exg_filter: Filter object

        """
        self.data = exg_filter.apply_notch_filter(self.data)

    def push_to_lsl(self, outlet):
        """Push data to lsl socket

        Args:
            outlet (lsl.StreamOutlet): lsl stream outlet
        """

        for sample in self.data.T:
            outlet.push_sample(sample.tolist())

    def push_to_dashboard(self, dashboard):
        n_sample = self.data.shape[1]
        time_vector = np.linspace(self.timestamp, self.timestamp+(n_sample-1)/250., n_sample)
        dashboard.doc.add_next_tick_callback(partial(dashboard.update_exg, time_vector=time_vector, ExG=self.data))


class EEG94(EEG):
    """EEG packet for 4 channel device"""
    def __init__(self, timestamp, payload):
        super().__init__(timestamp, payload)
        self._convert(payload[:-4])
        self._check_fletcher(payload[-4:])

    def _convert(self, bin_data):
        data = Packet.int24to32(bin_data)
        n_chan = -1
        v_ref = 2.4
        n_packet = 33
        data = data.reshape((n_packet, n_chan)).astype(np.float).T
        self.data = data[1:, :] * v_ref / ((2 ** 23) - 1) / 6.
        self.dataStatus = data[0, :]

    def _check_fletcher(self, fletcher):
        assert fletcher == b'\xaf\xbe\xad\xde', "Fletcher error!"

    def __str__(self):
        return "EEG: " + str(self.data[:, -1])

    def write_to_csv(self, csv_writer):
        tmpstmp = np.zeros([self.data.shape[1], 1])
        tmpstmp[:,:] = self.timestamp
        csv_writer.writerows(np.concatenate((tmpstmp, self.data.T), axis=1).tolist())


class EEG98(EEG):
    """EEG packet for 8 channel device"""
    def __init__(self, timestamp, payload):
        super().__init__(timestamp, payload)
        self._convert(payload[:-4])
        self._check_fletcher(payload[-4:])

    def _convert(self, bin_data):
        data = Packet.int24to32(bin_data)
        n_chan = -1
        v_ref = 2.4
        n_packet = -1
        data = data.reshape((n_packet, n_chan)).astype(np.float).T
        self.data = data[1:, :] * v_ref / ((2 ** 23) - 1) /6.
        self.status = data[0, :]

    def _check_fletcher(self, fletcher):
        assert fletcher == b'\xaf\xbe\xad\xde', "Fletcher error!"

    def __str__(self):
        return "EEG: " + str(self.data[:, -1])

    def write_to_csv(self, csv_writer):
        tmpstmp = np.zeros([self.data.shape[1], 1])
        tmpstmp[:,:] = self.timestamp
        csv_writer.writerows(np.concatenate((tmpstmp, self.data.T), axis=1).tolist())


class EEG99s(EEG):
    """EEG packet for 8 channel device"""
    def __init__(self, timestamp, payload):
        super().__init__(timestamp, payload)
        self._convert(payload[:-4])
        self._check_fletcher(payload[-4:])

    def _convert(self, bin_data):
        data = Packet.int24to32(bin_data)
        n_chan = -1
        v_ref = 4.5
        n_packet = -1
        data = data.reshape((n_packet, n_chan)).astype(np.float).T
        self.data = data[1:, :] * v_ref / ((2 ** 23) - 1) /6.
        self.status = data[0, :]

    def _check_fletcher(self, fletcher):
        assert fletcher == b'\xaf\xbe\xad\xde', "Fletcher error!"

    def __str__(self):
        return "EEG: " + str(self.data[:, -1])

    def write_to_csv(self, csv_writer):
        tmpstmp = np.zeros([self.data.shape[1], 1])
        for i in range(0,16):
            tmpstmp[i,:] = (self.timestamp-0.064+i*40)/10000

        csv_writer.writerows(np.concatenate((tmpstmp, self.data.T), axis=1).tolist())


class EEG99(EEG):
    """EEG packet for 8 channel device"""
    def __init__(self, timestamp, payload):
        super().__init__(timestamp, payload)
        self._convert(payload[:-4])
        self._check_fletcher(payload[-4:])

    def _convert(self, bin_data):
        data = Packet.int24to32(bin_data)
        n_chan = -1
        v_ref = 4.5
        n_packet = -1
        data = data.reshape((n_packet, n_chan)).astype(np.float).T
        self.data = data * v_ref / ((2 ** 23) - 1) /6.

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

    def push_to_lsl(self, outlet):
        outlet.push_sample(self.acc.tolist() + self.gyro.tolist() + self.mag.tolist())

    def push_to_dashboard(self, dashboard):
        data = self.acc.tolist() + self.gyro.tolist() + self.mag.tolist()
        dashboard.doc.add_next_tick_callback(partial(dashboard.update_orn, timestamp=self.timestamp, orn_data=data))


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
        self.battery_percentage = self._volt_to_percent(self.battery)

    def _check_fletcher(self, fletcher):
        assert fletcher == b'\xaf\xbe\xad\xde', "Fletcher error!"

    def __str__(self):
        return "Temperature: " + str(self.temperature) + "\tLight: " + str(self.light) + "\tBattery: " + str(self.battery)

    def push_to_dashboard(self, dashboard):
        data = {'battery': [self.battery_percentage],
                'temperature': [self.temperature],
                'light': [self.light]}
        dashboard.doc.add_next_tick_callback(partial(dashboard.update_info, new=data))

    @staticmethod
    def _volt_to_percent(voltage):
        if voltage < 3.1:
            percentage = 1
        elif voltage < 3.5:
            percentage = 1 + (voltage - 3.1) / .4 * 10
        elif voltage < 3.8:
            percentage = 10 + (voltage-3.5)/.3 * 40
        elif voltage < 3.9:
            percentage = 40 + (voltage-3.8)/.1 * 20
        elif voltage < 4.:
            percentage = 60 + (voltage-3.9)/.1 * 15
        elif voltage < 4.1:
            percentage = 75 + (voltage-4.)/.1 * 15
        elif voltage < 4.2:
            percentage = 90 + (voltage - 4.1)/.1 * 10
        elif voltage > 4.2:
            percentage = 100

        percentage = int(percentage)
        return percentage


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
        super(DeviceInfo, self).__init__(timestamp, payload)
        self._convert(payload[:-4])
        self._check_fletcher(payload[-4:])

    def _convert(self, bin_data):
        fw_num = np.frombuffer(bin_data, dtype=np.dtype(np.uint32).newbyteorder('<'))[0]
        self.firmware_version = '.'.join([char for char in str(fw_num)])

    def _check_fletcher(self, fletcher):
        assert fletcher == b'\xaf\xbe\xad\xde', "Fletcher error!"

    def __str__(self):
        return "Firmware version: " + self.firmware_version

    def push_to_dashboard(self, dashboard):
        data = {'firmware_version': [self.firmware_version]}
        dashboard.doc.add_next_tick_callback(partial(dashboard.update_info, new=data))
