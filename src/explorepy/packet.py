# -*- coding: utf-8 -*-
"""This module contains all packet classes of Mentalab Explore device"""
import abc
from enum import IntEnum
import logging

import numpy as np

from explorepy._exceptions import FletcherError

logger = logging.getLogger(__name__)


class PACKET_ID(IntEnum):
    """Packet ID enum"""
    ORN = 13
    ENV = 19
    TS = 27
    DISCONNECT = 111
    INFO = 99
    EEG94 = 144
    EEG98 = 146
    EEG99S = 30
    EEG99 = 62
    EEG94R = 208
    EEG98R = 210
    CMDRCV = 192
    CMDSTAT = 193
    MARKER = 194
    CALIBINFO = 195
    TRIGGER_OUT = 28
    TRIGGER_IN = 29


EXG_UNIT = 1e-6


class Packet:
    """An abstract base class for Explore packet"""
    __metadata__ = abc.ABCMeta

    def __init__(self, timestamp, payload):
        """Gets the timestamp and payload and initializes the packet object

        Args:
            payload (bytearray): a byte array including binary data and fletcher
        """
        self.timestamp = timestamp

    @abc.abstractmethod
    def _convert(self, bin_data):
        """Read the binary data and convert it to real values"""

    @abc.abstractmethod
    def _check_fletcher(self, fletcher):
        """Checks if the fletcher is valid"""

    @abc.abstractmethod
    def __str__(self):
        """Print the data/info"""

    @staticmethod
    def int24to32(bin_data):
        """Converts binary data to int32

        Args:
            bin_data (list): list of bytes with the structure of int24

        Returns:
            np.ndarray of int values
        """
        assert len(bin_data) % 3 == 0, "Packet length error!"
        return np.asarray([int.from_bytes(bin_data[x:x + 3],
                                          byteorder='little',
                                          signed=True) for x in range(0, len(bin_data), 3)])


class EEG(Packet):
    """EEG packet class"""
    __metadata__ = abc.ABCMeta

    def __init__(self, timestamp, payload):
        super().__init__(timestamp, payload)
        self.data = None
        self.imp_data = None

    def calculate_impedance(self, imp_calib_info):
        """calculate impedance with the help of impedance calibration info

        Args:
            imp_calib_info (dict): dictionary of impedance calibration info including slope, offset and noise level

        """
        self.imp_data = np.round((self.get_ptp() - imp_calib_info['noise_level']) * imp_calib_info['slope']/1.e6 -
                                 imp_calib_info['offset'], decimals=0)

    def get_data(self, exg_fs=None):
        """get time vector and data

        If exg_fs is given, it returns time vector and data. If exg_fs is not given, it returns the timestamp of the
        packet alongside with the data
        """
        if exg_fs:
            n_sample = self.data.shape[1]
            time_vector = np.linspace(self.timestamp, self.timestamp + (n_sample - 1) / exg_fs, n_sample)
            return time_vector, self.data
        return self.timestamp, self.data

    def get_impedances(self):
        """get electrode impedances"""
        return self.imp_data

    def get_ptp(self):
        """Get peak to peak value"""
        return np.ptp(self.data, axis=1)

    def __str__(self):
        pass

    def _check_fletcher(self, fletcher):
        pass

    def _convert(self, bin_data):
        pass


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
        gain = EXG_UNIT * ((2 ** 23) - 1) * 6.
        self.data = np.round(data[1:, :] * v_ref / gain, 2)
        self.data_status = data[0, :]

    def _check_fletcher(self, fletcher):
        if not fletcher == b'\xaf\xbe\xad\xde':
            raise FletcherError('Fletcher value is incorrect!')

    def __str__(self):
        return "EEG: " + str(self.data[:, -1]) + "\tEEG STATUS: " + str(self.data_status[-1])


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
        n_packet = 16
        data = data.reshape((n_packet, n_chan)).astype(np.float).T
        gain = EXG_UNIT * ((2 ** 23) - 1) * 6.
        self.data = np.round(data[1:, :] * v_ref / gain, 2)
        self.status = (hex(bin_data[0]), hex(bin_data[1]), hex(bin_data[2]))

    def _check_fletcher(self, fletcher):
        if not fletcher == b'\xaf\xbe\xad\xde':
            raise FletcherError('Fletcher value is incorrect!')

    def __str__(self):
        return "EEG: " + str(self.data[:, -1]) + "\tEEG STATUS: " + str(self.status)


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
        n_packet = 16
        data = data.reshape((n_packet, n_chan)).astype(np.float).T
        gain = EXG_UNIT * ((2 ** 23) - 1) * 6.
        self.data = np.round(data * v_ref / gain, 2)
        self.status = data[0, :]

    def _check_fletcher(self, fletcher):
        if not fletcher == b'\xaf\xbe\xad\xde':
            raise FletcherError('Fletcher value is incorrect!')

    def __str__(self):
        return "EEG: " + str(self.data[:, -1]) + "\tEEG STATUS: " + str(self.status)


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
        n_packet = 16
        data = data.reshape((n_packet, n_chan)).astype(np.float).T
        gain = EXG_UNIT * ((2 ** 23) - 1) * 6.
        self.data = np.round(data * v_ref / gain, 2)

    def _check_fletcher(self, fletcher):
        if not fletcher == b'\xaf\xbe\xad\xde':
            raise FletcherError('Fletcher value is incorrect!')

    def __str__(self):
        return "EEG: " + str(self.data[:, -1])


class Orientation(Packet):
    """Orientation data packet"""
    def __init__(self, timestamp, payload):
        super().__init__(timestamp, payload)
        self._convert(payload[:-4])
        self._check_fletcher(payload[-4:])
        self.theta = None
        self.rot_axis = None

    def _convert(self, bin_data):
        data = np.copy(np.frombuffer(bin_data, dtype=np.dtype(np.int16).newbyteorder('<'))).astype(np.float)
        self.acc = 0.061 * data[0:3]  # Unit [mg/LSB]
        self.gyro = 8.750 * data[3:6]  # Unit [mdps/LSB]
        self.mag = 1.52 * np.multiply(data[6:], np.array([-1, 1, 1]))  # Unit [mgauss/LSB]

    def _check_fletcher(self, fletcher):
        if not fletcher == b'\xaf\xbe\xad\xde':
            raise FletcherError('Fletcher value is incorrect!')

    def __str__(self):
        return "Acc: " + str(self.acc) + "\tGyro: " + str(self.gyro) + "\tMag: " + str(self.mag)

    def get_data(self, srate=None):
        """Get orientation timestamp and data"""
        return [self.timestamp], self.acc.tolist() + self.gyro.tolist() + self.mag.tolist()

    def compute_angle(self, matrix=None):
        """Compute physical angle"""
        trace = matrix[0][0]+matrix[1][1]+matrix[2][2]
        theta = np.arccos((trace-1)/2)*57.2958
        n_x = matrix[2][1] - matrix[1][2]
        n_y = matrix[0][2] - matrix[2][0]
        n_z = matrix[1][0] - matrix[0][1]
        rot_axis = 1/np.sqrt((3-trace)*(1+trace))*np.array([n_x, n_y, n_z])
        self.theta = theta
        self.rot_axis = rot_axis
        return [theta, rot_axis]


class Environment(Packet):
    """Environment data packet"""
    def __init__(self, timestamp, payload):
        super().__init__(timestamp, payload)
        self._convert(payload[:-4])
        self._check_fletcher(payload[-4:])

    def _convert(self, bin_data):
        self.temperature = bin_data[0]
        self.light = (1000 / 4095) * np.frombuffer(bin_data[1:3],
                                                   dtype=np.dtype(np.uint16).newbyteorder('<'))  # Unit Lux
        self.battery = (16.8 / 6.8) * (1.8 / 2457) * np.frombuffer(bin_data[3:5],
                                                                   dtype=np.dtype(np.uint16).newbyteorder(
                                                                       '<'))  # Unit Volt
        self.battery_percentage = self._volt_to_percent(self.battery)

    def _check_fletcher(self, fletcher):
        if not fletcher == b'\xaf\xbe\xad\xde':
            raise FletcherError('Fletcher value is incorrect!')

    def __str__(self):
        return "Temperature: " + str(self.temperature) + "\tLight: " + str(self.light) + "\tBattery: " + str(
            self.battery)

    def get_data(self):
        """Get environment data"""
        return {'battery': [self.battery_percentage],
                'temperature': [self.temperature],
                'light': [self.light]}

    @staticmethod
    def _volt_to_percent(voltage):
        if voltage < 3.1:
            percentage = 1
        elif voltage < 3.5:
            percentage = 1 + (voltage - 3.1) / .4 * 10
        elif voltage < 3.8:
            percentage = 10 + (voltage - 3.5) / .3 * 40
        elif voltage < 3.9:
            percentage = 40 + (voltage - 3.8) / .1 * 20
        elif voltage < 4.:
            percentage = 60 + (voltage - 3.9) / .1 * 15
        elif voltage < 4.1:
            percentage = 75 + (voltage - 4.) / .1 * 15
        elif voltage < 4.2:
            percentage = 90 + (voltage - 4.1) / .1 * 10
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
        self.raw_data = None

    def _convert(self, bin_data):
        self.host_timestamp = np.frombuffer(bin_data, dtype=np.dtype(np.uint64).newbyteorder('<'))

    def _check_fletcher(self, fletcher):
        if not fletcher == b'\xff\xff\xff\xff':
            raise FletcherError('Fletcher value is incorrect!')

    def __str__(self):
        return "Host timestamp: " + str(self.host_timestamp)


class EventMarker(Packet):
    """Marker packet"""
    def __init__(self, timestamp, payload):
        super().__init__(timestamp, payload)
        self._convert(payload[:-4])
        self._check_fletcher(payload[-4:])

    def _convert(self, bin_data):
        self.marker_code = np.frombuffer(bin_data, dtype=np.dtype(np.uint16).newbyteorder('<'))[0]

    def _check_fletcher(self, fletcher):
        if not fletcher == b'\xaf\xbe\xad\xde':
            raise FletcherError('Fletcher value is incorrect!')

    def __str__(self):
        return "Event marker: " + str(self.marker_code)

    def get_data(self, srate=None   ):
        """Get marker data
        Args:
            srate: NOT USED. Only for compatibility purpose"""
        return [self.timestamp], [self.marker_code]


class Disconnect(Packet):
    """Disconnect packet"""
    def __init__(self, timestamp, payload):
        super().__init__(timestamp, payload)
        self._check_fletcher(payload)

    def _convert(self, bin_data):
        """Disconnect packet has no data"""

    def _check_fletcher(self, fletcher):
        if not fletcher == b'\xaf\xbe\xad\xde':
            raise FletcherError('Fletcher value is incorrect!')

    def __str__(self):
        return "Device has been disconnected!"


class DeviceInfo(Packet):
    """Device information packet"""
    def __init__(self, timestamp, payload):
        super().__init__(timestamp, payload)
        self._convert(payload[:-4])
        self._check_fletcher(payload[-4:])

    def _convert(self, bin_data):
        fw_num = np.frombuffer(bin_data, dtype=np.dtype(np.uint16).newbyteorder('<'), count=1, offset=0)
        self.firmware_version = '.'.join(list(str(fw_num)[1:-1]))
        self.sampling_rate = 16000 / (2 ** bin_data[2])
        self.adc_mask = [int(bit) for bit in format(bin_data[3], '#010b')[2:]]

    def _check_fletcher(self, fletcher):
        if not fletcher == b'\xaf\xbe\xad\xde':
            raise FletcherError('Fletcher value is incorrect!')

    def get_info(self):
        """Get device information as a dictionary"""
        return dict(firmware_version=self.firmware_version,
                    adc_mask=self.adc_mask,
                    sampling_rate=self.sampling_rate)

    def __str__(self):
        return "Firmware version: " + self.firmware_version + " - sampling rate: " + str(self.sampling_rate)\
               + " Hz" + " - ADC mask: " + str(self.adc_mask)

    def get_data(self):
        """Get firmware version"""
        return {'firmware_version': [self.firmware_version]}


class CommandRCV(Packet):
    """Command Status packet"""
    def __init__(self, timestamp, payload):
        super().__init__(timestamp, payload)
        self._convert(payload[:-4])
        self._check_fletcher(payload[-4:])

    def _convert(self, bin_data):
        self.opcode = bin_data[0]

    def _check_fletcher(self, fletcher):
        if not fletcher == b'\xaf\xbe\xad\xde':
            raise FletcherError('Fletcher value is incorrect!')

    def __str__(self):
        return "an acknowledge message for command with this opcode has been received: " + str(self.opcode)


class CommandStatus(Packet):
    """Command Status packet"""
    def __init__(self, timestamp, payload):
        super().__init__(timestamp, payload)
        self._convert(payload[:-4])
        self._check_fletcher(payload[-4:])

    def _convert(self, bin_data):
        self.opcode = bin_data[0]
        self.status = bin_data[5]

    def _check_fletcher(self, fletcher):
        if not fletcher == b'\xaf\xbe\xad\xde':
            raise FletcherError('Fletcher value is incorrect!')

    def __str__(self):
        return "Command status: " + str(self.status) + "\tfor command with opcode: " + str(self.opcode)


class CalibrationInfo(Packet):
    """Calibration Info packet"""
    def __init__(self, timestamp, payload):
        super().__init__(timestamp, payload)
        self._convert(payload[:-4])
        self._check_fletcher(payload[-4:])

    def _convert(self, bin_data):
        slope = np.frombuffer(bin_data, dtype=np.dtype(np.uint16).newbyteorder('<'), count=1, offset=0)
        self.slope = slope * 10.0
        offset = np.frombuffer(bin_data, dtype=np.dtype(np.uint16).newbyteorder('<'), count=1, offset=2)
        self.offset = offset * 0.001

    def get_info(self):
        """Get calibration info"""
        return {'slope': self.slope,
                'offset': self.offset}

    def _check_fletcher(self, fletcher):
        if not fletcher == b'\xaf\xbe\xad\xde':
            raise FletcherError('Fletcher value is incorrect!')

    def __str__(self):
        return "calibration info: slope = " + str(self.slope) + "\toffset = " + str(self.offset)


class TriggerOut(Packet):
    """Trigger Out packet"""
    def __init__(self, timestamp, payload):
        super().__init__(timestamp, payload)
        self._convert(payload[:-4])
        self._check_fletcher(payload[-4:])

    def _convert(self, bin_data):
        precise_ts = np.frombuffer(bin_data, dtype=np.dtype(np.uint32).newbyteorder('<'), count=1, offset=0)
        self.precise_ts = precise_ts

    def _check_fletcher(self, fletcher):
        if not fletcher == b'\xaf\xbe\xad\xde':
            raise FletcherError('Fletcher value is incorrect!')

    def __str__(self):
        return "Trigger Out: precise_ts = " + str(self.precise_ts)


class TriggerIn(Packet):
    """Trigger In packet"""
    def __init__(self, timestamp, payload):
        super().__init__(timestamp, payload)
        self._convert(payload[:-4])
        self._check_fletcher(payload[-4:])

    def _convert(self, bin_data):
        precise_ts = np.frombuffer(bin_data, dtype=np.dtype(np.uint32).newbyteorder('<'), count=1, offset=0)
        self.precise_ts = precise_ts/10000
        print('Got trigger in: ', precise_ts, self.timestamp)

    def _check_fletcher(self, fletcher):
        if not fletcher == b'\xaf\xbe\xad\xde':
            raise FletcherError('Fletcher value is incorrect!')

    def __str__(self):
        return "Trigger In: precise_ts = " + str(self.precise_ts)

    def get_data(self, srate=None):
        """Get trigger data
        Args:
            srate: NOT USED. Only for compatibility purpose"""
        return [self.precise_ts], [1001]


PACKET_CLASS_DICT = {
    PACKET_ID.ORN: Orientation,
    PACKET_ID.ENV: Environment,
    PACKET_ID.TS: TimeStamp,
    PACKET_ID.DISCONNECT: Disconnect,
    PACKET_ID.INFO: DeviceInfo,
    PACKET_ID.EEG94: EEG94,
    PACKET_ID.EEG98: EEG98,
    PACKET_ID.EEG99S: EEG99s,
    PACKET_ID.EEG99: EEG99s,
    PACKET_ID.EEG94R: EEG94,
    PACKET_ID.EEG98R: EEG98,
    PACKET_ID.CMDRCV: CommandRCV,
    PACKET_ID.CMDSTAT: CommandStatus,
    PACKET_ID.CALIBINFO: CalibrationInfo,
    PACKET_ID.MARKER: EventMarker,
    PACKET_ID.TRIGGER_OUT: TriggerOut,
    PACKET_ID.TRIGGER_IN: TriggerIn,
}
