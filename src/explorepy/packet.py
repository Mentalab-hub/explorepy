# -*- coding: utf-8 -*-
"""This module contains all packet classes of Mentalab Explore device"""
import abc
import logging
import struct
from enum import IntEnum

import numpy as np

from explorepy._exceptions import FletcherError


logger = logging.getLogger(__name__)

TIMESTAMP_SCALE = 10000


class PACKET_ID(IntEnum):
    """Packet ID enum"""

    ORN = 13
    ENV = 19
    TS = 27
    DISCONNECT = 111
    # New info packet containing memory and board ID: this applies to all Explore+ systems
    INFO_V2 = 97
    INFO = 99
    EEG94 = 144
    EEG98 = 146
    EEG32 = 148
    EEG98_USBC = 150
    EEG99S = 30
    EEG99 = 62
    EEG94R = 208
    EEG98R = 210
    CMDRCV = 192
    CMDSTAT = 193
    PUSHMARKER = 194
    CALIBINFO = 195
    CALIBINFO_USBC = 197
    TRIGGER_OUT = 177  # Trigger-out of Explore device
    TRIGGER_IN = 178  # Trigger-in to Explore device


EXG_UNIT = 1e-6


class Packet:
    """An abstract base class for Explore packet"""

    __metadata__ = abc.ABCMeta

    def __init__(self, timestamp, payload, time_offset=0):
        """Gets the timestamp and payload and initializes the packet object

        Args:
            timestamp (double): Raw timestamp of the packet
            payload (bytearray): A byte array including binary data and fletcher
            time_offset (double): Time offset defined by parser. It will be the timestamp of the first packet when
                                    streaming in realtime. It will be zero while converting a binary file.
        """
        self.timestamp = timestamp / TIMESTAMP_SCALE + time_offset

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
        return np.asarray([
            int.from_bytes(bin_data[x:x + 3], byteorder="little", signed=True)
            for x in range(0, len(bin_data), 3)
        ])


class EEG(Packet):
    """EEG packet class"""

    __metadata__ = abc.ABCMeta

    def __init__(self, timestamp, payload, time_offset=0):
        super().__init__(timestamp, payload, time_offset)
        self.data = None
        self.imp_data = None

    def calculate_impedance(self, imp_calib_info):
        """calculate impedance with the help of impedance calibration info

        Args:
            imp_calib_info (dict): dictionary of impedance calibration info including slope, offset and noise level

        """
        scale = imp_calib_info["slope"]
        offset = imp_calib_info["offset"]
        self.imp_data = np.round(
            (self.get_ptp() - imp_calib_info["noise_level"]) * scale / 1.0e6 - offset,
            decimals=0,
        )

    def get_data(self, exg_fs=None):
        """get time vector and data

        If exg_fs is given, it returns time vector and data. If exg_fs is not given, it returns the timestamp of the
        packet alongside with the data
        """
        if exg_fs:
            n_sample = self.data.shape[1]
            time_vector = np.linspace(self.timestamp,
                                      self.timestamp + (n_sample - 1) / exg_fs,
                                      n_sample)
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

    def __init__(self, timestamp, payload, time_offset=0):
        super().__init__(timestamp, payload, time_offset)
        self._convert(payload[:-4])
        self._check_fletcher(payload[-4:])

    def _convert(self, bin_data):
        data = Packet.int24to32(bin_data)
        n_chan = -1
        v_ref = 2.4
        n_packet = 33
        data = data.reshape((n_packet, n_chan)).astype(np.float).T
        gain = EXG_UNIT * ((2 ** 23) - 1) * 6.0
        self.data = np.round(data[1:, :] * v_ref / gain, 2)
        self.data_status = data[0, :]

    def _check_fletcher(self, fletcher):
        if not fletcher == b"\xaf\xbe\xad\xde":
            raise FletcherError("Fletcher value is incorrect!")

    def __str__(self):
        return ("EEG: " + str(self.data[:, -1]) + "\tEEG STATUS: " + str(self.data_status[-1]))


class EEG98(EEG):
    """EEG packet for 8 channel device"""

    def __init__(self, timestamp, payload, time_offset=0):
        super().__init__(timestamp, payload, time_offset)
        self._convert(payload[:-4])
        self._check_fletcher(payload[-4:])

    def _convert(self, bin_data):
        data = Packet.int24to32(bin_data)
        n_chan = -1
        v_ref = 2.4
        n_packet = 16
        data = data.reshape((n_packet, n_chan)).astype(np.float).T
        gain = EXG_UNIT * ((2 ** 23) - 1) * 6.0
        self.data = np.round(data[1:, :] * v_ref / gain, 2)
        self.status = (hex(bin_data[0]), hex(bin_data[1]), hex(bin_data[2]))

    def _check_fletcher(self, fletcher):
        if not fletcher == b"\xaf\xbe\xad\xde":
            raise FletcherError("Fletcher value is incorrect!")

    def __str__(self):
        return "EEG: " + str(self.data[:, -1]) + "\tEEG STATUS: " + str(
            self.status)


class EEG98_USBC(EEG):
    """EEG packet for 8 channel device"""

    def __init__(self, timestamp, payload, time_offset=0):
        super().__init__(timestamp, payload, time_offset)
        self._convert(payload[:-4])
        self._check_fletcher(payload[-4:])

    def _convert(self, bin_data):
        data = Packet.int24to32(bin_data)
        n_chan = -1
        v_ref = 2.4
        n_packet = 16
        data = data.reshape((n_packet, n_chan)).astype(np.float).T
        gain = EXG_UNIT * ((2 ** 23) - 1) * 6.0
        self.data = np.round(data[1:, :] * v_ref / gain, 2)
        self.status = (hex(bin_data[0]), hex(bin_data[1]), hex(bin_data[2]))

    def _check_fletcher(self, fletcher):
        if not fletcher == b"\xaf\xbe\xad\xde":
            raise FletcherError("Fletcher value is incorrect!")

    def __str__(self):
        return "EEG: " + str(self.data[:, -1]) + "\tEEG STATUS: " + str(
            self.status)


class EEG99s(EEG):
    """EEG packet for 8 channel device"""

    def __init__(self, timestamp, payload, time_offset=0):
        super().__init__(timestamp, payload, time_offset)
        self._convert(payload[:-4])
        self._check_fletcher(payload[-4:])

    def _convert(self, bin_data):
        data = Packet.int24to32(bin_data)
        n_chan = -1
        v_ref = 4.5
        n_packet = 16
        data = data.reshape((n_packet, n_chan)).astype(np.float).T
        gain = EXG_UNIT * ((2 ** 23) - 1) * 6.0
        self.data = np.round(data * v_ref / gain, 2)
        self.status = data[0, :]

    def _check_fletcher(self, fletcher):
        if not fletcher == b"\xaf\xbe\xad\xde":
            raise FletcherError("Fletcher value is incorrect!")

    def __str__(self):
        return "EEG: " + str(self.data[:, -1]) + "\tEEG STATUS: " + str(
            self.status)


class EEG99(EEG):
    """EEG packet for 8 channel device"""

    def __init__(self, timestamp, payload, time_offset=0):
        super().__init__(timestamp, payload, time_offset)
        self._convert(payload[:-4])
        self._check_fletcher(payload[-4:])

    def _convert(self, bin_data):
        data = Packet.int24to32(bin_data)
        n_chan = -1
        v_ref = 4.5
        n_packet = 16
        data = data.reshape((n_packet, n_chan)).astype(np.float).T
        gain = EXG_UNIT * ((2 ** 23) - 1) * 6.0
        self.data = np.round(data * v_ref / gain, 2)

    def _check_fletcher(self, fletcher):
        if not fletcher == b"\xaf\xbe\xad\xde":
            raise FletcherError("Fletcher value is incorrect!")

    def __str__(self):
        return "EEG: " + str(self.data[:, -1])


class EEG32(EEG):
    """EEG packet for 32 channel device"""

    def __init__(self, timestamp, payload, time_offset=0):
        super().__init__(timestamp, payload, time_offset)
        self._convert(payload[:-4])
        self._check_fletcher(payload[-4:])

    def _convert(self, bin_data):
        data = Packet.int24to32(bin_data)
        n_chan = -1
        v_ref = 2.4
        """
        Explanation for calculation of n_packet variable:
        Actual data length(ADL) = max size 545 - 12 miscellaneous bytes(pid + count + timestamp + fletcher)
        One BT packet will hold multiple samples from sensors
        ADL in integer = actual data length / 24
        n_packet = ADL in integer / number of channels of explore device
        """
        # n_packet will be 5 in the future

        n_packet = 4
        data = data.reshape((n_packet, n_chan)).astype(np.float).T
        gain = EXG_UNIT * ((2 ** 23) - 1) * 6.
        self.data = np.round(data[1:, :] * v_ref / gain, 2)
        # status bits will change in future releases as we need to use 4 bytes for 32 channel status
        self.status = (hex(bin_data[0]), hex(bin_data[1]), hex(bin_data[2]))

    def _check_fletcher(self, fletcher):
        if not fletcher == b'\xaf\xbe\xad\xde':
            raise FletcherError('Fletcher value is incorrect!')

    def __str__(self):
        return "EEG: " + str(self.data[:, -1]) + "\tEEG STATUS: " + str(self.status)


class Orientation(Packet):
    """Orientation data packet"""

    def __init__(self, timestamp, payload, time_offset=0):
        super().__init__(timestamp, payload, time_offset)
        self._convert(payload[:-4])
        self._check_fletcher(payload[-4:])
        self.theta = None
        self.rot_axis = None

    def _convert(self, bin_data):
        data = np.copy(
            np.frombuffer(bin_data, dtype=np.dtype(
                np.int16).newbyteorder("<"))).astype(np.float)
        self.acc = 0.061 * data[0:3]  # Unit [mg/LSB]
        self.gyro = 8.750 * data[3:6]  # Unit [mdps/LSB]
        self.mag = 1.52 * np.multiply(data[6:], np.array(
            [-1, 1, 1]))  # Unit [mgauss/LSB]
        self.theta = None
        self.rot_axis = None

    def _check_fletcher(self, fletcher):
        if not fletcher == b"\xaf\xbe\xad\xde":
            raise FletcherError("Fletcher value is incorrect!")

    def __str__(self):
        return ("Acc: " + str(self.acc) + "\tGyro: " + str(self.gyro) + "\tMag: " + str(self.mag))

    def get_data(self, srate=None):
        """Get orientation timestamp and data"""
        return [self.timestamp
                ], self.acc.tolist() + self.gyro.tolist() + self.mag.tolist()

    def compute_angle(self, matrix=None):
        """Compute physical angle"""
        trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
        theta = np.arccos((trace - 1) / 2) * 57.2958
        nx = matrix[2][1] - matrix[1][2]
        ny = matrix[0][2] - matrix[2][0]
        nz = matrix[1][0] - matrix[0][1]
        rot_axis = 1 / np.sqrt(
            (3 - trace) * (1 + trace)) * np.array([nx, ny, nz])
        self.theta = theta
        self.rot_axis = rot_axis
        return [theta, rot_axis]


class Environment(Packet):
    """Environment data packet"""

    def __init__(self, timestamp, payload, time_offset=0):
        super().__init__(timestamp, payload, time_offset)
        self._convert(payload[:-4])
        self._check_fletcher(payload[-4:])

    def _convert(self, bin_data):
        self.temperature = bin_data[0]
        self.light = (1000 / 4095) * np.frombuffer(
            bin_data[1:3], dtype=np.dtype(
                np.uint16).newbyteorder("<"))  # Unit Lux
        self.battery = ((16.8 / 6.8) * (1.8 / 2457) * np.frombuffer(
            bin_data[3:5], dtype=np.dtype(np.uint16).newbyteorder("<")))  # Unit Volt
        self.battery_percentage = self._volt_to_percent(self.battery)

    def _check_fletcher(self, fletcher):
        if not fletcher == b"\xaf\xbe\xad\xde":
            raise FletcherError("Fletcher value is incorrect!")

    def __str__(self):
        return "Temperature: " + str(self.temperature) + "\tLight: " + str(
            self.light) + "\tBattery: " + str(self.battery)

    def get_data(self):
        """Get environment data"""
        return {
            "battery": [self.battery_percentage],
            "temperature": [self.temperature],
            "light": [self.light],
        }

    @staticmethod
    def _volt_to_percent(voltage):
        if voltage < 3.1:
            percentage = 1
        elif voltage < 3.5:
            percentage = 1 + (voltage - 3.1) / 0.4 * 10
        elif voltage < 3.8:
            percentage = 10 + (voltage - 3.5) / 0.3 * 40
        elif voltage < 3.9:
            percentage = 40 + (voltage - 3.8) / 0.1 * 20
        elif voltage < 4.0:
            percentage = 60 + (voltage - 3.9) / 0.1 * 15
        elif voltage < 4.1:
            percentage = 75 + (voltage - 4.0) / 0.1 * 15
        elif voltage < 4.2:
            percentage = 90 + (voltage - 4.1) / 0.1 * 10
        elif voltage > 4.2:
            percentage = 100

        percentage = int(percentage)
        return percentage


class TimeStamp(Packet):
    """Time stamp data packet"""

    def __init__(self, timestamp, payload, time_offset=0):
        super().__init__(timestamp, payload, time_offset)
        self._convert(payload[:-4])
        self._check_fletcher(payload[-4:])
        self.raw_data = None

    def _convert(self, bin_data):
        self.host_timestamp = np.frombuffer(bin_data,
                                            dtype=np.dtype(
                                                np.uint64).newbyteorder("<"))

    def _check_fletcher(self, fletcher):
        if not fletcher == b"\xff\xff\xff\xff":
            raise FletcherError("Fletcher value is incorrect!")

    def __str__(self):
        return "Host timestamp: " + str(self.host_timestamp)


class EventMarker(Packet):
    """Abstract class for event markers"""

    __metadata__ = abc.ABCMeta

    def __init__(self, timestamp, payload, time_offset=0):
        super().__init__(timestamp, payload, time_offset)
        self.code = None
        self._label_prefix = None

    @abc.abstractmethod
    def _convert(self, bin_data):
        pass

    def _check_fletcher(self, fletcher):
        if not fletcher == b"\xaf\xbe\xad\xde":
            raise FletcherError("Fletcher value is incorrect!")

    def get_data(self, srate=None):
        """Get marker data
        Args:
            srate: NOT USED. Only for compatibility purpose"""
        return [self.timestamp], [self._label_prefix + str(self.code)]

    def __str__(self):
        return (
            f"{self.__class__.__name__}, Timestamp: {self.timestamp}, Code: {self.code}"
        )


class PushButtonMarker(EventMarker):
    """Push Button Marker packet"""

    def __init__(self, timestamp, payload, time_offset=0):
        super().__init__(timestamp, payload, time_offset)
        self._convert(payload[:-4])
        self._check_fletcher(payload[-4:])
        self._label_prefix = "pb_"

    def _convert(self, bin_data):
        self.code = np.frombuffer(bin_data,
                                  dtype=np.dtype(
                                      np.uint16).newbyteorder("<"))[0]


class SoftwareMarker(EventMarker):
    """Software marker packet"""

    def __init__(self, timestamp, payload, time_offset=0):
        super().__init__(timestamp, payload, time_offset)
        self._convert(payload[:-4])
        self._check_fletcher(payload[-4:])
        self._label_prefix = "sw_"

    def _convert(self, bin_data):
        self.code = np.frombuffer(bin_data,
                                  dtype=np.dtype(
                                      np.uint16).newbyteorder("<"))[0]

    @staticmethod
    def create(local_time, code):
        """Create a software marker

        Args:
            local_time (double): Local time from LSL
            code (int): Event marker code

        Returns:
            SoftwareMarker
        """
        return SoftwareMarker(
            local_time * TIMESTAMP_SCALE,
            payload=bytearray(struct.pack("<H", code) + b"\xaf\xbe\xad\xde"),
        )


class TriggerIn(EventMarker):
    """Trigger in packet"""

    def __init__(self, timestamp, payload, time_offset=0):
        super(TriggerIn, self).__init__(timestamp, payload, time_offset)
        self._time_offset = time_offset
        self._convert(payload[:-4])
        self._check_fletcher(payload[-4:])
        self._label_prefix = "in_"

    def _convert(self, bin_data):
        precise_ts = np.asscalar(
            np.frombuffer(bin_data,
                          dtype=np.dtype(np.uint32).newbyteorder("<"),
                          count=1,
                          offset=0))
        self.timestamp = precise_ts / TIMESTAMP_SCALE + self._time_offset
        code = np.asscalar(
            np.frombuffer(bin_data,
                          dtype=np.dtype(np.uint16).newbyteorder("<"),
                          count=1,
                          offset=4))
        self.code = code
        mac_address = hex(
            int(
                np.frombuffer(
                    bin_data,
                    dtype=np.dtype(np.uint16).newbyteorder("<"),
                    count=1,
                    offset=6,
                )))
        self.mac_address = mac_address


class TriggerOut(EventMarker):
    """Trigger-out packet"""

    def __init__(self, timestamp, payload, time_offset=0):
        super(TriggerOut, self).__init__(timestamp, payload, time_offset)
        self._time_offset = time_offset
        self._convert(payload[:-4])
        self._check_fletcher(payload[-4:])
        self._label_prefix = "out_"

    def _convert(self, bin_data):
        precise_ts = np.asscalar(
            np.frombuffer(bin_data,
                          dtype=np.dtype(np.uint32).newbyteorder("<"),
                          count=1,
                          offset=0))

        self.timestamp = precise_ts / TIMESTAMP_SCALE + self._time_offset
        code = np.asscalar(
            np.frombuffer(bin_data,
                          dtype=np.dtype(np.uint16).newbyteorder("<"),
                          count=1,
                          offset=4))
        """
        if label == 240:
            label = "Sync"
        if label == 15:
            label = "ADS_Start"
        """
        self.code = code
        mac_address = hex(
            int(
                np.frombuffer(
                    bin_data,
                    dtype=np.dtype(np.uint16).newbyteorder("<"),
                    count=1,
                    offset=6,
                )))
        self.mac_address = mac_address


class Disconnect(Packet):
    """Disconnect packet"""

    def __init__(self, timestamp, payload, time_offset=0):
        super().__init__(timestamp, payload, time_offset)
        self._check_fletcher(payload)

    def _convert(self, bin_data):
        """Disconnect packet has no data"""

    def _check_fletcher(self, fletcher):
        if not fletcher == b"\xaf\xbe\xad\xde":
            raise FletcherError("Fletcher value is incorrect!")

    def __str__(self):
        return "Device has been disconnected!"


class DeviceInfo(Packet):
    """Device information packet"""

    def __init__(self, timestamp, payload, time_offset=0):
        super(DeviceInfo, self).__init__(timestamp, payload, time_offset)
        self._convert(payload[:-4])
        self._check_fletcher(payload[-4:])

    def _convert(self, bin_data):
        fw_num = np.frombuffer(bin_data,
                               dtype=np.dtype(np.uint16).newbyteorder("<"),
                               count=1,
                               offset=0)

        self.firmware_version = ".".join([char for char in str(fw_num)[1:-1]])
        self.sampling_rate = 16000 / (2 ** bin_data[2])
        self.adc_mask = [int(bit) for bit in format(bin_data[3], "#010b")[2:]]

    def _check_fletcher(self, fletcher):
        if not fletcher == b"\xaf\xbe\xad\xde":
            raise FletcherError("Fletcher value is incorrect!")

    def get_info(self):
        """Get device information as a dictionary"""
        return dict(
            firmware_version=self.firmware_version,
            adc_mask=self.adc_mask,
            sampling_rate=self.sampling_rate,
        )

    def __str__(self):
        return "Firmware version: {} - sampling rate: {} - ADC mask: {}".format(
            self.firmware_version, self.sampling_rate, self.adc_mask)

    def get_data(self):
        """Get firmware version"""
        return {"firmware_version": [self.firmware_version]}


class DeviceInfoV2(Packet):
    """Device information packet containing additional information board id and memory info"""

    def __init__(self, timestamp, payload, time_offset=0):
        super(DeviceInfoV2, self).__init__(timestamp, payload, time_offset)
        self._convert(payload[:-4])
        self._check_fletcher(payload[-4:])

    def _convert(self, bin_data):
        self.board_id = bin_data[:15].decode('utf-8', errors='ignore')

        fw_num = np.frombuffer(bin_data,
                               dtype=np.dtype(np.uint16).newbyteorder("<"),
                               count=1,
                               offset=16)
        self.firmware_version = ".".join([char for char in str(fw_num)[1:-1]])
        self.sampling_rate = 16000 / (2 ** bin_data[18])
        self.adc_mask = [int(bit) for bit in format(bin_data[19], "#010b")[2:]]
        self.is_memory_available = bin_data[20]

    def _check_fletcher(self, fletcher):
        if not fletcher == b"\xaf\xbe\xad\xde":
            raise FletcherError("Fletcher value is incorrect!")

    def get_info(self):
        """Get device information as a dictionary"""
        return dict(
            firmware_version=self.firmware_version,
            adc_mask=self.adc_mask,
            sampling_rate=self.sampling_rate,
            board_id=self.board_id,
            memory_info=self.is_memory_available
        )

    def __str__(self):
        return "Firmware version: {} - sampling rate: {} - ADC mask: {}".format(
            self.firmware_version, self.sampling_rate, self.adc_mask)

    def get_data(self):
        """Get firmware version"""
        return {"firmware_version": [self.firmware_version]}


class CommandRCV(Packet):
    """Command Status packet"""

    def __init__(self, timestamp, payload, time_offset=0):
        super(CommandRCV, self).__init__(timestamp, payload, time_offset)
        self._convert(payload[:-4])
        self._check_fletcher(payload[-4:])

    def _convert(self, bin_data):
        self.opcode = bin_data[0]

    def _check_fletcher(self, fletcher):
        if not fletcher == b"\xaf\xbe\xad\xde":
            raise FletcherError("Fletcher value is incorrect!")

    def __str__(self):
        return (
            "an acknowledge message for command with this opcode has been received: " + str(self.opcode))


class CommandStatus(Packet):
    """Command Status packet"""

    def __init__(self, timestamp, payload, time_offset=0):
        super(CommandStatus, self).__init__(timestamp, payload, time_offset)
        self._convert(payload[:-4])
        self._check_fletcher(payload[-4:])

    def _convert(self, bin_data):
        self.opcode = bin_data[0]
        self.status = bin_data[5]

    def _check_fletcher(self, fletcher):
        if not fletcher == b"\xaf\xbe\xad\xde":
            raise FletcherError("Fletcher value is incorrect!")

    def __str__(self):
        return ("Command status: " + str(self.status) + "\tfor command with opcode: " + str(self.opcode))


class CalibrationInfo(Packet):
    """Calibration Info packet"""

    def __init__(self, timestamp, payload, time_offset=0):
        super(CalibrationInfo, self).__init__(timestamp, payload, time_offset)
        self._convert(payload[:-4])
        self._check_fletcher(payload[-4:])

    def _convert(self, bin_data):
        slope = np.frombuffer(bin_data,
                              dtype=np.dtype(np.uint16).newbyteorder("<"),
                              count=1,
                              offset=0)
        self.slope = slope * 10.0
        offset = np.frombuffer(bin_data,
                               dtype=np.dtype(np.uint16).newbyteorder("<"),
                               count=1,
                               offset=2)
        self.offset = offset * 0.001

    def get_info(self):
        """Get calibration info"""
        return {"slope": self.slope, "offset": self.offset}

    def _check_fletcher(self, fletcher):
        if not fletcher == b"\xaf\xbe\xad\xde":
            raise FletcherError("Fletcher value is incorrect!")

    def __str__(self):
        return ("calibration info: slope = " + str(self.slope) + "\toffset = " + str(self.offset))


class CalibrationInfo_USBC(CalibrationInfo):
    """Calibration Info packet"""

    def __init__(self, timestamp, payload, time_offset=0):
        super(CalibrationInfo_USBC, self).__init__(timestamp, payload,
                                                   time_offset)
        self._convert(payload[:-4])
        self._check_fletcher(payload[-4:])

    def _convert(self, bin_data):
        slope = np.frombuffer(bin_data,
                              dtype=np.dtype(np.uint16).newbyteorder("<"),
                              count=1,
                              offset=0)
        self.slope = slope * 10.0
        offset = np.frombuffer(bin_data,
                               dtype=np.dtype(np.uint16).newbyteorder("<"),
                               count=1,
                               offset=2)
        self.offset = offset * 0.01

    def get_info(self):
        """Get calibration info"""
        return {"slope": self.slope, "offset": self.offset}

    def _check_fletcher(self, fletcher):
        if not fletcher == b"\xaf\xbe\xad\xde":
            raise FletcherError("Fletcher value is incorrect!")

    def __str__(self):
        return ("calibration info: slope = " + str(self.slope) + "\toffset = " + str(self.offset))


PACKET_CLASS_DICT = {
    PACKET_ID.ORN: Orientation,
    PACKET_ID.ENV: Environment,
    PACKET_ID.TS: TimeStamp,
    PACKET_ID.DISCONNECT: Disconnect,
    PACKET_ID.INFO: DeviceInfo,
    PACKET_ID.INFO_V2: DeviceInfoV2,
    PACKET_ID.EEG94: EEG94,
    PACKET_ID.EEG98: EEG98,
    PACKET_ID.EEG99S: EEG99s,
    PACKET_ID.EEG99: EEG99s,
    PACKET_ID.EEG94R: EEG94,
    PACKET_ID.EEG98R: EEG98,
    PACKET_ID.EEG98_USBC: EEG98_USBC,
    PACKET_ID.EEG32: EEG32,
    PACKET_ID.CMDRCV: CommandRCV,
    PACKET_ID.CMDSTAT: CommandStatus,
    PACKET_ID.CALIBINFO: CalibrationInfo,
    PACKET_ID.CALIBINFO_USBC: CalibrationInfo_USBC,
    PACKET_ID.PUSHMARKER: PushButtonMarker,
    PACKET_ID.TRIGGER_IN: TriggerIn,
    PACKET_ID.TRIGGER_OUT: TriggerOut,
}
