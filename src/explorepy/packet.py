# -*- coding: utf-8 -*-
"""This module contains all packet classes of Mentalab Explore device"""
import abc
import binascii
import logging
import struct
from enum import IntEnum

import numba as nb
import numpy as np

import explorepy.tools
from explorepy._exceptions import FletcherError


logger = logging.getLogger(__name__)


class PACKET_ID(IntEnum):
    """Packet ID enum"""

    ORN_V1 = 13
    ORN_V2 = 14
    ENV = 19
    TS = 27
    DISCONNECT = 111
    # Info packet from BLE devices, applies to Explore Pro
    INFO_BLE = 98
    INFO_HYP = 99
    # New info packet containing memory and board ID: this applies to all Explore+ systems
    INFO_V2 = 97
    INFO = 96
    EEG94 = 144
    EEG98 = 146
    EEG32 = 148
    EEG98_USBC = 150
    EEG98_BLE = 151
    EEG32_BLE = 152
    EEG16_BLE = 153
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
    VERSION_INFO = 199


EXG_UNIT = 1e-6
GAIN = EXG_UNIT * 8388607 * 6.0


class Packet(abc.ABC):
    """An abstract base class for Explore packet"""

    def __init__(self, timestamp, payload, time_offset=0):
        """Gets the timestamp and payload and initializes the packet object

        Args:
            timestamp (double): Raw timestamp of the packet
            payload (bytearray): A byte array including binary data and fletcher
            time_offset (double): Time offset defined by parser. It will be the timestamp of the first packet when
                                    streaming in realtime. It will be zero while converting a binary file.
        """
        self.timestamp = timestamp
        self._convert(payload[:-4])
        self._check_fletcher(payload[-4:])

    @abc.abstractmethod
    def _convert(self, bin_data):
        """Read the binary data and convert it to real values"""

    def _check_fletcher(self, fletcher):
        if not fletcher == b"\xaf\xbe\xad\xde":
            raise FletcherError("Fletcher value is incorrect!")

    @abc.abstractmethod
    def __str__(self):
        """Print the data/info"""

    @staticmethod
    def parse_packets_batch(packet_list):
        """Process multiple packets at once for offline file parsing.
        Args:
            packet_list (list): List of tuples containing (packet_id, timestamp, binary_data, time_offset)
                            where:
                            - packet_id (int): The ID of the packet from PACKET_ID enum
                            - timestamp (float): Timestamp of the packet
                            - binary_data (bytearray): The binary payload data
                            - time_offset (float, optional): Time offset for the packet. Defaults to 0.
        Returns:
            list: List of parsed packet objects corresponding to their respective packet types
        """
        parsed_packets = []
        parsed_packets_append = parsed_packets.append
        packet_classes = {}
        for packet_info in packet_list:
            try:
                if len(packet_info) == 3:
                    pid, timestamp, bin_data = packet_info
                    time_offset = 0
                else:
                    pid, timestamp, bin_data, time_offset = packet_info
                packet_class = packet_classes.get(pid)
                if packet_class is None:
                    if pid not in PACKET_CLASS_DICT:
                        logger.warning(f"Invalid packet ID: {pid}")
                        continue
                    packet_class = PACKET_CLASS_DICT[pid]
                    packet_classes[pid] = packet_class
                packet = packet_class(timestamp, bin_data, time_offset)
                parsed_packets_append(packet)
            except FletcherError:
                logger.warning("Fletcher checksum error in packet")
                continue
            except (ValueError, KeyError) as e:
                logger.warning(f"Error processing packet: {str(e)}")
                continue
        return parsed_packets

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def int24to32(bin_data, byteorder_data='little'):
        """Converts binary data to int32 using Numba.
        fastmath and cache enabled for repeated calls.

        Args:
            bin_data (bytes or bytearray): Binary data structured as int24 values.
            byteorder_data (str): Byte order ('little' or 'big').

        Returns:
            np.ndarray: Converted int32 values.
        """
        length = len(bin_data)
        assert length % 3 == 0, "Packet length error!"

        num_values = length // 3
        output = np.empty(num_values, dtype=np.int32)
        bin_array = np.frombuffer(bin_data, dtype=np.uint8)

        if byteorder_data == 'little':
            shift = np.array([1, 256, 65536], dtype=np.int32)
        else:  # 'big'
            shift = np.array([65536, 256, 1], dtype=np.int32)

        for i in range(num_values):
            idx = i * 3
            val = (bin_array[idx] * shift[0] + bin_array[idx + 1] * shift[1] + bin_array[idx + 2] * shift[2])

            if val >= 0x800000:
                val -= 0x1000000

            output[i] = val

        return output


class PacketBIN(Packet):
    """
    Packet that holds complete, raw binary data (from ID to fletcher). An instance of this class is generated by the
    parser for each incoming packet, dispatched by the stream_processor with topic TOPICS.packet_bin and used in the
    Debug class.
    """

    def __init__(self, raw_data):
        self.bin_data = raw_data

    def _convert(self, bin_data):
        """Read the binary data and convert it to real values"""
        pass

    def __str__(self):
        return f"{binascii.hexlify(bytearray(self.bin_data))}"


class EEG(Packet):
    """EEG packet class"""

    @abc.abstractmethod
    def __init__(self, timestamp, payload, time_offset=0, v_ref=None, n_packet=None):
        self.v_ref = v_ref
        self.n_packet = n_packet
        self.data = None
        self.imp_data = None
        if not isinstance(self, EEG_BLE):
            self.byteorder_data = 'little'
        super().__init__(timestamp, payload, time_offset)

    def _convert(self, bin_data):
        """Read the binary data and convert it to real values"""
        if not self.v_ref or not self.n_packet:
            raise ValueError(
                "v_ref or n_packet cannot be null for conversion!")
        try:
            data = Packet.int24to32(bin_data, self.byteorder_data)
            n_chan = -1
            data = data.reshape((self.n_packet, n_chan)).astype(float).T
            if isinstance(self, EEG_BLE):
                self.data = np.round(data * self.v_ref / GAIN, 2)
                return
            self.data = np.round(data[1:, :] * self.v_ref / GAIN, 2)
            self.status = self.int32_to_status(data[0, :])
        except UnboundLocalError as error:
            logger.debug('Got UnboundLocalError in packet conversion')
            raise error
        except TypeError as error:
            logger.debug('Got TypeError in packet conversion')
            raise error
        except ValueError as error:
            logger.debug('Got ValueError in packet conversion')
            raise error

    @staticmethod
    def int32_to_status(data):
        data = data.astype(int)
        ads = data & 255
        empty = data >> 8 & 255
        sr = (16000 / (2 ** (data >> 16 & 255))).astype(int)
        status = {
            "ads": ads,
            "empty": empty,
            "sr": sr,
        }
        return status

    def calculate_impedance(self, imp_calib_info):
        """calculate impedance with the help of impedance calibration info

        Args:
            imp_calib_info (dict): dictionary of impedance calibration info including slope, offset and noise level

        """
        scale = imp_calib_info["slope"]
        offset = imp_calib_info["offset"]
        self.imp_data = np.round(
            (self.get_ptp()
             - imp_calib_info["noise_level"]) * scale / 1.0e6 - offset,
            decimals=0,
        )

    def get_data(self, exg_fs=None):
        """get time vector and data

        If exg_fs is given, it returns time vector and data. If exg_fs is not given, it returns the timestamp of the
        packet alongside with the data
        """
        return np.array([self.timestamp]), self.data

    def get_impedances(self):
        """get electrode impedances"""
        return self.imp_data

    def get_ptp(self):
        """Get peak to peak value"""
        return np.ptp(self.data, axis=1)

    def __str__(self):
        return "EEG: " + str(self.data[:, -1])


class EEG94(EEG):
    """EEG packet for 4 channel device"""

    def __init__(self, timestamp, payload, time_offset=0):
        super().__init__(timestamp, payload, time_offset, v_ref=2.4, n_packet=33)


class EEG98(EEG):
    """EEG packet for 8 channel device"""

    def __init__(self, timestamp, payload, time_offset=0):
        super().__init__(timestamp, payload, time_offset, v_ref=2.4, n_packet=16)


class EEG98_USBC(EEG):
    """EEG packet for 8 channel device"""

    def __init__(self, timestamp, payload, time_offset=0):
        super().__init__(timestamp, payload, time_offset, v_ref=2.4, n_packet=16)


class EEG_BLE(EEG):
    def __init__(self, timestamp, payload, time_offset=0):
        self.byteorder_data = 'big'
        self.channel_order = [7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8, 23, 22, 21, 20, 19, 18, 17, 16,
                              31, 30, 29, 28, 27, 26, 25, 24]
        super().__init__(timestamp, payload, time_offset, v_ref=2.4, n_packet=1)
        data_length = len(self.data)
        self.data = self.data[self.channel_order[:data_length]]


class EEG98_BLE(EEG_BLE):
    """EEG packet for 8 channel device"""

    def __init__(self, timestamp, payload, time_offset=0):
        super().__init__(timestamp, payload, time_offset)


class EEG32_BLE(EEG_BLE):
    """EEG packet for 32 channel BLE device"""

    def __init__(self, timestamp, payload, time_offset=0):
        super().__init__(timestamp, payload, time_offset)


class EEG16_BLE(EEG_BLE):
    """EEG packet for 16 channel BLE device"""

    def __init__(self, timestamp, payload, time_offset=0):
        super().__init__(timestamp, payload, time_offset)


class EEG99(EEG):
    """EEG packet for 8 channel device"""

    def __init__(self, timestamp, payload, time_offset=0):
        super().__init__(timestamp, payload, time_offset, v_ref=4.5, n_packet=16)


class EEG32(EEG):
    """EEG packet for 32 channel device"""

    def __init__(self, timestamp, payload, time_offset=0):
        super().__init__(timestamp, payload, time_offset, v_ref=2.4, n_packet=4)


class Orientation(Packet):
    def __init__(self, timestamp, payload, time_offset=0):
        super().__init__(timestamp, payload, time_offset)
        self.theta = None
        self.rot_axis = None


class OrientationV1(Orientation):
    """Orientation data packet"""

    def __init__(self, timestamp, payload, time_offset=0):
        super().__init__(timestamp, payload, time_offset)

    def _convert(self, bin_data):
        data = np.copy(
            np.frombuffer(bin_data, dtype=np.dtype(
                np.int16).newbyteorder("<"))).astype(float)
        self.acc = 0.061 * data[0:3]  # Unit [mg/LSB]
        self.gyro = 8.750 * data[3:6]  # Unit [mdps/LSB]
        self.mag = 1.52 * np.multiply(data[6:], np.array(
            [-1, 1, 1]))  # Unit [mgauss/LSB]
        self.theta = None
        self.rot_axis = None

    def __str__(self):
        return "Acc: " + str(self.acc) + "\tGyro: " + str(self.gyro) + "\tMag: " + str(self.mag)

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


class OrientationV2(Orientation):
    """Orientation data packet"""

    def __init__(self, timestamp, payload, time_offset=0):
        self.quat = None
        super().__init__(timestamp, payload, time_offset)

    def _convert(self, bin_data):
        data = np.copy(
            np.frombuffer(bin_data[0:18], dtype=np.dtype(
                np.int16).newbyteorder("<"))).astype(float)
        self.acc = 0.122 * data[0:3]  # Unit [mg/LSB]
        self.gyro = 70 * data[3:6]  # Unit [mdps/LSB]
        self.mag = 1.52 * np.multiply(data[6:9], np.array(
            [-1, 1, 1]))  # Unit [mgauss/LSB]
        data = np.copy(
            np.frombuffer(bin_data[18:34], dtype=np.dtype(
                np.float32).newbyteorder("<"))).astype(float)
        self.quat = data
        self.theta = None
        self.rot_axis = None

    def __str__(self):
        return "Acc: " + str(self.acc) + "\tGyro: " + str(self.gyro) + "\tMag: " + str(
            self.mag) + "\tQuat: " + str(self.quat)

    def get_data(self, srate=None):
        """Get orientation timestamp and data"""
        return [self.timestamp
                ], self.acc.tolist() + self.gyro.tolist() + self.mag.tolist() + self.quat.tolist()

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

    def _convert(self, bin_data):
        self.temperature = bin_data[0]
        self.light = (1000 / 4095) * np.frombuffer(
            bin_data[1:3], dtype=np.dtype(
                np.uint16).newbyteorder("<"))  # Unit Lux
        self.battery = ((16.8 / 6.8) * (1.8 / 2457) * np.frombuffer(
            bin_data[3:5], dtype=np.dtype(np.uint16).newbyteorder("<")))  # Unit Volt

        # constant, measured in recording, actually 4.2, but let's say 4.10 is better
        max_voltage = 4.1
        min_voltage = 3.45  # constant , measured in recording
        voltage_span = max_voltage - min_voltage
        percent = int(((self.battery - min_voltage) / voltage_span) * 100)
        self.battery_percentage = max(0, min(percent, 100))

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


class TimeStamp(Packet):
    """Time stamp data packet"""

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

    @abc.abstractmethod
    def __init__(self, timestamp, payload, time_offset=0):
        self.code = None
        self._label_prefix = None
        super().__init__(timestamp, payload, time_offset)

    @abc.abstractmethod
    def _convert(self, bin_data):
        pass

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
        self._label_prefix = "pb_"

    def _convert(self, bin_data):
        self.code = np.frombuffer(bin_data,
                                  dtype=np.dtype(
                                      np.uint16).newbyteorder("<"))[0]


class ExternalMarker(EventMarker):
    """External marker packet"""

    def __init__(self, timestamp, payload, name):
        super().__init__(timestamp, payload, 0)
        self._label_prefix = "lsl_"
        self.name = name

    def _convert(self, bin_data):
        self.code = bin_data[:15].decode('utf-8', errors='ignore')

    @staticmethod
    def create(lsl_time, marker_string, name):
        """Create a software marker

        Args:
            lsl_time (double): Local time from LSL
            marker_string (string): Event marker code
            name (string): Marker stream name

        Returns:
            SoftwareMarker
        """
        if not isinstance(marker_string, str):
            raise ValueError("Marker label must be a string")
        if len(marker_string) > 20 or len(marker_string) < 1:
            raise ValueError(
                "Marker label length must be between 1 and 7 characters")
        byte_array = bytes(marker_string, 'utf-8')
        return ExternalMarker(
            lsl_time,
            payload=bytearray(byte_array + b"\xaf\xbe\xad\xde"),
            name=name
        )

class SoftwareMarker(ExternalMarker):
    def __init__(self, timestamp, payload, name):
        super().__init__(timestamp, payload, 0)
        _label_prefix = 'sw_'


class Trigger(EventMarker):
    @abc.abstractmethod
    def __init__(self, timestamp, payload, time_offset=0):
        self._time_offset = time_offset
        super().__init__(timestamp, payload, time_offset)

    def _convert(self, bin_data):
        precise_ts = np.ndarray.item(
            np.frombuffer(bin_data,
                          dtype=np.dtype(np.uint32).newbyteorder("<"),
                          count=1,
                          offset=0))
        scale = 100000 if explorepy.tools.is_explore_pro_device() else 10000
        self.timestamp = precise_ts / scale + self._time_offset
        code = np.ndarray.item(
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


class TriggerIn(Trigger):
    def __init__(self, timestamp, payload, time_offset=0):
        super().__init__(timestamp, payload, time_offset)
        self._label_prefix = "in_"


class TriggerOut(Trigger):
    def __init__(self, timestamp, payload, time_offset=0):
        super().__init__(timestamp, payload, time_offset)
        self._label_prefix = "out_"


class Disconnect(Packet):
    """Disconnect packet"""

    def _convert(self, bin_data):
        """Disconnect packet has no data"""
        pass

    def __str__(self):
        return "Device has been disconnected!"


class DeviceInfo(Packet):
    def _convert(self, bin_data):
        fw_num = np.frombuffer(bin_data,
                               dtype=np.dtype(np.uint16).newbyteorder("<"),
                               count=1,
                               offset=0)

        self.firmware_version = ".".join([char for char in str(fw_num)[1:-1]])
        self.sampling_rate = int(16000 / (2 ** bin_data[2]))
        self.adc_mask = [int(bit) for bit in format(bin_data[3], "#010b")[2:]]
        self.is_imp_mode = False

    def get_info(self):
        """Get device information as a dictionary"""
        return dict(
            firmware_version=self.firmware_version,
            adc_mask=self.adc_mask,
            sampling_rate=self.sampling_rate,
            is_imp_mode=self.is_imp_mode
        )

    def __str__(self):
        return "Firmware version: {} - sampling rate: {} - ADC mask: {}".format(
            self.firmware_version, self.sampling_rate, self.adc_mask)


class DeviceInfoV2(DeviceInfo):
    def _convert(self, bin_data):
        self.board_id = bin_data[:15].decode('utf-8', errors='ignore')
        super()._convert(bin_data[16:])
        self.is_memory_available = bin_data[20]

    def get_info(self):
        as_dict = super().get_info()
        as_dict['board_id'] = self.board_id
        as_dict['memory_info'] = self.is_memory_available
        as_dict['is_imp_mode'] = self.is_imp_mode
        return as_dict


class DeviceInfoBLE(DeviceInfoV2):
    def _convert(self, bin_data):
        super()._convert(bin_data)
        # basic binary conversion shows up binary number with leading zeroes cut off
        # here we format the raw byte to full 8 bits
        # https://stackoverflow.com/questions/10411085/converting-integer-to-binary-in-python
        self.sps_info = '{0:08b}'.format(bin_data[21])
        self.max_online_sps = 250 * pow(2, 6 - int(self.sps_info[4:], 2))
        self.max_offline_sps = 250 * pow(2, 6 - int(self.sps_info[:4], 2))
        # second LSB is impedance mode indicator
        self.is_imp_mode = True if self.is_memory_available == 3 else False

    def get_info(self):
        as_dict = super().get_info()
        as_dict['max_online_sps'] = self.max_online_sps
        as_dict['max_offline_sps'] = self.max_offline_sps
        as_dict['is_imp_mode'] = self.is_imp_mode
        return as_dict


class DeviceInfoHyp(DeviceInfoBLE):
    pass


class CommandRCV(Packet):
    """Command Status packet"""

    def _convert(self, bin_data):
        self.opcode = bin_data[0]

    def __str__(self):
        return (
            "an acknowledge message for command with this opcode has been received: " + str(self.opcode))


class CommandStatus(Packet):
    """Command Status packet"""

    def _convert(self, bin_data):
        self.opcode = bin_data[0]
        self.status = bin_data[5]

    def __str__(self):
        return "Command status: " + str(self.status) + "\tfor command with opcode: " + str(self.opcode)


class CalibrationInfoBase(Packet):
    @abc.abstractmethod
    def _convert(self, bin_data, offset_multiplier=0.001):
        slope = np.frombuffer(bin_data,
                              dtype=np.dtype(np.uint16).newbyteorder("<"),
                              count=1,
                              offset=0).item()
        self.slope = slope * 10.0
        offset = np.frombuffer(bin_data,
                               dtype=np.dtype(np.uint16).newbyteorder("<"),
                               count=1,
                               offset=2).item()
        self.offset = offset * offset_multiplier

    def get_info(self):
        """Get calibration info"""
        return {"slope": self.slope, "offset": self.offset}

    def __str__(self):
        return "calibration info: slope = " + str(self.slope) + "\toffset = " + str(self.offset)


class CalibrationInfo(CalibrationInfoBase):
    def _convert(self, bin_data):
        super()._convert(bin_data, offset_multiplier=0.001)


class CalibrationInfo_USBC(CalibrationInfoBase):
    def _convert(self, bin_data):
        super()._convert(bin_data, offset_multiplier=0.01)


class BleImpedancePacket(EEG98_USBC):

    def __init__(self, timestamp, payload, time_offset=0):
        self.timestamp = timestamp

    def _convert(self, bin_data):
        pass

    def populate_packet_with_data(self, ble_packet_list):
        data_array = None
        for i in range(len(ble_packet_list)):
            _, data = ble_packet_list[i].get_data()
            if data_array is None:
                data_array = data
            else:
                data_array = np.concatenate((data_array, data), axis=1)
        self.data = data_array


class VersionInfoPacket(Packet):
    def __init__(self, timestamp, payload, time_offset=0):
        self.timestamp = timestamp
        self._convert(payload[:-4])
        self._check_fletcher(payload[-4:])
        self.timestamp = timestamp

        print(self.info)

    def _convert(self, bin_data):
        self.info = bin_data.decode('utf-8', errors='ignore')

    def __str__(self):
        return str(self.info)


PACKET_CLASS_DICT = {
    PACKET_ID.ORN_V1: OrientationV1,
    PACKET_ID.ORN_V2: OrientationV2,
    PACKET_ID.ENV: Environment,
    PACKET_ID.TS: TimeStamp,
    PACKET_ID.DISCONNECT: Disconnect,
    PACKET_ID.INFO: DeviceInfo,
    PACKET_ID.INFO_V2: DeviceInfoV2,
    PACKET_ID.INFO_BLE: DeviceInfoBLE,
    PACKET_ID.INFO_HYP: DeviceInfoHyp,
    PACKET_ID.EEG94: EEG94,
    PACKET_ID.EEG98: EEG98,
    PACKET_ID.EEG99: EEG99,
    PACKET_ID.EEG94R: EEG94,
    PACKET_ID.EEG98R: EEG98,
    PACKET_ID.EEG98_USBC: EEG98_USBC,
    PACKET_ID.EEG98_BLE: EEG98_BLE,
    PACKET_ID.EEG32_BLE: EEG32_BLE,
    PACKET_ID.EEG16_BLE: EEG16_BLE,
    PACKET_ID.EEG32: EEG32,
    PACKET_ID.CMDRCV: CommandRCV,
    PACKET_ID.CMDSTAT: CommandStatus,
    PACKET_ID.CALIBINFO: CalibrationInfo,
    PACKET_ID.CALIBINFO_USBC: CalibrationInfo_USBC,
    PACKET_ID.PUSHMARKER: PushButtonMarker,
    PACKET_ID.TRIGGER_IN: TriggerIn,
    PACKET_ID.TRIGGER_OUT: TriggerOut,
    PACKET_ID.VERSION_INFO: VersionInfoPacket,
}
