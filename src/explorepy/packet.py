# -*- coding: utf-8 -*-
import numpy as np
import abc
import struct
from functools import partial
from enum import IntEnum
from datetime import datetime


class PACKET_ID(IntEnum):
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
    def write_to_file(self, recorder):
        """
        Write EEG data to csv file

        Args:
            recorder(explorepy.tools.FileRecorder): File recorder object

        """
        pass

    def apply_bp_filter(self, exg_filter):
        """Bandpass filtering of ExG data

        Args:
        exg_filter: Filter object
        """
        self.data = exg_filter.apply_bp_filter(self.data)

    def apply_bp_filter_noise(self, exg_filter):
        """Bandpass filtering of ExG data

        Args:
        exg_filter: Filter object
        """
        self.data = exg_filter.apply_bp_filter_noise(self.data)

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

    def calculate_impedance(self, imp_calib_info):
        """
        calculate impedance with the help of impedance calibration info

        Args:
            imp_calib_info (dict): dictionary of impedance calibration info including slope, offset and noise level

        """
        mag = np.ptp(self.data, axis=1)
        self.imp_data = np.round(
            (mag - imp_calib_info['noise_level']) * imp_calib_info['slope'] - imp_calib_info['offset'], decimals=0)

    def push_to_dashboard(self, dashboard):
        n_sample = self.data.shape[1]
        time_vector = np.linspace(self.timestamp, self.timestamp + (n_sample - 1) / dashboard.EEG_SRATE, n_sample)
        dashboard.doc.add_next_tick_callback(partial(dashboard.update_exg, time_vector=time_vector, ExG=self.data))

    def push_to_imp_dashboard(self, dashboard, imp_calib_info):
        self.calculate_impedance(imp_calib_info)
        dashboard.doc.add_next_tick_callback(partial(dashboard.update_imp, imp=self.imp_data))

    def write_to_file(self, recorder):
        tmpstmp = np.linspace(self.timestamp, self.timestamp + (self.data.shape[1]-1)*0.004,
                              self.data.shape[1])  # 250 Hz
        recorder.write_data(np.concatenate((tmpstmp[:, np.newaxis], self.data.T), axis=1).T)


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
        return "EEG: " + str(self.data[:, -1]) + "\tEEG STATUS: " + str(self.dataStatus[-1]  )


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
        self.data = data[1:, :] * v_ref / ((2 ** 23) - 1) / 6.
        self.status = (hex(bin_data[0]), hex(bin_data[1]), hex(bin_data[2]))

    def _check_fletcher(self, fletcher):
        assert fletcher == b'\xaf\xbe\xad\xde', "Fletcher error!"

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
        self.data = data[1:, :] * v_ref / ((2 ** 23) - 1) / 6.
        self.status = data[0, :]

    def _check_fletcher(self, fletcher):
        assert fletcher == b'\xaf\xbe\xad\xde', "Fletcher error!"

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
        self.data = data * v_ref / ((2 ** 23) - 1) / 6.

    def _check_fletcher(self, fletcher):
        assert fletcher == b'\xaf\xbe\xad\xde', "Fletcher error!"

    def __str__(self):
        return "EEG: " + str(self.data[:, -1])


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

    def write_to_file(self, recorder):
        recorder.write_data(np.array([self.timestamp] + self.acc.tolist() +
                                     self.gyro.tolist() + self.mag.tolist())[:, np.newaxis])

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
        self.light = (1000 / 4095) * np.frombuffer(bin_data[1:3],
                                                   dtype=np.dtype(np.uint16).newbyteorder('<'))  # Unit Lux
        self.battery = (16.8 / 6.8) * (1.8 / 2457) * np.frombuffer(bin_data[3:5],
                                                                   dtype=np.dtype(np.uint16).newbyteorder(
                                                                       '<'))  # Unit Volt
        self.battery_percentage = self._volt_to_percent(self.battery)

    def _check_fletcher(self, fletcher):
        assert fletcher == b'\xaf\xbe\xad\xde', "Fletcher error!"

    def __str__(self):
        return "Temperature: " + str(self.temperature) + "\tLight: " + str(self.light) + "\tBattery: " + str(
            self.battery)

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
        self.hostTimeStamp = np.frombuffer(bin_data, dtype=np.dtype(np.uint64).newbyteorder('<'))

    def _check_fletcher(self, fletcher):
        assert fletcher == b'\xff\xff\xff\xff', "Fletcher error!"

    def translate(self):
        now = datetime.now()
        timestamp = int(1000000000 * datetime.timestamp(now))  # time stamp in nanosecond
        ts_str = hex(timestamp)
        ts_str = ts_str[2:18]
        host_ts = bytes.fromhex(ts_str)
        ID = b'\x1B'
        CNT = b'\x01'
        payload_len = b'\x10\x00'  # i.e. 0x0010
        device_ts = b'\x00\x00\x00\x00'
        fletcher = b'\xFF\xFF\xFF\xFF'
        self.raw_data = ID + CNT + payload_len + device_ts + host_ts + fletcher

    def __str__(self):
        return "Host timestamp: " + str(self.hostTimeStamp)

    def write_to_csv(self, recorder):
        recorder.write_data([self.timestamp])

    def push_to_lsl(self, outlet):
        outlet.push_sample([1])


class MarkerEvent(Packet):
    """Marker packet"""

    def __init__(self, timestamp, payload):
        super().__init__(timestamp, payload)
        self._convert(payload[:-4])
        self._check_fletcher(payload[-4:])

    def _convert(self, bin_data):
        self.marker_code = np.frombuffer(bin_data, dtype=np.dtype(np.uint16).newbyteorder('<'))[0]

    def _check_fletcher(self, fletcher):
        assert fletcher == b'\xaf\xbe\xad\xde', "Fletcher error!"

    def __str__(self):
        return "Event marker: " + str(self.marker_code)

    def write_to_file(self, recorder):
        recorder.write_data(np.array([self.timestamp, self.marker_code])[:,np.newaxis])

    def push_to_lsl(self, outlet):
        outlet.push_sample([self.marker_code])

    def push_to_dashboard(self, dashboard):
        pass


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
        fw_num = np.frombuffer(bin_data, dtype=np.dtype(np.uint16).newbyteorder('<'), count=1, offset=0)
        self.firmware_version = '.'.join([char for char in str(fw_num)[1:-1]])
        self.data_rate_info = 16000/(2**bin_data[2])
        self.adc_mask = bin(bin_data[3])
        print(self)

    def _check_fletcher(self, fletcher):
        assert fletcher == b'\xaf\xbe\xad\xde', "Fletcher error!"

    def __str__(self):
        return "Firmware version: " + self.firmware_version + "\tdata rate: " + str(self.data_rate_info)\
               + " sample per sec" + "\tADC mask: " + str(self.adc_mask)

    def write_to_file(self, recorder):
        recorder.write_data([self.timestamp, self.firmware_version, self.data_rate_info, self.adc_mask])

    def push_to_dashboard(self, dashboard):
        data = {'firmware_version': [self.firmware_version]}
        dashboard.doc.add_next_tick_callback(partial(dashboard.update_info, new=data))


class CommandRCV(Packet):
    """Command Status packet"""
    def __init__(self, timestamp, payload):
        super(CommandRCV, self).__init__(timestamp, payload)
        self._convert(payload[:-4])
        self._check_fletcher(payload[-4:])

    def _convert(self, bin_data):
        self.opcode = bin_data[0]
        pass

    def _check_fletcher(self, fletcher):
        assert fletcher == b'\xaf\xbe\xad\xde', "Fletcher error!"

    def __str__(self):
        return "an acknowledge message for command with this opcode has been received: " + str(self.opcode)


class CommandStatus(Packet):
    """Command Status packet"""
    def __init__(self, timestamp, payload):
        super(CommandStatus, self).__init__(timestamp, payload)
        self._convert(payload[:-4])
        self._check_fletcher(payload[-4:])

    def _convert(self, bin_data):
        self.opcode = bin_data[0]
        self.status = bin_data[5]

    def _check_fletcher(self, fletcher):
        assert fletcher == b'\xaf\xbe\xad\xde', "Fletcher error!"

    def __str__(self):
        return "Command status: " + str(self.status) + "\tfor command with opcode: " + str(self.opcode)


class CalibrationInfo(Packet):
    """Calibration Info packet"""
    def __init__(self, timestamp, payload):
        super(CalibrationInfo, self).__init__(timestamp, payload)
        self._convert(payload[:-4])
        self._check_fletcher(payload[-4:])

    def _convert(self, bin_data):
        slope = np.frombuffer(bin_data, dtype=np.dtype(np.uint16).newbyteorder('<'), count=1, offset=0)
        self.slope = slope * 10.0
        offset = np.frombuffer(bin_data, dtype=np.dtype(np.uint16).newbyteorder('<'), count=1, offset=2)
        self.offset = offset * 0.001

    def _check_fletcher(self, fletcher):
        assert fletcher == b'\xaf\xbe\xad\xde', "Fletcher error!"

    def __str__(self):
        return "calibration info: slope = " + str(self.slope) + "\toffset = " + str(self.offset)


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
    PACKET_ID.MARKER: MarkerEvent
}
