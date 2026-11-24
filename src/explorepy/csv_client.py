import os
import time
from collections import deque
from enum import (
    Enum,
    auto
)

import numpy as np
from pylsl import local_clock

from explorepy.packet import (
    BleImpedancePacket,
    CalibrationInfo,
    CommandRCV,
    CommandStatus,
    DeviceInfoBLE,
    OrientationV2
)


# Byte position of the opcode in a translated command (pid, cnt, payload_len, host_ts, opcode, ...)
CMD_OPCODE_INDEX = 8

# Simulated impedance calibration (typical device values, already scaled like CalibrationInfo: raw slope * 10,
# raw offset * 0.01)
IMP_CALIB_SLOPE = 263000.0
IMP_CALIB_OFFSET = 118.0


class ClientState(Enum):
    DISCONNECTED = auto()
    CONNECTED = auto()
    STREAMING = auto()
    STOPPED = auto()


class PacketSize(Enum):
    EEG_8 = 40
    EEG_32 = 112
    EEG_64 = 208
    ORN = 50
    DEVICE_INFO = 38
    CMD = 14
    CALIB_INFO = 20


class CsvClient:
    def __init__(self, channel_count, file_path: str, device_name: str = None):
        """
        Args:
            channel_count (int): Number of ExG channels in the CSV file (8, 32 or 64)
            file_path (str): Path of the ExG CSV file to stream
            device_name (str): Name reported in the device info packet
        """
        if file_path is None:
            raise ValueError("The csv interface needs an ExG CSV file: pass it with Explore.connect(file_path=...)")
        self.file_name = os.path.split(file_path)[-1]

        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        self.server = CsvServer(
            channel_count=channel_count,
            csv_path=file_path,
            loop=True
        )
        if device_name is not None:
            self.server.device_info = dict(self.server.device_info, device_name=device_name)
        self._state = ClientState.DISCONNECTED
        # packets a real device would send in response to events (calibration info, command ack/status)
        self._pending_packets = deque()

    def set_state(self, state: ClientState):
        if not isinstance(state, ClientState):
            raise ValueError("Invalid client state")
        self._state = state

    def get_state(self) -> ClientState:
        return self._state

    def connect(self):
        if self._state != ClientState.DISCONNECTED:
            return False
        self.set_state(ClientState.CONNECTED)
        return True

    def disconnect(self):
        self.set_state(ClientState.DISCONNECTED)
        return True

    def start_streaming(self):
        if self._state != ClientState.CONNECTED:
            raise RuntimeError(f"Cannot start streaming from state {self._state}")
        self.set_state(ClientState.STREAMING)

    def stop_streaming(self):
        if self._state == ClientState.STREAMING:
            self.set_state(ClientState.STOPPED)

    def read(self):
        if self._state != ClientState.STREAMING:
            if self._state == ClientState.CONNECTED:
                self.set_state(ClientState.STREAMING)
                device_info_packet = DeviceInfoMock(timestamp=self.server.ts, payload=None)
                device_info_packet.packet_size = PacketSize.DEVICE_INFO
                device_info_packet.set_info(self.server.device_info)
                # a real device sends its impedance calibration info after connecting
                self._pending_packets.append(CalibrationInfoMock(self.server.ts, IMP_CALIB_SLOPE, IMP_CALIB_OFFSET))
                return device_info_packet

        if self._pending_packets:
            return self._pending_packets.popleft()

        self.server.tick += 1
        if self.server.tick % 5 == 0:
            orn_packet = OrientationMock(timestamp=self.server.ts, payload=None)
            orn_packet.set_data(self.server.orn_value)
            orn_packet.packet_size = PacketSize.ORN
            return orn_packet

        self.server.ts += self.server.time_period
        sleep_time = self.server.ts - local_clock()
        if sleep_time > 0:
            time.sleep(sleep_time)
        eeg_packet = BleImpedancePacket(timestamp=self.server.ts, payload=None)
        try:
            eeg_packet.data, eeg_packet.timestamp = self.server.read_sample()
        except StopIteration:
            self.set_state(ClientState.STOPPED)
            return None
        eeg_packet.packet_size = self.server.packet_size
        return eeg_packet

    def write(self, bytes):
        pass

    def send(self, data):
        """Emulates the device's response to a command: an ack followed by a successful status packet"""
        opcode = data[CMD_OPCODE_INDEX]
        self._pending_packets.append(CommandRCVMock(self.server.ts, opcode))
        self._pending_packets.append(CommandStatusMock(self.server.ts, opcode))


class CsvServer:
    def __init__(self, channel_count: int, csv_path: str, loop: bool = True):
        if channel_count not in (8, 32, 64):
            raise ValueError("Only 8 or 32 channels supported")

        self.channel_count = channel_count
        self.loop = loop
        self.csv_data = np.loadtxt(csv_path, delimiter=',', skiprows=1)  # skip row 0

        self.csv_ts = self.csv_data[:, 0]
        self.csv_data = self.csv_data[:, 1:]

        if self.csv_data.shape[1] != channel_count:
            print('######################', self.csv_data.shape)
            raise ValueError(
                f"CSV has {self.csv_data.shape[1]} columns, "
                f"expected {channel_count}"
            )

        self.row_idx = 0
        self.num_rows = self.csv_data.shape[0]
        self.device_info_ble_64ch = {
            'device_name': 'Explore_XXXX',
            'firmware_version': '6.6.9',
            'adc_mask': [1] * 64,
            'sampling_rate': 250,
            'is_imp_mode': False,
            'board_id': 'PCB_305_801p2_X',
            'memory_info': 1,
            'max_online_sps': 250,
            'max_offline_sps': 8000
        }
        self.device_info_ble_32ch = {
            'device_name': 'Explore_DABD',
            'firmware_version': '9.6.9',
            'adc_mask': [1] * 32,
            'sampling_rate': 250,
            'is_imp_mode': False,
            'board_id': 'PCB_304_801p2_X',
            'memory_info': 1,
            'max_online_sps': 250,
            'max_offline_sps': 2000
        }

        self.device_info_ble_8ch = {
            'device_name': 'Explore_AAAQ',
            'firmware_version': '7.6.9',
            'adc_mask': [1] * 8,
            'sampling_rate': 250,
            'is_imp_mode': False,
            'board_id': 'PCB_303_801E_XX',
            'memory_info': 1,
            'max_online_sps': 1000,
            'max_offline_sps': 8000
        }

        if channel_count == 8:
            self.device_info = self.device_info_ble_8ch
            self.packet_size = PacketSize.EEG_8
        elif channel_count == 32:
            self.device_info = self.device_info_ble_32ch
            self.packet_size = PacketSize.EEG_32
        elif channel_count == 64:
            self.device_info = self.device_info_ble_64ch
            self.packet_size = PacketSize.EEG_64

        self.fs = self.device_info['sampling_rate']
        self.time_period = np.round(1 / self.fs, 3)
        self.ts = local_clock()
        self.tick = 0

        self.orn_value = [
            5.002, -3.904, 1001.01, 420.0, -70.0, 210.0,
            103.36, 804.08, -532.0, -0.0023,
            -0.0028, 0.0371, 0.9993
        ]

    def read_sample(self):
        if self.row_idx >= self.num_rows:
            if not self.loop:
                raise StopIteration("End of CSV reached")
            self.row_idx = 0

        ts = self.csv_ts[self.row_idx]
        sample = self.csv_data[self.row_idx]
        self.row_idx += 1

        return sample.reshape(self.channel_count, 1), ts

    def read_device_info(self):
        return self.device_info


class DeviceInfoMock(DeviceInfoBLE):
    def __init__(self, timestamp, payload, time_offset=0):
        self.timestamp = timestamp

    def _convert(self, bin_data):
        pass

    def set_info(self, info):
        self.info = info

    def get_info(self):
        return self.info


class OrientationMock(OrientationV2):
    """Orientation data packet"""

    def __init__(self, timestamp, payload, time_offset=0):
        self.timestamp = timestamp

    def _convert(self, bin_data):
        pass

    def get_data(self, srate=None):
        return [self.timestamp], self.data

    def set_data(self, data):
        self.data = data


class CommandRCVMock(CommandRCV):
    """Command acknowledge packet"""

    def __init__(self, timestamp, opcode):
        self.timestamp = timestamp
        self.opcode = opcode
        self.packet_size = PacketSize.CMD


class CommandStatusMock(CommandStatus):
    """Command status packet (always successful)"""

    def __init__(self, timestamp, opcode):
        self.timestamp = timestamp
        self.opcode = opcode
        self.status = 1
        self.packet_size = PacketSize.CMD


class CalibrationInfoMock(CalibrationInfo):
    """Impedance calibration info packet"""

    def __init__(self, timestamp, slope, offset):
        self.timestamp = timestamp
        self.slope = [slope] * self.channels
        self.offset = [offset] * self.channels
        self.packet_size = PacketSize.CALIB_INFO
