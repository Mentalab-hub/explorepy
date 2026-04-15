import time
from enum import Enum, auto
import numpy as np
from explorepy.packet import BleImpedancePacket, DeviceInfoBLE, OrientationV1, OrientationV2
from pylsl import local_clock
from typing_extensions import override


class ClientState(Enum):
    DISCONNECTED = auto()
    CONNECTED = auto()
    STREAMING = auto()
    STOPPED = auto()

class PacketSize(Enum):
    EEG_8 = 40
    EEG_32 = 112
    ORN = 50
    DEVICE_INFO = 38

class CsvClient:
    def __init__(self, channel_count):
        file_path = "../../explorepy/tests/sample_data/"
        file_name = "test_" + str(channel_count) + ".csv"
        self.server = server = CsvServer(
    channel_count=channel_count,
    csv_path=file_path + file_name,
    loop=True
)
        self._state = ClientState.DISCONNECTED

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
                return device_info_packet


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
            eeg_packet.data = self.server.read_sample()
        except StopIteration:
            self.set_state(ClientState.STOPPED)
            return None
        eeg_packet.packet_size = self.server.packet_size
        return eeg_packet

    def write(self, bytes):
        pass

import numpy as np
from pylsl import local_clock


class CsvServer:
    def __init__(self, channel_count: int, csv_path: str, loop: bool = True):
        if channel_count not in (8, 32):
            raise ValueError("Only 8 or 32 channels supported")

        self.channel_count = channel_count
        self.loop = loop
        self.csv_data = np.loadtxt(csv_path, delimiter=',', skiprows=1) # skip row 0

        self.csv_data = self.csv_data[:, 1:]
        if self.csv_data.shape[1] != channel_count:
            print('######################', self.csv_data.shape)
            raise ValueError(
                f"CSV has {self.csv_data.shape[1]} columns, "
                f"expected {channel_count}"
            )

        self.row_idx = 0
        self.num_rows = self.csv_data.shape[0]
        self.device_info_ble_32ch = {
            'device_name': 'Explore_DABD',
            'firmware_version': '9.6.9',
            'adc_mask': [1] * 8,
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

        self.device_info = (
            self.device_info_ble_8ch
            if channel_count == 8
            else self.device_info_ble_32ch
        )

        self.fs = self.device_info['sampling_rate']
        self.time_period = np.round(1 / self.fs, 3)
        self.ts = local_clock()
        self.tick = 0

        self.packet_size = (
            PacketSize.EEG_8 if channel_count == 8 else PacketSize.EEG_32
        )

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

        sample = self.csv_data[self.row_idx]
        self.row_idx += 1

        return sample.reshape(self.channel_count, 1)

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
