import time
from enum import Enum, auto

import numpy as np
from pyarrow import timestamp

from explorepy.packet import BleImpedancePacket, DeviceInfoBLE

class ClientState(Enum):
    DISCONNECTED = auto()
    CONNECTED = auto()
    STREAMING = auto()
    STOPPED = auto()


class CsvClient:
    def __init__(self):
        self.server = CsvServer(8)
        self._state = ClientState.DISCONNECTED

    # -------- state management --------
    def set_state(self, state: ClientState):
        if not isinstance(state, ClientState):
            raise ValueError("Invalid client state")

        self._state = state

    def get_state(self) -> ClientState:
        return self._state

    # -------- device-like API --------
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
            raise RuntimeError(
                f"Cannot start streaming from state {self._state}"
            )
        self.set_state(ClientState.STREAMING)

    def stop_streaming(self):
        if self._state == ClientState.STREAMING:
            self.set_state(ClientState.STOPPED)

    def read(self):
        if self._state != ClientState.STREAMING:
            if self._state == ClientState.CONNECTED:
                self.set_state(ClientState.STREAMING)
                device_info = self.server.read_device_info()
                device_info_packet = DeviceInfoMock(timestamp=self.server.ts, payload=None)
                device_info_packet.set_info(self.server.device_info_ble_32ch)
                return device_info_packet
        else:
            self.server.ts += self.server.time_period
            data = self.server.read_sample()
            eeg_packet = BleImpedancePacket(timestamp=self.server.ts, payload=None)
            eeg_packet.data = self.server.eeg_32
            return eeg_packet

    def write(self, bytes):
        # TODO parse the bytes and hadndle the state
        pass


class CsvServer:
    def __init__(self, channel_count):
        self.device_info_ble_8ch = {}
        self.eeg_32 = [
            -17.83, -400000.05, -400000.05, -400000.05,
            -400000.05, -400000.05, -400000.05, -400000.05,
            -400000.05, -400000.05, -400000.05, -400000.05,
            -400000.05, -400000.05, -400000.05, -400000.05,
            -400000.05, -400000.05, -400000.05, -400000.05,
            -400000.05, -400000.05, -400000.05, -400000.05,
            -400000.05, -400000.05, -400000.05, -400000.05,
            -400000.05, -400000.05, -400000.05, -400000.05
        ]
        self.eeg_32 = np.array(self.eeg_32).reshape(32, 1)
        self.device_info_ble_32ch = {'device_name': 'Explore_DABD', 'firmware_version': '9.1.0',
                                     'adc_mask': [1, 1, 1, 1, 1, 1, 1, 1], 'sampling_rate': 250, 'is_imp_mode': False,
                                     'board_id': 'PCB_304_801p2_X', 'memory_info': 1, 'max_online_sps': 250,
                                     'max_offline_sps': 2000}
        self.time_period = 0.004
        self.ts = 193.5597  # as dummy

    def read_sample(self):
        return self.ts, self.eeg_32

    def read_device_info(self):
        return self.device_info_ble_8ch

class DeviceInfoMock(DeviceInfoBLE):

    def __init__(self, timestamp, payload, time_offset=0):
        self.timestamp = timestamp

    def _convert(self, bin_data):
        pass
    def set_info(self, info):
        self.info = info

    def get_info(self):
        return self.info
