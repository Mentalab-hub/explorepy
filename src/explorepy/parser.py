# -*- coding: utf-8 -*-
import numpy as np
import struct
from explorepy.packet import PACKET_ID, PACKET_CLASS_DICT, TimeStamp, EEG, Environment, CommandRCV, CommandStatus,\
                                Orientation, DeviceInfo, Disconnect, MarkerEvent, CalibrationInfo
from explorepy.filters import Filter
import copy
import time


def generate_packet(pid, timestamp, bin_data):
    """Generates the packets according to the pid

    Args:
        pid (int): Packet ID
        timestamp (int): Timestamp
        bin_data: Binary dat

    Returns:
        Packet
    """

    if pid in PACKET_CLASS_DICT:
        packet = PACKET_CLASS_DICT[pid](timestamp, bin_data)
    else:
        print("Unknown Packet ID:" + str(pid))
        print("Length of the binary data:", len(bin_data))
        packet = None
    return packet


class Parser:
    def __init__(self, bp_freq=None, notch_freq=None, socket=None, fid=None):
        """Parser class for explore device

        Args:
            socket (BluetoothSocket): Bluetooth Socket (Should be None if fid is provided)
            fid (file object): File object for reading data (Should be None if socket is provided)
            bp_freq (tuple): Tuple of cut-off frequencies of bandpass filter (low cut-off frequency, high cut-off frequency)
            notch_freq (int): Notch filter frequency (50 or 60 Hz)
        """
        self.socket = socket
        self.fid = fid
        self.dt_int16 = np.dtype(np.int16).newbyteorder('<')
        self.dt_uint16 = np.dtype(np.uint16).newbyteorder('<')
        self.time_offset = None
        self.events = []
        self.start_timer = None
        if bp_freq is not None:
            assert bp_freq[0] < bp_freq[1], "High cut-off frequency must be larger than low cut-off frequency"
            self.bp_freq = bp_freq
            self.apply_bp_filter = True
        else:
            self.apply_bp_filter = False
            self.bp_freq = (0, 100)  # dummy values
        self.notch_freq = notch_freq
        self._filter = None
        self.firmware_version = None
        self.fs = None
        self.adc_mask = None
        self.n_chan = None
        self.imp_calib_info = {}
        self.calibre_set = None
        self.init_set = None
        self.ED_prv = None
        self.theta = 0.
        self.axis = np.array([0, 0, -1])
        self.matrix = np.identity(4)
        packet = None
        while not isinstance(packet, DeviceInfo):
            packet = self.parse_packet(mode=None)
        self.signal_dc = np.zeros((self.n_chan,), dtype=np.float)

    @property
    def filter(self):
        if self._filter is None:
            self._init_filters()
        return self._filter

    def parse_packet(self, mode="print", recorders=None, outlets=None, dashboard=None):
        """Reads and parses a package from a file or socket

        Args:
            mode (str): logging mode {'print', 'record', 'lsl', 'visualize', None}
            recorders (tuple): Tuple of recorder objects (ExG_recorder, ORN_recorder, Marker_recorder)
            outlets (tuple): Tuple of lsl StreamOutlet (orientation_outlet, EEG_outlet, marker_outlet)
            dashboard (Dashboard): Dashboard object for visualization
        Returns:
            packet object
        """
        pid = struct.unpack('B', self.read(1))[0]
        cnt = self.read(1)[0]
        payload = struct.unpack('<H', self.read(2))[0]
        timestamp = struct.unpack('<I', self.read(4))[0]

        # Timestamp conversion
        if self.time_offset is None:
            self.time_offset = timestamp * .0001
            timestamp = 0
            self.start_timer = time.time()
        else:
            timestamp = timestamp * .0001 - self.time_offset   # Timestamp unit is .1 ms
            # print(f"Packet time: {timestamp:.5f} \t diff time: {(time.time()-self.start_timer - timestamp):.5f}")

        payload_data = self.read(payload - 4)
        packet = generate_packet(pid, timestamp, payload_data)

        if isinstance(packet, DeviceInfo):
            self.firmware_version = packet.firmware_version
            self.fs = int(packet.data_rate_info)
            self.adc_mask = packet.adc_mask
            self.n_chan = self.adc_mask.count('1')
        if mode == "print":
            print(packet)

        elif mode == "record":
            assert isinstance(recorders, tuple), "Invalid writer objects!"
            if isinstance(packet, Orientation):
                packet.write_to_file(recorders[1])
            elif isinstance(packet, EEG):
                packet.write_to_file(recorders[0])
            elif isinstance(packet, MarkerEvent):
                packet.write_to_file(recorders[2])
            for event in self.events:
                event.write_to_file(recorders[2])
            self.events = []
            # elif isinstance(packet, DeviceInfo):
            #     packet.write_to_file(recorders[3])

        elif mode == "lsl":
            if isinstance(packet, Orientation):
                packet.push_to_lsl(outlets[0])
            elif isinstance(packet, EEG):
                packet.push_to_lsl(outlets[1])
            elif isinstance(packet, MarkerEvent):
                packet.push_to_lsl(outlets[2])

        elif mode == "visualize":
            if isinstance(packet, EEG):
                if self.notch_freq:
                    packet.apply_notch_filter(exg_filter=self.filter)
                if self.apply_bp_filter:
                    packet.apply_bp_filter(exg_filter=self.filter)
                # remove DC
                # n_samples = packet.data.shape[1]
                # for column in range(n_samples):
                #     self.signal_dc = (self.bp_freq[0] / (self.fs*0.5)) * packet.data[:, column] + (
                #             1 - (self.bp_freq[0] / (self.fs*0.5))) * self.signal_dc
                #     packet.data[:, column] = packet.data[:, column] - self.signal_dc
            if isinstance(packet, Orientation):
                packet = self.compute_NED(packet)
            packet.push_to_dashboard(dashboard)

        elif mode == "listen":
            if isinstance(packet, CommandRCV):
                print(packet)
            elif isinstance(packet, CommandStatus):
                print(packet)
            elif isinstance(packet, CalibrationInfo):
                print(packet)

        elif mode == "debug":
            if isinstance(packet, EEG):
                print(packet)

        elif mode == "impedance":
            if isinstance(packet, EEG):
                if self.notch_freq:
                    packet.apply_notch_filter(exg_filter=self.filter)
                if self.apply_bp_filter:
                    temp_packet = copy.deepcopy(packet)
                    temp_packet.apply_bp_filter_noise(exg_filter=self.filter)
                    mag = np.ptp(temp_packet.data, axis=1)
                    self.imp_calib_info['noise_level'] = mag
                    packet.apply_bp_filter(exg_filter=self.filter)
                packet.push_to_imp_dashboard(dashboard, self.imp_calib_info)
            elif isinstance(packet, Environment) | isinstance(packet, DeviceInfo):
                packet.push_to_dashboard(dashboard)

        elif mode == "calibrate":
            assert isinstance(recorders, tuple), "Invalid csv writer objects!"
            if isinstance(packet, Orientation):
                packet.write_to_csv(recorders[0])
            elif isinstance(packet, MarkerEvent):
                packet.write_to_file(recorders[1])

        elif mode == "initialize":
            if isinstance(packet, Orientation):
                th = np.zeros(3)
                T_init = np.zeros((3, 3))
                D = packet.acc / (np.dot(packet.acc, packet.acc) ** 0.5)
                [kx, ky, kz, mx_offset, my_offset, mz_offset] = self.calibre_set
                packet.mag[0] = kx * (packet.mag[0] - mx_offset)
                packet.mag[1] = ky * (packet.mag[1] - my_offset)
                packet.mag[2] = kz * (packet.mag[2] - mz_offset)
                E = -1*np.cross(D, packet.mag)
                E = E / (np.dot(E, E) ** 0.5)
                # here you can find an estimation of actual north from packet.mag, it is perpendicular to D and still
                # co-planar with D and mag, somehow reducing error
                N = -1*np.cross(E, D)
                N = N / (np.dot(N, N) ** 0.5)
                T_init = np.column_stack((E,N,D))
                N_init = np.matmul(np.transpose(T_init), N)
                E_init = np.matmul(np.transpose(T_init), E)
                D_init = np.matmul(np.transpose(T_init), D)
                self.init_set = [T_init, N_init, E_init, D_init]
                self.ED_prv = [E, D]
        return packet

    def read(self, n_bytes):
        """Read n_bytes from socket or file

        Args:
            n_bytes (int): number of bytes to be read

        Returns:
            list of bytes
        """
        if self.socket is not None:
            try:
                byte_data = self.socket.recv(n_bytes)
            except ValueError as e:
                raise ConnectionAbortedError(e.__str__())
        elif not self.fid.closed:
            byte_data = self.fid.read(n_bytes)
        else:
            raise ValueError("File has been closed unexpectedly!")
        if len(byte_data) != n_bytes:
            raise ValueError("Number of received bytes is less than expected")
            # TODO: Create a specific exception for this case
        return byte_data

    def set_marker(self, marker_code):
        if type(marker_code) is not int:
            raise TypeError('Marker code must be an integer!')
        if 0 <= marker_code <= 7:
            raise ValueError('Marker code value is not valid')

        self.events.append(MarkerEvent(timestamp=time.time()-self.start_timer,
                                       payload=bytearray(struct.pack('<H', marker_code) + b'\xaf\xbe\xad\xde')))

    def _init_filters(self):
        if self.apply_bp_filter or self.notch_freq:
            self._filter = Filter(l_freq=self.bp_freq[0],
                                  h_freq=self.bp_freq[1],
                                  line_freq=self.notch_freq,
                                  sampling_freq=self.fs)

    def compute_NED(self, packet):
        [kx, ky, kz, mx_offset, my_offset, mz_offset] = self.calibre_set
        D_prv = self.ED_prv[1]
        E_prv = self.ED_prv[0]
        acc = packet.acc
        acc = acc / (np.dot(acc, acc) ** 0.5)
        gyro = packet.gyro * 1.745329e-5 #radian per second
        #mag = np.array([0, 0, 0])
        packet.mag[0] = kx * (packet.mag[0] - mx_offset)
        packet.mag[1] = ky * (packet.mag[1] - my_offset)
        packet.mag[2] = kz * (packet.mag[2] - mz_offset)
        mag = packet.mag
        D = acc
        dD = D-D_prv
        da = np.cross(D_prv, dD)
        E = -1*np.cross(D, mag)
        E = E / (np.dot(E, E) ** 0.5)
        dE = E-E_prv
        dm = np.cross(E_prv, dE)
        dg = 0.05 * gyro
        dth = -0.95 * dg + 0.025 * da + 0.025 * dm
        D = D_prv + np.cross(dth, D_prv)
        D = D / (np.dot(D, D) ** 0.5)
        Err = np.dot(D, E)
        D_tmp = D - 0.5*Err*E
        E_tmp = E - 0.5*Err*D
        D = D_tmp / (np.dot(D_tmp, D_tmp) ** 0.5)
        E = E_tmp / (np.dot(E_tmp, E_tmp) ** 0.5)
        N = -1*np.cross(E, D)
        N = N / (np.dot(N, N) ** 0.5)

        '''
        If you comment this block it will give you the absolute orientation based on {East,North,Up} coordinate system.
        If you keep this block of code it will give you the relative orientation based on itial state of the device. so
        It is important to keep the device steady, so that the device can capture the initial direction properly.
        '''
        ##########################
        T = np.zeros((3,3))
        T_init = self.init_set[0]
        N_init = self.init_set[1]
        E_init = self.init_set[2]
        D_init = self.init_set[3]
        T = np.column_stack((E, N, D))
        T_test = np.matmul(T, T_init.transpose())
        N = np.matmul(T_test.transpose(), N_init)
        E = np.matmul(T_test.transpose(), E_init)
        D = np.matmul(T_test.transpose(), D_init)
        ##########################
        matrix = np.identity(4)
        matrix[0][0] = E[0]
        matrix[0][1] = N[0]
        matrix[0][2] = D[0]

        matrix[1][0] = E[1]
        matrix[1][1] = N[1]
        matrix[1][2] = D[1]

        matrix[2][0] = E[2]
        matrix[2][1] = N[2]
        matrix[2][2] = D[2]
        N = N / (np.dot(N, N) ** 0.5)
        E = E / (np.dot(E, E) ** 0.5)
        D = D / (np.dot(D, D) ** 0.5)
        packet.NED = np.array([N, E, D])
        self.ED_prv = [E, D]
        self.matrix = self.matrix*0.9+0.1*matrix
        [theta, rot_axis] = packet.compute_angle(matrix=self.matrix)
        self.theta = self.theta*0.9+0.1*theta
        packet.theta = self.theta
        self.axis = self.axis*0.9+0.1*rot_axis
        packet.rot_axis = self.axis
        return packet
