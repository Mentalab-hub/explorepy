import random


class MockBtServer:
    """
    Mocks the behaviour of BTSerialPortBinding.cpp as well as the Explore device
    """
    # packet structure: [pid, count, payload_length, timestamp, [data + fletcher]*i]
    EEG98_USBC_PID = b'\x96'
    EEG98_USBC_PAYLOAD_LENGTH = b'\xB8\x01'
    EEG98_USBC_DATA_ROW_8_CH = b'\x00\x00\x80\x00\x00\x80\x00\x00\x80\x00\x00\x80\x00\x00\x80\x00\x00\x80\x00\x00\x80' \
                               b'\x00\x00\x80'
    EEG98_USBC_DEV_INFO_V2_PID = b'\x61'
    EEG98_USBC_DEV_INFO_V2_PAYLOAD_LENGTH = b'\x1D\x00'
    EEG98_USBC_DEV_INFO_V2_PCB_FW = b'\x50\x43\x42\x5F\x33\x30\x33\x5F\x38\x30\x31\x5F\x58\x58\x58\x00\x2D\x01'

    ORN_PID = b'\x0D'
    ORN_PAYLOAD_LENGTH = b'\x1A\x00'
    ORN_DATA = b'\x1C\xFF\x1A\xF7\x30\xC1\x26\x00\xAA\xFF\xB5\xFF\xBF\x00\x1E\x01\x4D\x00'

    ENV_PID = b'\x13'
    ENV_PAYLOAD_LENGTH = b'\x0D\x00'
    ENV_TEMP_LIGHT_BATTERY = b'\x15\xBB\x0F\x15\x08'
    ENV_MEMORY = b'\xC8'

    CMD_RCV_PID = b'\xC0'
    CMD_RCV_PAYLOAD_LENGTH = b'\x0D\x00'
    CMD_STATUS_PID = b'\xC1'
    CMD_STATUS_PAYLOAD_LENGTH = b'\x0E\x00'

    FLETCHER = b'\xAF\xBE\xAD\xDE'

    def __init__(self):
        self.is_connected = False
        self.timestamp = random.randint(100000, 200000)
        self.counter = 0
        self.exg_sr = 250
        self.orn_sr = 20
        self.env_sr = 1
        self.channel_mask = b'\xFF'
        self.impedance_mode = False  # Not used currently
        self.exg = True
        self.orn = True
        self.env = True
        self.buffer = None

    def sr_to_byte(self):
        """
        Converts device sampling rate int to status byte.
        """
        if self.exg_sr == 250:
            return b'\x06'
        elif self.exg_sr == 500:
            return b'\x05'
        elif self.exg_sr == 1000:
            return b'\x04'
        raise ValueError('Server sr is invalid')

    def cmd_sr_to_sr(self, sr_byte):
        """
        Converts command sr byte to sampling rate int.

        Args:
            sr_byte (bytestring): SR parameter taken from an incoming command packet to set the sampling rate
        """
        if sr_byte == b'\x01':
            return 250
        elif sr_byte == b'\x02':
            return 500
        elif sr_byte == b'\x03':
            return 1000
        raise ValueError('Received sr byte is invalid')

    def popcount(self, as_bytes):
        """
        Calculates the hamming weight of a bytestring.

        Args:
            as_bytes (bytestring): Bytestring to calculate hamming weight for (i.e. b'\xFF', b'\xA0' etc.)
        """
        as_int = int.from_bytes(as_bytes, byteorder='little')
        counter = 0
        while as_int:
            as_int &= as_int - 1
            counter += 1
        return counter

    def generate_exg_packet(self):
        """
        Generates a single ExG packet of EEG98_USBC type (16 * (8 channels + status message) + fletcher). Returns empty
        bytestring if exg module is turned off and only returns as much data as expected from the channel mask.
        """
        # TODO: change this so it returns fake impedance values if imp mode is on
        if not self.exg:
            return b''

        exg = self.EEG98_USBC_PID + \
            self.counter.to_bytes(1, byteorder='little') + \
            self.EEG98_USBC_PAYLOAD_LENGTH + \
            self.timestamp.to_bytes(4, byteorder='little')
        sr_bits = self.sr_to_byte()
        current_status = self.channel_mask + b'\x00' + sr_bits
        num_bits = self.popcount(self.channel_mask)
        # Only add 24-bit ints of channel data for active channels (3 bytes * num_active_channels)
        row = self.EEG98_USBC_DATA_ROW_8_CH[:3 * num_bits]
        for i in range(0, 16):
            exg += current_status
            exg += row
        exg += self.FLETCHER

        return exg

    def generate_env_packet(self):
        """
        Generates a single environment packet. The values for temperature, light and battery are constant. Return empty
        bytestring in environment module is turned off.
        """
        if not self.env:
            return b''

        env = self.ENV_PID
        env += self.counter.to_bytes(1, byteorder='little')
        env += self.ENV_PAYLOAD_LENGTH
        env += self.timestamp.to_bytes(4, byteorder='little')
        env += self.ENV_TEMP_LIGHT_BATTERY
        env += self.FLETCHER
        return env

    def generate_orn_packet(self):
        """
        Generates a single orientation packet. Return empty bytestring if orientation module is turned off.
        """
        if not self.orn:
            return b''

        orn = self.ORN_PID
        orn += self.counter.to_bytes(1, byteorder='little')
        orn += self.ORN_PAYLOAD_LENGTH
        orn += self.timestamp.to_bytes(4, byteorder='little')
        orn += self.ORN_DATA
        orn += self.FLETCHER
        return orn

    def generate_dev_info_v2_packet(self):
        """
        Generates a single Dev_Info_V2 packet according to (constant) PCB and FW version strings, current SR, current
        channel mask and (constant) memory byte.
        """
        dev_info = self.EEG98_USBC_DEV_INFO_V2_PID
        dev_info += self.counter.to_bytes(1, byteorder='little')
        dev_info += self.EEG98_USBC_DEV_INFO_V2_PAYLOAD_LENGTH
        dev_info += self.timestamp.to_bytes(4, byteorder='little')
        dev_info += self.EEG98_USBC_DEV_INFO_V2_PCB_FW
        dev_info += self.sr_to_byte()
        dev_info += self.channel_mask
        dev_info += self.ENV_MEMORY
        dev_info += self.FLETCHER
        return dev_info

    def generate_cmd_rcv(self, cmd_pid, cmd_ts):
        """
        Generates a command received packet based on command opcode (pid) and timestamp received.

        Args:
            cmd_pid: Received opcode
            cmd_ts: Received timestamp
        """
        cmd_rcv = self.CMD_RCV_PID + \
            self.counter.to_bytes(1, byteorder='little') + \
            self.CMD_RCV_PAYLOAD_LENGTH + \
            self.timestamp.to_bytes(4, byteorder='little')
        cmd_rcv += cmd_pid.to_bytes(1, byteorder='little')
        cmd_rcv += cmd_ts
        cmd_rcv += self.FLETCHER
        return cmd_rcv

    def generate_cmd_status(self, cmd_pid, cmd_ts):
        """
        Generates a single command status packet according to a received PID and timestamp. The status byte is a
        constant 0x01.

        Args:
            cmd_pid (bytestring): PID as bytestring of the received command that this status packet reacts to
            cmd_ts (bytestring): Timestamp as bytestring of the received command that this status packet reacts to
        """
        cmd_status = self.CMD_STATUS_PID + \
            self.counter.to_bytes(1, byteorder='little') + \
            self.CMD_STATUS_PAYLOAD_LENGTH + \
            self.timestamp.to_bytes(4, byteorder='little')
        cmd_status += cmd_pid.to_bytes(1, byteorder='little')
        cmd_status += cmd_ts
        cmd_status += b'\x01'  # Can't find list of possible status messages, have only found b'\x01' in streams
        cmd_status += self.FLETCHER
        return cmd_status

    def generate_command_packets(self, cmd_pid, cmd_ts):
        """
        Generates a set of three packets: command received, dev_info_v2 and command status (in this order). Also resets
        the counter according to observed behaviour.

        Args:
            cmd_pid (bytestring): PID as bytestring of the received command that this status packet reacts to
            cmd_ts (bytestring): Timestamp as bytestring of the received command that this status packet reacts to
        """
        cmd = self.generate_cmd_rcv(cmd_pid, cmd_ts)
        self.counter = 0
        cmd += self.generate_dev_info_v2_packet()
        cmd += self.generate_cmd_status(cmd_pid, cmd_ts)
        self.counter = 1
        # Counter behaviour taken from a stream, is 0 for dev-info and cmd-status and then starts counting up again
        return cmd

    def generate_packet_buffer(self, cmd=None, duration=1):
        """
        Generates a second worth of packets (ExG, ORN, ENV).

        Args:
            duration(int): duration in seconds of bluetooth stream to generate

        Returns:
            A bytestring containing device packet data
        """
        num_packets = int((self.exg_sr + self.orn_sr + 1) * duration)
        elapsed_time = int(60000 / num_packets)
        orn_pos = int(num_packets / self.orn_sr)
        env_pos = 10
        packet_buffer = b''
        start = 0

        for i in range(start, num_packets):
            if i % orn_pos == 0:
                packet_buffer += self.generate_orn_packet()
            elif i == env_pos:
                packet_buffer += self.generate_env_packet()
            else:
                packet_buffer += self.generate_exg_packet()
            if self.counter < 255:
                self.counter += 1
            else:
                self.counter = 0
            self.timestamp += elapsed_time

        # self.status = 2
        return packet_buffer

    def Connect(self):
        """
        Connects to the (fake) device and resets counter and buffer. All other settings remain unchanged.
        """
        self.is_connected = True
        self.counter = 0
        self.buffer = self.generate_dev_info_v2_packet() + self.generate_packet_buffer()
        return 0

    def Read(self, length):
        """
        Reads from the mocked Bluetooth stream.

        Args:
            length(int): Number of bytes to read

        Returns:
            A list of bytes
        """
        if len(self.buffer) <= length:
            self.buffer += self.generate_packet_buffer()

        read_data = self.buffer[:length]
        self.buffer = self.buffer[length:]

        return read_data

    def process_incoming_data(self, data):
        """
        Takes incoming command packet and sets internal settings accordingly.
        Soft reset is not currently implemented.

        Args:
            data (bytestring): An entire, incoming command packet

        Returns:
            opcode (bytestring): The opcode of the received command
            ts (bytestring): The timestamp of the received command
        """
        pid = data[0]
        if pid == 160 or pid == 176:
            # 160 == Command(API2BCMD), 176 == Command(API2BCMD), 27 = TS (TS is sent at the start)
            # cnt = data[1]
            # payload_length = data[2:4]
            ts = data[4:8]
            opcode = data[8]
            param = data[9]
            # fletcher = data[10:]
            if opcode == 161:
                # set sampling rate
                self.exg_sr = self.cmd_sr_to_sr(param)
            elif opcode == 162:
                # set channel mask
                if pid == 160:
                    # API2BCMD
                    self.channel_mask = param
                elif pid == 176:
                    # API4BCMD
                    self.channel_mask = param
            elif opcode == 163:
                # format device memory
                self.impedance_mode = False
                self.exg_sr = 250
                self.channel_mask = b'\xFF'
                self.exg = True
                self.orn = True
                self.env = True
                self.counter = 0
                self.buffer = None
            elif opcode == 164:
                # disable specific module
                if param == 1:
                    self.env = False
                elif param == 2:
                    self.orn = False
                elif param == 3:
                    self.exg = False
            elif opcode == 165:
                # enable specific module
                if param == 1:
                    self.env = True
                elif param == 2:
                    self.orn = True
                elif param == 3:
                    self.exg = True
            elif opcode == 166:
                # disable Z measurement
                self.impedance_mode = False
            elif opcode == 167:
                # enable Z measurement
                self.impedance_mode = True
            elif opcode == 168:
                # soft reset device
                self.impedance_mode = False
                self.exg_sr = 250
                self.channel_mask = b'\xFF'
                self.exg = True
                self.orn = True
                self.env = True
            return opcode, ts
        return None, None

    def Write(self, data):
        """
        Sends data to the mocked device.

        Args:
            data (bytestring): The command packet to be sent to the (fake) device.
        """
        opcode, ts = self.process_incoming_data(data)
        if opcode is not None and ts is not None:
            self.buffer = self.generate_command_packets(opcode, ts)
            self.buffer += self.generate_packet_buffer()

    def Close(self):
        self.is_connected = False
        self.impedance_mode = False
        self.counter = 0
