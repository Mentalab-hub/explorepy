import random
import time


class MockBtServer:
    '''
    Mocks the behaviour of BTSerialPortBinding.cpp as well as the Explore device
    '''

    # EEG98_USBC_HEADER = [b'\x96', b'\x00', b'\xB8', b'\x01', b'\x28', b'\x6E', b'\x04', b'\x00']
    # packet structure: [pid, count, payload_length, timestamp, [data + fletcher]*i]
    # EEG98_USBC_PID_ls = [b'\x96']
    EEG98_USBC_PID = b'\x96'
    # EEG98_USBC_PAYLOAD_LENGTH = [b'\xB8', b'\x01']
    EEG98_USBC_PAYLOAD_LENGTH = b'\xB8\x01'
    # EEG98_USBC_DATA_ROW_8_CH_ls = [b'\x00', b'\x00', b'\x80', b'\x00', b'\x00', b'\x80',
    #                            b'\x00', b'\x00', b'\x80', b'\x00', b'\x00', b'\x80',
    #                            b'\x00', b'\x00', b'\x80', b'\x00', b'\x00', b'\x80',
    #                            b'\x00', b'\x00', b'\x80', b'\x00', b'\x00', b'\x80']
    EEG98_USBC_DATA_ROW_8_CH = b'\x00\x00\x80\x00\x00\x80\x00\x00\x80\x00\x00\x80\x00\x00\x80\x00\x00\x80\x00\x00\x80' \
                               b'\x00\x00\x80'
    # EEG98_USBC_DEV_INFO_V2_PID_ls = [b'\x61']
    EEG98_USBC_DEV_INFO_V2_PID = b'\x61'
    # EEG98_USBC_DEV_INFO_V2_PAYLOAD_LENGTH_ls = [b'\x1D', b'\x00']
    EEG98_USBC_DEV_INFO_V2_PAYLOAD_LENGTH = b'\x1D\x00'
    # EEG98_USBC_DEV_INFO_V2_PCB_ls = [b'\x50', b'\x43', b'\x42', b'\x5F', b'\x33', b'\x30', b'\x33', b'\x5F',
    #                              b'\x38', b'\x30', b'\x31', b'\x5F', b'\x58', b'\x58', b'\x58', b'\x00']
    EEG98_USBC_DEV_INFO_V2_PCB_FW = b'\x50\x43\x42\x5F\x33\x30\x33\x5F\x38\x30\x31\x5F\x58\x58\x58\x00\x2D\x01'
    # ORN_PID_ls = [b'\x0D']
    ORN_PID = b'\x0D'
    # ORN_PAYLOAD_LENGTH_ls = [b'\x1A', b'\x00']
    ORN_PAYLOAD_LENGTH = b'\x1A\x00'
    ORN_DATA = b'\x1C\xFF\x1A\xF7\x30\xC1\x26\x00\xAA\xFF\xB5\xFF\xBF\x00\x1E\x01\x4D\x00'
    ENV_PID = b'\x13'
    ENV_PAYLOAD_LENGTH = b'\x0D\x00'
    ENV_TEMP_LIGHT_BATTERY = b'\x15\xBB\x0F\x15\x08'
    ENV_MEMORY = b'\xC8'
    # FLETCHER_ls = [b'\xAF', b'\xBE', b'\xAD', b'\xDE']
    CMD_RCV_PID = b'\xC0'
    CMD_RCV_PAYLOAD_LENGTH = b'\x0D\x00'
    CMD_STATUS_PID = b'\xC1'
    CMD_STATUS_PAYLOAD_LENGTH = b'\x0E\x00'
    FLETCHER = b'\xAF\xBE\xAD\xDE'

    def __init__(self):
        self.is_connected = False
        # status: 0 == connecting, 1 == command processing, 2 == streaming
        self.status = 0
        # self.timestamp = int(time.time() * 1000)  # time in ms
        # random between 100,000 and 200,000?
        self.timestamp = random.randint(100000, 200000)
        self.counter = 0
        self.exg_sr = 250
        self.orn_sr = 20
        self.env_sr = 1
        self.channel_mask = b'\xFF'
        self.buffer = None

    def sr_to_byte(self):
        sr_bits = b'\x06'
        if self.exg_sr == 250:
            sr_bits = b'\x06'
        elif self.exg_sr == 500:
            sr_bits = b'\x05'
        elif self.exg_sr == 1000:
            sr_bits = b'\x04'
        return sr_bits

    def generate_exg_packet(self):
        exg = self.EEG98_USBC_PID + \
              self.counter.to_bytes(1, byteorder='little') + \
              self.EEG98_USBC_PAYLOAD_LENGTH + \
              self.timestamp.to_bytes(4, byteorder='little')
        sr_bits = self.sr_to_byte()
        current_status = self.channel_mask + b'\x00' + sr_bits
        for i in range(0, 16):
            exg += current_status
            exg += self.EEG98_USBC_DATA_ROW_8_CH
        exg += self.FLETCHER

        return exg

    def generate_env_packet(self):
        env = self.ENV_PID
        env += self.counter.to_bytes(1, byteorder='little')
        env += self.ENV_PAYLOAD_LENGTH
        env += self.timestamp.to_bytes(4, byteorder='little')
        env += self.ENV_TEMP_LIGHT_BATTERY
        env += self.FLETCHER
        return env

    def generate_orn_packet(self):
        orn = self.ORN_PID
        orn += self.counter.to_bytes(1, byteorder='little')
        orn += self.ORN_PAYLOAD_LENGTH
        orn += self.timestamp.to_bytes(4, byteorder='little')
        orn += self.ORN_DATA
        orn += self.FLETCHER
        return orn

    def generate_dev_info_v2_packet(self):
        dev_info = self.EEG98_USBC_DEV_INFO_V2_PID
        dev_info += self.counter.to_bytes(1, byteorder='little')
        dev_info += self.EEG98_USBC_DEV_INFO_V2_PAYLOAD_LENGTH
        dev_info += self.timestamp.to_bytes(4, byteorder='little')
        dev_info += self.EEG98_USBC_DEV_INFO_V2_PCB_FW
        dev_info += self.sr_to_byte()
        dev_info += self.channel_mask
        dev_info += self.ENV_MEMORY
        return dev_info

    def generate_cmd_rcv(self, cmd_pid, cmd_ts):
        '''
        Generates a command received packet based on command opcode (pid) and timestamp received
        Args:
            cmd_pid: Received opcode
            cmd_ts: Received timestamp
        '''
        cmd_rcv = self.CMD_RCV_PID + \
                  self.counter.to_bytes(1, byteorder='little') + \
                  self.CMD_RCV_PAYLOAD_LENGTH + \
                  self.timestamp.to_bytes(4, byteorder='little')
        cmd_rcv += cmd_pid
        cmd_rcv += cmd_ts
        cmd_rcv += self.FLETCHER
        return cmd_rcv

    def generate_cmd_status(self, cmd_pid, cmd_ts):
        # Can't find list of possible status messages, can only find 0x01 in stream
        # (received and successfully executed? I think explorepy assumes that if it gets a
        # status packet with X opcode, X has been performed successfully
        cmd_status = self.CMD_STATUS_PID +\
                     self.counter.to_bytes(1, byteorder='little') +\
                     self.CMD_STATUS_PAYLOAD_LENGTH +\
                     self.timestamp.to_bytes(4, byteorder='little')
        cmd_status += cmd_pid
        cmd_status += cmd_ts
        cmd_status += b'\x01'
        cmd_status += self.FLETCHER
        return cmd_status

    def generate_command_packets(self, cmd_pid, cmd_ts):
        cmd = self.generate_cmd_rcv(cmd_pid, cmd_ts)
        self.counter = 0
        cmd += self.generate_dev_info_v2_packet()
        cmd += self.generate_cmd_status(cmd_pid, cmd_ts)
        self.counter = 1
        # Counter behaviour taken from a stream, is 0 for dev-info and cmd-status
        # and then starts counting up again
        return cmd

    def generate_packet_buffer(self, cmd=None, duration=1):
        '''
        Generates a second worth of packets (ExG, ORN, ENV)

        Args:
            duration(int): duration in seconds of bluetooth stream to generate

        Returns:
            A bytestring containing device packet data
        '''
        num_packets = int((self.exg_sr + self.orn_sr + 1) * duration)
        elapsed_time = int(60000 / num_packets)
        orn_pos = int(num_packets / self.orn_sr)
        env_pos = 10
        packet_buffer = b''
        start = 0

        # if self.status == 0:
        #    # Starting
        #    start = 1
        #    packet_buffer += self.generate_dev_info_v2_packet()
        # elif self.status == 1:
        #    # Command received
        #    start = 3
        #    packet_buffer += self.generate_command_packets()

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
        self.is_connected = True
        self.status = 0
        # self.timestamp = int(time.time() * 1000)  # time in ms
        self.counter = 0
        self.buffer = 0
        self.buffer = self.generate_dev_info_v2_packet() \
                      + self.generate_packet_buffer()
        return 0

    def Read(self, length):
        '''
        Reads from the mocked Bluetooth stream

        Args:
            length(int): Number of bytes to read

        Returns:
            A list of bytes
        '''
        raise NotImplementedError

    def process_incoming_data(self, data):
        pid = data[0]
        if pid == 160 or pid == 176:
            # 160 == Command(API2BCMD), 176 == Command(API2BCMD), 27 = TS (TS is sent at the start)
            ts = data[4:7]
            opcode = data[8]
            setting = data[9]
            if opcode == 161:
                # set sampling rate
                raise NotImplementedError
            elif opcode == 162:
                # set channel mask
                if pid == 160:
                    # API2BCMD
                    raise NotImplementedError
                elif pid == 176:
                    # API4BCMD
                    raise NotImplementedError
            elif opcode == 163:
                # format device memory
                raise NotImplementedError
            elif opcode == 164:
                # disable specific module
                raise NotImplementedError
            elif opcode == 165:
                # enable specific module
                raise NotImplementedError
            elif opcode == 166:
                # disable Z measurement (impedance?)
                raise NotImplementedError
            elif opcode == 167:
                # enable Z measurement (impedance?)
                raise NotImplementedError
            elif opcode == 168:
                # soft reset device
                raise NotImplementedError
        raise NotImplementedError

    def Write(self, data):
        '''
        Sends data to the mocked device
        '''
        self.process_incoming_data(data)
        '''
        From command.py:
            def translate(self):
        """Translates the command to binary array understandable by Explore device. """
        self.get_time()
        result = [self.pid.value, self.cnt, self.payload_length,
                  self.host_ts, self.opcode.value, self.param, self.fletcher]
        return b''.join(result)
        '''

    def Close(self):
        self.is_connected = False


if __name__ == '__main__':
    bt_interface = MockBtServer()
    now = time.time() * 1000
    # packet_buffer = bt_interface.generate_packet_buffer()
    bt_interface.Connect()
    diff = time.time() * 1000 - now
    print(f'Time diff: {diff}')
    print(len(bt_interface.buffer))
    # print(bt_interface.buffer)

    # TODO: change structure so that command packets are generated outside of the buffer generation
    # TODO: should clear the buffer and add a stream of packet (cmd_rcv, dev_info, cmd_status)
    # TODO: generate buffer should add to this
    # TODO: could also do this for device info?
