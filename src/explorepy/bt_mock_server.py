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
    EEG98_USBC_DEV_INFO_V2_PCB = b'\x50\x43\x42\x5F\x33\x30\x33\x5F\x38\x30\x31\x5F\x58\x58\x58\x00'
    # ORN_PID_ls = [b'\x0D']
    ORN_PID = b'\x0D'
    # ORN_PAYLOAD_LENGTH_ls = [b'\x1A', b'\x00']
    ORN_PAYLOAD_LENGTH = b'\x1A\x00'
    # FLETCHER_ls = [b'\xAF', b'\xBE', b'\xAD', b'\xDE']
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

    def generate_exg_packet(self):
        exg = self.EEG98_USBC_PID +\
              self.counter.to_bytes(1, byteorder='little') +\
              self.EEG98_USBC_PAYLOAD_LENGTH +\
              self.timestamp.to_bytes(4, byteorder='little')
        sr_bits = b'\x06'
        if self.exg_sr == 250:
            sr_bits = b'\x06'
        elif self.exg_sr == 500:
            sr_bits = b'\x05'
        elif self.exg_sr == 1000:
            sr_bits = b'\x04'
        current_status = self.channel_mask + b'\x00' + sr_bits
        for i in range(0, 16):
            exg += current_status
            exg += self.EEG98_USBC_DATA_ROW_8_CH
        exg += self.FLETCHER

        return exg

    def generate_env_packet(self):
        raise NotImplementedError

    def generate_orn_packet(self):
        raise NotImplementedError

    def generate_dev_info_v2_packet(self):
        raise NotImplementedError

    def generate_command_packets(self):
        raise NotImplementedError

    def process_incoming_data(self):
        raise NotImplementedError

    def generate_packet_buffer(self):
        '''
        Generates a second worth of packets (ExG, ORN, ENV)
        '''
        num_packets = self.exg_sr + self.orn_sr + 1
        elapsed_time = int(60000 / num_packets)
        orn_pos = int(num_packets / self.orn_sr)
        env_pos = 10
        packet_buffer = b''
        start = 0

        if self.status == 0:
            # Starting
            start = 1
            packet_buffer += self.generate_dev_info_v2_packet()
        elif self.status == 1:
            # Command received
            start = 3
            packet_buffer += self.generate_command_packets()

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

        self.status = 2
        return packet_buffer

    def Connect(self):
        self.is_connected = True
        self.status = 0
        # self.timestamp = int(time.time() * 1000)  # time in ms
        self.counter = 0
        self.buffer = 0
        self.buffer = self.generate_packet_buffer()
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

    def Write(self, data):
        '''
        Sends data to the mocked device
        '''
        self.process_incoming_data(data)

    def Close(self):
        self.is_connected = False


if __name__ == '__main__':
    bt_interface = MockBtServer()
    now = time.time_ns()
    packet_buffer = bt_interface.generate_packet_buffer()
    diff = time.time_ns() - now
    print(f'Time diff: {diff}')
    print(len(packet_buffer))
    print(packet_buffer)
