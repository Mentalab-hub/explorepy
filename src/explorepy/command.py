from explorepy.packet import Packet, PACKET_ID, PACKET_CLASS_DICT
from datetime import datetime
import abc
from enum import IntEnum
import time
from explorepy.bt_client import BtClient

class OpcodeID(IntEnum):
    CMD_SPS_SET = 0xA1
    CMD_CH_SET = 0XA2
    CMD_MEM_FORMAT = 0XA3
    CMD_REC_TIME_SET = 0XB1
    CMD_MODULE_DISABLE = 0XA4
    CMD_MODULE_ENABLE = 0XA5


class DeliveryState(IntEnum):
    NOT_SENT = 0
    SENT_NO_ACK_RCVD = 1
    SENT_ACK_RCVD = 2


class Result(IntEnum):
    API_CMD_SUCCESSFUL = 0x01,
    API_CMD_ILLEGAL = 0x02,
    API_CMD_FAILED = 0x00
    API_CMD_NA = 0xFF


class Command(Packet):
    """An abstract base class for Explore command packet"""
    def __init__(self):
        self.ID
        self.cnt
        self.payload_length
        self.host_ts
        self.opcode
        self.param
        self.delivery_state
        self.result
        self.raw_data

    @abc.abstractmethod
    def translate(self):
        """translate the command to binary array understandable by Explore device. """
        pass

    def issue(self):
        """issue a command and gets the status from the device. """
        self.get_time()
        self.translate()
        self.send()
        self.listen()

    @abc.abstractmethod
    def send(self):
        pass

    @abc.abstractmethod
    def listen(self):
        pass

    def get_time(self):

        """
        gets the current machine time based on unix format and fills the corresponding field.

        Args:

        """
        now = datetime.now()
        timestamp = int(1000000000 * datetime.timestamp(now))  # time stamp in nanosecond
        self.host_ts = timestamp

    @abc.abstractmethod
    def get_ack(self):
        """issue a command and gets the acknowledge from the device. """
        pass

    @abc.abstractmethod
    def get_status(self):
        """issue a command and gets the status from the device. """
        pass

    def int2bytearray(self, x, n):
        """
        gets an integer and convert it to a byte array with specified number of bytes
        Args:
            x: integer
            n: number of bytes

        Returns:
            bytearray
        """
        x_str = hex(x)
        x_str = x_str[2:(2*n)]
        return bytes.fromhex(x_str)

    @abc.abstractmethod
    def __str__(self):
        """prints the appropriate info about the command. """
        pass


class Command2B(Command):
    """An abstract base class for Explore 2 Byte command data length packets"""
    def __init__(self):
        super().__init__()
        self.ID = PACKET_ID.API2BCMD
        self.payload_length = 10
    pass



class Command4B(Command):
    """An abstract base class for Explore 4 Byte command data length packets"""
    def __init__(self):
        super().__init__()
        self.ID = PACKET_ID.API4BCMD
        self.payload_length = 12
    pass


class SetSPS(Command2B):
    @abc.abstractmethod
    def __init__(self, sps_rate, cnt):
        """
        Gets the desired rate and initializes the packet

        Args:
            sampling rate per seconds. It should be one of these values: 250, 500, 1000
        """
        super().__init__()
        self.opcode = OpcodeID.CMD_SPS_SET
        self.cnt = cnt
        assert (sps_rate==250) or (sps_rate==500) or (sps_rate==1000), "Value of sps_rate should be 250, 500 or 1000!"
        self.param = sps_rate
        self.host_ts = None
        self.delivery_state = DeliveryState.NOT_SENT
        self.result = Result.NA

    def translate(self):
        # just update rate, cnt, host_ts in the following raw_data
        self.raw_data = b'\xA0\x00\x0A\x00\x00\x00\x00\x00\xA1\x01\xaf\xbe\xad\xde'
        # Updating param
        if self.param==500:
            self.raw_data[-5] = b'\x02'
        if self.param==1000:
            self.raw_data[-5] = b'\x03'
        # Updating cnt
        self.raw_data[1] = self.int2bytearray(self.cnt, 1)
        # Updating host_ts
        self.raw_data[4:8] = self.int2bytearray(self.host_ts, 4)

    def __str__(self):
        return "set sampling rate command!!!"


