from explorepy.packet import Packet, PACKET_ID, PACKET_CLASS_DICT, CommandRCV, CommandStatus
from datetime import datetime
import abc
from enum import Enum
import time


class OpcodeID(Enum):
    CMD_SPS_SET = b'\xA1'
    CMD_CH_SET = b'\xA2'
    CMD_MEM_FORMAT = b'\xA3'
    CMD_REC_TIME_SET = b'\xB1'
    CMD_MODULE_DISABLE = b'\xA4'
    CMD_MODULE_ENABLE = b'\xA5'


class DeliveryState(Enum):
    NOT_SENT = 0
    SENT_NO_ACK_RCVD = 1
    SENT_ACK_RCVD = 2


class Result(Enum):
    API_CMD_SUCCESSFUL = b'\x01'
    API_CMD_ILLEGAL = b'\x02'
    API_CMD_FAILED = b'\x00'
    API_CMD_NA = b'\xFF'


class Command:
    """An abstract base class for Explore command packet"""
    def __init__(self):
        self.ID
        self.cnt
        self.payload_length
        self.host_ts
        self.opcode
        self.param
        self.fletcher = b'\xaf\xbe\xad\xde'

        self.delivery_state
        self.result

    def translate(self):
        """translate the command to binary array understandable by Explore device. """
        return self.ID.value + self.cnt + self.payload_length + self.host_ts + \
               self.opcode.value + self.param + self.fletcher

    def get_time(self):
        """
        gets the current machine time based on unix format and fills the corresponding field.

        Args:

        """
        now = datetime.now()
        timestamp = int(1000000000 * datetime.timestamp(now))  # time stamp in nanosecond
        self.host_ts = self.int2bytearray(timestamp, 4)

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
        x_str = x_str[2:(2 * n)]
        out = bytes.fromhex(x_str)

        # Change byte order for MCU
        if n == 2:
            out = bytes([out[1], out[0]])
        return out

    @abc.abstractmethod
    def __str__(self):
        """prints the appropriate info about the command. """
        pass


class Command2B(Command):
    """An abstract base class for Explore 2 Byte command data length packets"""

    def __init__(self):
        super().__init__()
        self.ID = PACKET_ID.API2BCMD
        self.payload_length = self.int2bytearray(10, 2)


class Command4B(Command):
    """An abstract base class for Explore 4 Byte command data length packets"""

    def __init__(self):
        super().__init__()
        self.ID = PACKET_ID.API4BCMD
        self.payload_length = self.int2bytearray(12, 2)


class SetSPS(Command2B):
    def __init__(self, sps_rate):
        """
        Gets the desired rate and initializes the packet

        Args:
            sps_rate (int): sampling rate per seconds. It should be one of these values: 250, 500, 1000
        """
        super().__init__()
        self.opcode = OpcodeID.CMD_SPS_SET
        if sps_rate == 250:
            self.param = b'\x01'
        elif sps_rate == 500:
            self.param = b'\x02'
        elif sps_rate == 1000:
            self.param = b'\x03'
        else:
            raise ValueError("Invalid input")

        self.param = sps_rate
        self.get_time()
        self.delivery_state = DeliveryState.NOT_SENT
        self.result = Result.NA

    def __str__(self):
        return "set sampling rate command!!!"


class MemoryFormat(Command2B):
    def __init__(self):
        """
        Format device memory
        """
        super().__init__()
        self.opcode = OpcodeID.CMD_MEM_FORMAT


class ModuleDisable(Command2B):
    def __init__(self, module_name):
        """

        Args:
            module_name (str): Module name to be disabled. Options: "EEG", "ORN", "ENV"
        """
        super().__init__()
        self.opcode = OpcodeID.CMD_MODULE_DISABLE
        if module_name == "ENV":
            self.param = b'x01'
        elif module_name == "ORN":
            self.param = b'x02'
        elif module_name == "EEG":
            self.param = b'x03'


class ModuleEnable(Command2B):
    def __init__(self, module_name):
        """

        Args:
            module_name (str): Module name to be disabled. Options: "EEG", "ORN", "ENV"
        """
        super().__init__()
        self.opcode = OpcodeID.CMD_MODULE_ENABLE
        if module_name == "ENV":
            self.param = b'x01'
        elif module_name == "ORN":
            self.param = b'x02'
        elif module_name == "EEG":
            self.param = b'x03'


def send_command(command, socket):
    """

    Args:
        command (explorepy.command.Command):
        socket (socket):
        parser (explorepy.Parser)

    Returns:

    """
    print("Sending the message...")
    socket.send(command.translate())
    print(" Message Sent :)")
