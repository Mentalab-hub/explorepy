from datetime import datetime
import abc
from enum import Enum


class COMMAND_ID(Enum):
    API2BCMD = b'\xA0'
    API4BCMD = b'\xB0'


class OpcodeID(Enum):
    CMD_SPS_SET = b'\xA1'
    CMD_CH_SET = b'\xA2'
    CMD_MEM_FORMAT = b'\xA3'
    CMD_REC_TIME_SET = b'\xB1'
    CMD_MODULE_DISABLE = b'\xA4'
    CMD_MODULE_ENABLE = b'\xA5'
    CMD_ZM_DISABLE = b'\xA6'
    CMD_ZM_ENABLE = b'\xA7'
    CMD_SOFT_RESET = b'\xA8'


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
        self.ID = None
        self.cnt = b'\x00'
        self.payload_length = None
        self.host_ts = None
        self.opcode = None
        self.param = None
        self.fletcher = b'\xaf\xbe\xad\xde'

        self.delivery_state = None
        self.result = None

    def translate(self):
        """Translates the command to binary array understandable by Explore device. """
        self.get_time()
        return self.ID.value + self.cnt + self.payload_length + self.host_ts + \
               self.opcode.value + self.param + self.fletcher

    def get_time(self):
        """Gets the current machine time based on unix format and fills the corresponding field.

        Args:

        """
        now = datetime.now()
        timestamp = int(1000000000 * datetime.timestamp(now))  # time stamp in nanosecond
        self.host_ts = self.int2bytearray(timestamp, 4)

    @abc.abstractmethod
    def get_ack(self):
        """Gets the acknowledge from the device. """
        pass

    @abc.abstractmethod
    def get_status(self):
        """Gets the status from the device. """
        pass

    def int2bytearray(self, x, n):
        """Gets an integer and convert it to a byte array with specified number of bytes

        Args:
            x: integer
            n: number of bytes

        Returns:
            bytearray
        """
        x_str = hex(x)
        x_str = x_str[2:(2 * n+2)]
        i = len(x_str)
        if i < (n*2):
            for j in range(0, 2*n-i):
                x_str = '0' + x_str
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
        self.ID = COMMAND_ID.API2BCMD
        self.payload_length = self.int2bytearray(10, 2)


class Command4B(Command):
    """An abstract base class for Explore 4 Byte command data length packets"""

    def __init__(self):
        super().__init__()
        self.ID = COMMAND_ID.API4BCMD
        self.payload_length = self.int2bytearray(12, 2)


class SetSPS(Command2B):
    def __init__(self, sps_rate):
        """Gets the desired rate and initializes the packet

        Args:
            sps_rate (int): sampling rate per seconds. It should be one of these values: 250 or 500
        """
        super().__init__()
        self.opcode = OpcodeID.CMD_SPS_SET
        if sps_rate == 250:
            self.param = b'\x01'
        elif sps_rate == 500:
            self.param = b'\x02'
        else:
            raise ValueError("Invalid input")

        self.get_time()
        self.delivery_state = DeliveryState.NOT_SENT

    def __str__(self):
        return "Set sampling rate command"


class MemoryFormat(Command2B):
    def __init__(self):
        """Format device memory"""
        super().__init__()
        self.opcode = OpcodeID.CMD_MEM_FORMAT
        self.param = b'\x00'

    def __str__(self):
        return "Format memory command"


class ModuleDisable(Command2B):
    def __init__(self, module_name):
        """Disable module class

        Args:
            module_name (str): Module name to be disabled. Options: "EEG", "ORN", "ENV"
        """
        super().__init__()
        self.opcode = OpcodeID.CMD_MODULE_DISABLE
        if module_name == "ENV":
            self.param = b'\x01'
        elif module_name == "ORN":
            self.param = b'\x02'
        elif module_name == "EEG":
            self.param = b'\x03'

    def __str__(self):
        return "Module disable command"


class ModuleEnable(Command2B):
    def __init__(self, module_name):
        """Enable module command class

        Args:
            module_name (str): Module name to be disabled. Options: "EEG", "ORN", "ENV"
        """
        super().__init__()
        self.opcode = OpcodeID.CMD_MODULE_ENABLE
        if module_name == "ENV":
            self.param = b'\x01'
        elif module_name == "ORN":
            self.param = b'\x02'
        elif module_name == "EEG":
            self.param = b'\x03'

    def __str__(self):
        return "Module enable command"


class ZmeasurementDisable(Command2B):
    def __init__(self):
        """Enables Z measurement"""
        super().__init__()
        self.opcode = OpcodeID.CMD_ZM_DISABLE
        self.param = b'\x00'

    def __str__(self):
        return "Impedance measurement disable command"


class ZmeasurementEnable(Command2B):
    def __init__(self):
        """Enables Z measurement"""
        super().__init__()
        self.opcode = OpcodeID.CMD_ZM_ENABLE
        self.param = b'\x00'

    def __str__(self):
        return "Impedance measurement enable command"


class SetCh(Command2B):
    def __init__(self, ch_mask):
        """Gets the desired rate and initializes the packet

        Args:
            ch_mask (int): ExG channel mask. It should be integers between 1 and 255.
        """
        super().__init__()
        self.opcode = OpcodeID.CMD_CH_SET
        if 1 <= ch_mask <= 255:
            self.param = bytes([ch_mask])
        else:
            raise ValueError("Invalid input")
        self.get_time()
        self.delivery_state = DeliveryState.NOT_SENT

    def __str__(self):
        return "Channel set command"


class SoftReset(Command2B):
    def __init__(self):
        """Reset the Device."""
        super().__init__()
        self.opcode = OpcodeID.CMD_SOFT_RESET
        self.param = b'\x00'

    def __str__(self):
        return "Reset command"


def send_command(command, socket):
    """Send a command to the device

    Args:
        command (explorepy.command.Command): Command object
        socket (socket): Bluetooth socket

    """
    print("Sending the message...")

    socket.send(command.translate())
    print(" Message Sent.")


COMMAND_CLASS_DICT = {
    COMMAND_ID.API2BCMD: Command2B,
    COMMAND_ID.API4BCMD: Command4B
}
