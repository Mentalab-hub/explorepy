# -*- coding: utf-8 -*-
"""
A module providing classes for Explore device configuration
"""
import abc
import logging
import time
from datetime import datetime
from enum import Enum


logger = logging.getLogger(__name__)


class CommandID(Enum):
    """Command ID enum class"""
    API2BCMD = b'\xA0'
    API4BCMD = b'\xB0'


class OpcodeID(Enum):
    """Op Code ID class"""
    CMD_SPS_SET = b'\xA1'
    CMD_CH_SET = b'\xA2'
    CMD_MEM_FORMAT = b'\xA3'
    CMD_REC_TIME_SET = b'\xB1'
    CMD_MODULE_DISABLE = b'\xA4'
    CMD_MODULE_ENABLE = b'\xA5'
    CMD_ZM_DISABLE = b'\xA6'
    CMD_ZM_ENABLE = b'\xA7'
    CMD_SOFT_RESET = b'\xA8'
    CMD_TEST_SIG = b'\xAA'


class Result(Enum):
    """Results of the command execution"""
    API_CMD_SUCCESSFUL = b'\x01'
    API_CMD_ILLEGAL = b'\x02'
    API_CMD_FAILED = b'\x00'
    API_CMD_NA = b'\xFF'


class DeviceConfiguration:
    """Device Configuration Class"""

    def __init__(self, bt_interface):
        """
        Args:
            bt_interface: Bluetooth interface
        """
        self._bt_interface = bt_interface
        self._last_ack_message = None
        self._last_status_message = None

    def get_device_info(self):
        """Get device information including adc mask, sampling rate and firmware version."""
        raise NotImplementedError

    def change_setting(self, command):
        """Change the settings of the device based on the input command

        Args:
            command (explorepy.command.Command): Command to be executed

        Returns:
              bool: True if the command has been successfully executed.
        """
        self._last_ack_message = None
        self._last_status_message = None
        self._send_command(command)
        logger.info("waiting for ack and status messages...")
        cmd_received = False
        for _ in range(10):
            if not self._last_ack_message:
                time.sleep(1)
            elif int2bytearray(self._last_ack_message.opcode, 1) == command.opcode.value:
                logger.info("Command has been received by Explore.")
                cmd_received = True
                self._last_ack_message = None
                break
        if cmd_received:
            for _ in range(10):
                if not self._last_status_message:
                    time.sleep(1)
                elif int2bytearray(self._last_status_message.opcode, 1) == command.opcode.value:
                    logger.info("Command has been successfully executed by the device.")
                    return True

        if not cmd_received:
            logger.warning("Command has not been received by the device. Try again.")
        return False

    def update_ack(self, packet):
        """Update ack message"""
        self._last_ack_message = packet

    def update_cmd_status(self, packet):
        """Update status message"""
        self._last_status_message = packet

    def _send_command(self, command):
        """Send a command to the device

        Args:
            command (explorepy.command.Command): Command object
            socket (socket): Bluetooth socket

        """
        logger.info("Sending the command: " + str(command))
        self._bt_interface.send(command.translate())
        logger.info("Command has been sent successfully.")

    def send_timestamp(self):
        ts = HostTimeStamp()
        self._send_command(ts)


class HostTimeStamp:
    """Host timestamp data packet"""

    def __init__(self):
        self.pid = b'\x1B'
        self.cnt = b'\x01'
        self.payload_len = int2bytearray(16, 2)
        self.device_ts = b'\x00\x00\x00\x00'
        self.host_ts = None
        self.fletcher = b'\xFF\xFF\xFF\xFF'

    def translate(self):
        """Translate content to bytearray"""
        timestamp = int(time.time() + time.localtime().tm_gmtoff)
        self.host_ts = int2bytearray(timestamp, 8)
        return self.pid + self.cnt + self.payload_len + self.device_ts + self.host_ts + self.fletcher

    def __str__(self):
        return "Host timestamp"


class Command:
    """An abstract base class for Explore command packet"""

    def __init__(self):
        self.pid = None
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
        result = [self.pid.value, self.cnt, self.payload_length,
                  self.host_ts, self.opcode.value, self.param, self.fletcher]
        return b''.join(result)

    def get_time(self):
        """Gets the current machine time based on unix format and fills the corresponding field.

        Args:

        """
        now = datetime.now()
        timestamp = int(1000000000 * datetime.timestamp(now))  # time stamp in nanosecond
        self.host_ts = int2bytearray(timestamp, 4)

    @abc.abstractmethod
    def __str__(self):
        """prints the appropriate info about the command. """


class Command2B(Command):
    """An abstract base class for Explore 2 Byte command data length packets"""

    def __init__(self):
        super().__init__()
        self.pid = CommandID.API2BCMD
        self.payload_length = int2bytearray(10, 2)

    @abc.abstractmethod
    def __str__(self):
        """prints the appropriate info about the command. """


class Command4B(Command):
    """An abstract base class for Explore 4 Byte command data length packets"""

    def __init__(self):
        super().__init__()
        self.pid = CommandID.API4BCMD
        self.payload_length = int2bytearray(12, 2)

    @abc.abstractmethod
    def __str__(self):
        """prints the appropriate info about the command. """


class SetSPS(Command2B):
    """Set the sampling rate of ExG device"""

    def __init__(self, sps_rate):
        """
        Args:
            sps_rate (int): sampling rate per seconds. It should be one of these values: 250, 500 or 1000
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

    def __str__(self):
        return "Set sampling rate command"


class MemoryFormat(Command2B):
    """Format device memory"""

    def __init__(self):
        super().__init__()
        self.opcode = OpcodeID.CMD_MEM_FORMAT
        self.param = b'\x00'

    def __str__(self):
        return "Format memory command"


class ModuleDisable(Command2B):
    """Module disable command"""

    def __init__(self, module_name):
        """

        Args:
            module_name (str): Module name to be disabled. Options: "EXG", "ORN", "ENV"
        """
        super().__init__()
        self.opcode = OpcodeID.CMD_MODULE_DISABLE
        if module_name == "ENV":
            self.param = b'\x01'
        elif module_name == "ORN":
            self.param = b'\x02'
        elif module_name == "EXG":
            self.param = b'\x03'

    def __str__(self):
        return "Module disable command"


class ModuleEnable(Command2B):
    """Module enable command"""

    def __init__(self, module_name):
        """
        Args:
            module_name (str): Module name to be disabled. Options: "EXG", "ORN", "ENV"
        """
        super().__init__()
        self.opcode = OpcodeID.CMD_MODULE_ENABLE
        if module_name == "ENV":
            self.param = b'\x01'
        elif module_name == "ORN":
            self.param = b'\x02'
        elif module_name == "EXG":
            self.param = b'\x03'

    def __str__(self):
        return "Module enable command"


class ZMeasurementDisable(Command2B):
    """Enables Z measurement mode"""

    def __init__(self):
        super().__init__()
        self.opcode = OpcodeID.CMD_ZM_DISABLE
        self.param = b'\x00'

    def __str__(self):
        return "Impedance measurement disable command"


class ZMeasurementEnable(Command2B):
    """Enables Z measurement"""

    def __init__(self):
        super().__init__()
        self.opcode = OpcodeID.CMD_ZM_ENABLE
        self.param = b'\x00'

    def __str__(self):
        return "Impedance measurement enable command"


class SetCh(Command2B):
    """Change channel mask command"""

    def __init__(self, ch_mask):
        """

        Args:
            ch_mask (int): ExG channel mask. It should be integers between 1 and 255.
        """
        super().__init__()
        self.opcode = OpcodeID.CMD_CH_SET
        if 1 <= ch_mask <= 255:
            self.param = bytes([ch_mask])
        else:
            raise ValueError("Invalid input")

    def __str__(self):
        return "Channel set command"


class SoftReset(Command2B):
    """Reset the setting of the device."""

    def __init__(self):
        super().__init__()
        self.opcode = OpcodeID.CMD_SOFT_RESET
        self.param = b'\x00'

    def __str__(self):
        return "Reset command"


class SetChTest(Command2B):
    """Enable test signal"""
    def __init__(self, ch_mask):
        """
        Args:
            ch_mask (int): ExG channel mask on which the test signal should be activated.
                            It should be integers between 1 and 255 (equivalent of binary representation of the mask).
        """
        super().__init__()
        self.opcode = OpcodeID.CMD_TEST_SIG
        if 0 <= ch_mask <= 255:
            self.param = bytes([ch_mask])
        else:
            raise ValueError("Invalid input")

    def __str__(self):
        return "Test signal activation command"


COMMAND_CLASS_DICT = {
    CommandID.API2BCMD: Command2B,
    CommandID.API4BCMD: Command4B
}


def int2bytearray(data, num_bytes):
    """Gets an integer and convert it to a byte array with specified number of bytes

    Args:
        data: integer
        num_bytes: number of bytes

    Returns:
        bytearray
    """
    x_str = hex(data)
    x_str = x_str[2:(2 * num_bytes + 2)]
    i = len(x_str)
    if i < (num_bytes * 2):
        for j in range(0, 2 * num_bytes - i):
            x_str = '0' + x_str
    output = bytes.fromhex(x_str)

    # Change byte order for MCU
    if num_bytes == 2:
        output = bytes([output[1], output[0]])
    return output
