# -*- coding: utf-8 -*-
"""A module for bluetooth connection"""

import ctypes
import logging
import multiprocessing as mp
import os
import queue
import sys
import threading
import time
from copy import copy
from multiprocessing import shared_memory

import serial
from serial.tools import list_ports

from explorepy._exceptions import DeviceNotFoundError

logger = logging.getLogger(__name__)


class SerialStream:
    """Responsible for Connecting and reconnecting explore devices via usb interface"""

    def __init__(self, device_name):
        """Initialize Bluetooth connection"""
        self.device_name = device_name
        self.device_manager = None
        self.bt_sdk = None
        self.state = SharedState(self.device_name)
        self.device_process: mp.Process = None
        self.page = 0
        self.offset = 0
        self.idx = 0

    def connect(self):
        """Connect to the device and return the socket

        Returns:
            socket (bluetooth.socket)
        """
        logger.info("Trying to connect")
        self.device_process = mp.Process(target=device_process_main, args=(self.state.clone(),))
        self.device_process.start()
        while not self.state.is_connected:
            time.sleep(0.05)

    def reconnect(self):
        """Reconnect to the last connected device

        Generally speaking this is not necessary for USB stream, but we keep it as placeholder
        in case it is needed in future
        """
        raise NotImplementedError

    def disconnect(self):
        """Disconnect from the device"""
        self.state.usb_stop_flag = True
        self.state.is_connected = False
        self.state.close_buffers(self.page)
        self.device_process.join(timeout=2)

    def read(self, n_bytes):
        """Read n_bytes from the socket

        Args:
            n_bytes (int): number of bytes to be read

        Returns:
            list of bytes
        """
        chunk = bytearray()
        if self.idx % 8000 == 0:
            writer_page = self.state.buffer.page
            reader_page = self.page
            print(
                f"Writer page :: {writer_page} - Reader page :: {reader_page} - offset :: {writer_page - reader_page}"
            )
        self.idx += 1

        try:
            if not self.state.is_connected:
                raise Exception("not connected")

            while len(chunk) < n_bytes:
                try:
                    chunk, self.page, self.offset = self.state.buffer.read(
                        n_bytes, self.page, self.offset
                    )
                except IndexError:
                    time.sleep(0.002)
            return chunk
        except Exception as error:
            logger.warning(f"Serial Client - Got error or read request {type(error)}: {error}")
            print("maziar Got error or read request: {}".format(error))

    def send(self, data):
        """Send data to the device

        Args:
            data (bytearray): Data to be sent
        """
        if not self.device_process.is_alive():
            raise Exception("Device Process is not alive, cannot communicate with device")
        self.state.write_queue.put(data)


def get_correct_com_port(device_name):
    """Returns correct COM/tty port for usb connection
    Args: device name: the name of the device to connect to
    """
    fletcher = b"\xaf\xbe\xad\xde"

    ports = list(list_ports.comports())
    for p in ports:
        if p.vid == 0x0483 and p.pid == 0x5740:
            serial_port = serial.Serial(port=p.device, baudrate=115200, timeout=2)
            # stop stream
            cmd = b"\xe5" * 10 + fletcher
            serial_port.write(cmd)
            time.sleep(0.1)
            # read all the stream data
            serial_port.readall()

            get_name_cmd = b"\xc6" * 10 + fletcher
            serial_port.write(get_name_cmd)
            data = serial_port.read(24)
            if len(data) == 0:
                # incompatible explore device, continue connection
                print("got data as zero")
                serial_port.close()
                return p.device
            name = data[8:-4].decode("utf-8", errors="ignore")
            if name == device_name:
                return p.device
    raise DeviceNotFoundError()


class SharedBuffer:
    def __init__(
        self,
        page=None,
        buffer_len=None,
        create: bool = False,
    ):
        self._page = page if page else mp.Value("i", 0)
        self._len = buffer_len if buffer_len else mp.Value("i", 0)
        self.shm: shared_memory.SharedMemory = None

        if sys.platform == "win32":
            self._shm_list: list[shared_memory.SharedMemory] = []

        self._create_shm(create)

    @property
    def len(self):
        return self._len.value

    @property
    def page(self):
        return self._page.value

    def clone(self):
        return SharedBuffer(
            page=self._page,
            buffer_len=self._len,
        )

    def append(self, byte):
        """Append a new byte to the shared buffer

        Args:
            byte (byte): byte to be appended
        """
        write_off = copy(self.len)
        if write_off >= self.shm.size:
            self._page.value += 1
            old_shd = self.shm

            try:
                self._create_shm(True)
            except FileExistsError:
                self._create_shm(False)
            except Exception as e:
                logger.error(
                    f"Device Process - Got exception while trying to create a new shared memory block {e}"
                )
                self._page.value -= 1
                return

            self._len.value = 0
            write_off = 0

            if sys.platform != "win32":
                old_shd.close()

        self.shm.buf[write_off] = byte
        write_off += 1
        self._len.value = write_off

    def read(self, num_bytes: int, page: int, offset: int) -> tuple[bytes, int, int]:
        """Read a number of bytes from the buffer

        Args:
            num_bytes (int): Number of bytes to read
            page (int): The current page
            offset (int): Offset at which to start reading

        Returns:
            buffer (bytes): list of bytes
            page (int): Next page
            offset (int): Next starting offset

        Raises:
            IndexError: if either the page or offset are out of bound
        """
        end = offset + num_bytes
        if end < self.shm.size and end < self.len and self.page >= page:
            buf = bytes(self.shm.buf[offset:end])
            assert len(buf) == num_bytes
            return buf, page, end
        elif end >= self.shm.size and self.page > page:
            old_shd = self.shm

            try:
                page += 1
                self._create_shm(False, page)
            except Exception as e:
                logger.error(
                    f"Serial Client - Got exception while trying to connect to next shared memory block {e}"
                )
                page -= 1
                raise IndexError("Index out of bound")

            end = num_bytes - (self.shm.size - offset)
            buf = bytes(old_shd.buf[offset : self.shm.size]) + bytes(self.shm.buf[0:end])
            assert len(buf) == num_bytes

            if sys.platform != "win32":
                old_shd.close()
                old_shd.unlink()

            return buf, page, end

        raise IndexError("Index out of bound")

    def close(self):
        if sys.platform != "win32":
            self.shm.close()

    def unlink(self):
        if sys.platform == "win32":
            for shm in self._shm_list:
                shm.close()

        self.shm.unlink()

    def _buffer_name(self, page: int | None = None):
        return f"explorepy-serial-{page if page else self.page}"

    def _create_shm(self, create: bool, page: int | None = None):
        if sys.platform == "win32" and create and self.shm is not None:
            self._shm_list.append(self.shm)

        self.shm = shared_memory.SharedMemory(
            name=self._buffer_name(page),
            create=create,
            size=2048 * 100,
        )


class SharedState:
    def __init__(
        self,
        device_name,
        buffer: SharedBuffer | None = None,
        write_queue=None,
    ):
        self.device_name = device_name
        self.block_read_size = 2048
        self.buffer = buffer if buffer else SharedBuffer(create=True)
        self.write_queue = write_queue if write_queue else mp.Queue()
        self._is_connected = mp.Value(ctypes.c_bool, False)
        self._usb_stop_flag = mp.Value(ctypes.c_bool, False)

    @property
    def is_connected(self):
        return self._is_connected.value

    @is_connected.setter
    def is_connected(self, value):
        self._is_connected.value = value

    @property
    def usb_stop_flag(self):
        return self._usb_stop_flag.value

    @usb_stop_flag.setter
    def usb_stop_flag(self, value):
        self._usb_stop_flag.value = value

    def clone(self):
        state = SharedState(
            self.device_name,
            buffer=self.buffer.clone(),
            write_queue=self.write_queue,
        )
        state._is_connected = self._is_connected
        state._usb_stop_flag = self._usb_stop_flag
        return state

    def close_buffers(self, start_page):
        if sys.platform == "win32":
            self.buffer.unlink()
        else:
            current_page = start_page
            while current_page <= self.buffer.page:
                shd = SharedBuffer(page=current_page)
                shd.close()
                shd.unlink()
                current_page += 1


def device_process_main(state: SharedState):
    try:
        p = DeviceProcess(state)
        p.run()
    except Exception as e:
        logger.critical(f"Device Process - Got exception: {e}")
        exit()


class DeviceProcess:
    def __init__(self, state: SharedState):
        self.state = state
        self.lock = threading.Lock()
        self.writer_thread = None
        self.presence_thread = None

        fletcher = b"\xaf\xbe\xad\xde"
        for _ in range(5):
            try:
                self.port = get_correct_com_port(self.state.device_name)
                self.comm_manager = serial.Serial(port=self.port, baudrate=115200, timeout=0.5)
                self.comm_manager.timeout = 0.5

                # stop stream
                cmd = b"\xe5" * 10 + fletcher
                self.comm_manager.write(cmd)
                time.sleep(1)
                self.comm_manager.readall()

                self.reader_thread = threading.Thread(
                    target=self.read_serial_in_chunks, daemon=True
                )
                self.reader_thread.start()

                # start stream
                fletcher = b"\xaf\xbe\xad\xde"
                cmd = b"\xe4" * 10 + fletcher
                self.comm_manager.write(cmd)

                self.state.is_connected = True
                return
            except PermissionError:
                # do nothing here as this comes from posix
                pass
            except serial.serialutil.SerialException:
                logger.info(
                    "Permission denied on serial port access, please run this command via"
                    "terminal: sudo chmod 777 {}".format(self.port)
                )
                raise serial.serialutil.SerialException(
                    "Permission error on USB port access. Please set the permission"
                    " temporarily via the terminal to allow port access:\n\nchmod 7"
                    "77 <port_name>\n\nAlternatively, set up a udev rule following "
                    "the ExplorePy documentation:\n\nhttps://explorepy.readthedocs."
                    "io/en/latest/installation.html#set-up-usb-streaming-in-linux"
                )
            except Exception as error:
                state.is_connected = False
                logger.info(
                    "Got an exception while connecting to the device: {} of type: {}".format(
                        error, type(error)
                    )
                )
                logger.debug("trying to connect again as tty port is not visible yet")
                logger.warning("Could not connect; Retrying in 2s...")
                time.sleep(2)

        state.is_connected = False
        raise DeviceNotFoundError(
            "Could not find the device! Please turn on the device, wait a few seconds and connect to"
            "serial port before starting ExplorePy"
        )

    def run(self):
        logger.debug(f"Device Process is running on pid {os.getpid()}")

        self.writer_thread = threading.Thread(target=self.write_serial, daemon=True)
        self.writer_thread.start()

        self.presence_thread = threading.Thread(target=self.check_presence, daemon=True)
        self.presence_thread.start()

        while True:
            if self.state.usb_stop_flag or not self.state.is_connected:
                break
            time.sleep(0.1)

        if sys.platform == "win32":
            self.state.close_buffers()

        self.state.is_connected = False
        self.comm_manager.cancel_read()
        self.comm_manager.close()
        self.reader_thread.join(timeout=2)
        self.writer_thread.join(timeout=2)
        self.presence_thread.join(timeout=2)

    def read_serial_in_chunks(self):
        """Reads data in fixed-size chunks from the serial port until stopped."""
        logger.info("Start reading bytes from serial connection")
        while not self.state.usb_stop_flag:
            try:
                data = self.comm_manager.read(self.state.block_read_size)
                if data is not None:
                    for byte in data:
                        self.state.buffer.append(byte)
            except Exception as e:
                logger.debug(f"Got Exception in USB read method: {e}")

        logger.debug("Stopping USB data retrieval thread")

    def write_serial(self):
        while True:
            try:
                data = self.state.write_queue.get()
                if data:
                    with threading.Lock():
                        self.comm_manager.write(data)
            except queue.Empty:
                pass

    def check_presence(self):
        while True:
            open_ports = [p.device for p in list(serial.tools.list_ports.comports())]
            if self.port not in open_ports:
                self.state.is_connected = False
                break
            time.sleep(0.1)
