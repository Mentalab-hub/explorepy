# -*- coding: utf-8 -*-
"""A module for bluetooth connection"""
import logging
import threading
import time
from collections import deque

import serial
from serial.tools import list_ports

from explorepy._exceptions import DeviceNotFoundError
from pylsl import local_clock

logger = logging.getLogger(__name__)

class CountingSemaphore:
    def __init__(self, count=1):
        self.count = count
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.max_semaphore = 0

    def acquire(self):
        with self.condition:
            while self.count == 0:
                self.condition.wait()  # Wait until a resource is available
            self.count -= 1

    def release(self):
        with self.condition:
            self.count += 1
            self.condition.notify()  # Wake up one waiting thread
            if self.max_semaphore < self.count:
                self.max_semaphore = self.count
                if self.max_semaphore > 100:
                    print("max_semaphore: " + str(self.max_semaphore))


class SerialStream:
    """ Responsible for Connecting and reconnecting explore devices via usb interface"""
    def __init__(self, device_name):
        """Initialize Bluetooth connection
        """
        self.is_connected = False
        self.block_read_size = 2048
        self.device_name = device_name
        self.comm_manager = None
        self.device_manager = None
        self.bt_sdk = None
        self.usb_stop_flag = threading.Event()
        self.copy_buffer = bytearray()
        self.reader_thread = None
        self.lock = threading.Lock()
        self.max_time = 0
        self.start = 0
        self.stop = 0

        self.data_array_size = 100
        self.data_array = [bytearray(2048) for _ in range(self.data_array_size)]
        self.data_array_read = 0
        self.data_array_write = 0
        self.counting_semaphore = CountingSemaphore(0)



    def read_serial_in_chunks(self):
        """Reads data in fixed-size chunks from the serial port until stopped."""
        # start stream
        fletcher = b'\xaf\xbe\xad\xde'
        cmd = b'\xE4' * 10 + fletcher
        self.comm_manager.write(cmd)

        self.is_connected = True
        while not self.usb_stop_flag.is_set():
            try:
                # no time consuming operation here
                # critical section

                self.data_array[self.data_array_write] = self.comm_manager.read(self.block_read_size)
                self.start = local_clock()
                #print("out")
                #print(self.data_array[self.data_array_write])
                if len(self.data_array[self.data_array_write]) !=  self.block_read_size:
                    self.usb_stop_flag.set()
                    self.stop = local_clock() - self.start
                    if (self.max_time < self.stop) and self.start != 0:
                        self.max_time = self.stop
                        print("max time: " + str(self.max_time))
                    print("self.usb_stop_flag.set()self.usb_stop_flag.set()self.usb_stop_flag.set()self.usb_stop_flag.set()self.usb_stop_flag.set()")

                self.data_array_write += 1
                if self.data_array_write >= self.data_array_size:
                    self.data_array_write = 0
                self.start = local_clock()
                self.counting_semaphore.release()
                self.stop = local_clock() - self.start
                if (self.max_time < self.stop) and self.start != 0:
                    self.max_time = self.stop
                    print("max time: " + str(self.max_time))


            except Exception as e:
                logger.debug('Got Exception in USB read method: {}'.format(e))
                print('test Got Exception in USB read method: {}'.format(e))

                self.stop = local_clock() - self.start
                if (self.max_time < self.stop) and self.start != 0:
                    self.max_time = self.stop
                    print("max time: " + str(self.max_time))
        logger.debug('Stopping USB data retrieval thread')
        print('Stopping USB data retrieval threadStopping USB data retrieval threadStopping USB data retrieval threadStopping USB data retrieval threadStopping USB data retrieval thread')

    def connect(self):
        """Connect to the device and return the socket

        Returns:
            socket (bluetooth.socket)
        """

        fletcher = b'\xaf\xbe\xad\xde'
        for _ in range(5):
            try:
                port = get_correct_com_port(self.device_name)
                self.comm_manager = serial.Serial(port=port, baudrate=115200, timeout=0.5)

                # stop stream
                cmd = b'\xE5' * 10 + fletcher
                self.comm_manager.write(cmd)
                # read all waits till timeout
                self.comm_manager.readall() 
				
                self.comm_manager.timeout = 2
                self.reader_thread = threading.Thread(
                    target=self.read_serial_in_chunks,
                    daemon=True
                )
                self.reader_thread.start()

                return 0
            except PermissionError:
                # do nothing here as this comes from posix
                pass
            except serial.serialutil.SerialException:
                raise serial.serialutil.SerialException('Permission error on USB port access. Please set the permission'
                                                        ' temporarily via the terminal to allow port access:\n\nchmod 7'
                                                        '77 <port_name>\n\nAlternatively, set up a udev rule following '
                                                        'the ExplorePy documentation:\n\nhttps://explorepy.readthedocs.'
                                                        'io/en/latest/installation.html#set-up-usb-streaming-in-linux')
                logger.info(
                    "Permission denied on serial port access, please run this command via"
                    "terminal: sudo chmod 777 {}".format(port)
                )
            except Exception as error:
                self.is_connected = False
                logger.info(
                    "Got an exception while connecting to the device: {} of type: {}".format(error, type(error))
                )
                logger.debug('trying to connect again as tty port is not visible yet')
                logger.warning("Could not connect; Retrying in 2s...")
                time.sleep(2)

        self.is_connected = False
        raise DeviceNotFoundError(
            "Could not find the device! Please turn on the device, wait a few seconds and connect to"
            "serial port before starting ExplorePy"
        )

    def reconnect(self):
        """Reconnect to the last connected device

        Generally speaking this is not necessary for USB stream, but we keep it as placeholder
        in case it is needed in future
        """
        raise NotImplementedError

    def disconnect(self):
        """Disconnect from the device"""
        self.usb_stop_flag.set()
        self.reader_thread.join(timeout=2)
        self.is_connected = False
        self.comm_manager.cancel_read()
        self.comm_manager.close()

    def read(self, n_bytes):
        """Read n_bytes from the socket

            Args:
                n_bytes (int): number of bytes to be read

            Returns:
                list of bytes
        """
        try:

            if len(self.copy_buffer) < n_bytes:
                self.counting_semaphore.acquire()
                # while self.data_array_write != self.data_array_read:

                self.copy_buffer.extend(self.data_array[self.data_array_read])

                self.data_array_read += 1
                if self.data_array_read >= self.data_array_size:
                    self.data_array_read = 0


            chunk = bytearray()

            #while len(chunk) < n_bytes:
                #chunk.append(self.copy_buffer.popleft())
            chunk = self.copy_buffer[:n_bytes]
            self.copy_buffer = self.copy_buffer[n_bytes:]

            return chunk
        except Exception as error:
            logger.debug('Got error or read request: {}'.format(error))
            print('test Got error or read request: {}'.format(error))
            print('nbtes: ' + str(n_bytes))
            print ("len of copy buffer: " +str(len(self.copy_buffer)))

    def send(self, data):
        """Send data to the device

        Args:
            data (bytearray): Data to be sent
        """
        with threading.Lock():
            self.comm_manager.write(data)


def get_correct_com_port(device_name):
    """ Returns correct COM/tty port for usb connection
    Args: device name: the name of the device to connect to
    """
    fletcher = b'\xaf\xbe\xad\xde'

    ports = list(list_ports.comports())
    max_time = 0
    cnt = 0
    for p in ports:
        if p.vid == 0x0483 and p.pid == 0x5740:
            serial_port = serial.Serial(port=p.device, baudrate=115200, timeout=0.5)
            # stop stream
            cmd = b'\xE5' * 10 + fletcher
            serial_port.write(cmd)
            start = local_clock()
            # read all the stream data, waits till time out
            serial_port.readall()

            get_name_cmd = b'\xC6' * 10 + fletcher
            serial_port.write(get_name_cmd)
            data = serial_port.read(24)
            print("name len:" + str(len(data)))
            if len(data) == 0:
                # incompatible explore device, continue connection
                print('got data as zero')
                serial_port.close()
                return p.device
            name = data[8:-4].decode('utf-8', errors='ignore')
            if name == device_name:
                stop = local_clock() -start
                print("find duration:" + str(stop))
                return p.device
    raise DeviceNotFoundError()
