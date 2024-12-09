# -*- coding: utf-8 -*-
"""A module for bluetooth connection"""
import abc
import asyncio
import atexit
import logging
import queue
import threading
import time
from queue import Queue

from bleak import (
    BleakClient,
    BleakScanner
)

from explorepy import (
    exploresdk,
    settings_manager
)
from explorepy._exceptions import (
    BleDisconnectionError,
    DeviceNotFoundError,
    InputError
)


logger = logging.getLogger(__name__)


class BTClient(abc.ABC):
    @abc.abstractmethod
    def __init__(self, device_name=None, mac_address=None):
        if (mac_address is None) and (device_name is None):
            raise InputError("Either name or address options must be provided!")
        self.is_connected = False
        self.mac_address = mac_address
        self.device_name = device_name
        self.bt_serial_port_manager = None
        self.device_manager = None

    @abc.abstractmethod
    def connect(self):
        """Connect to the device and return the socket

        Returns:
            socket (bluetooth.socket)
        """

    @abc.abstractmethod
    def reconnect(self):
        """Reconnect to the last used bluetooth socket.

        This function reconnects to the last bluetooth socket. If after 1 minute the connection doesn't succeed,
        program will end.
        """

    @abc.abstractmethod
    def disconnect(self):
        """Disconnect from the device"""

    @abc.abstractmethod
    def _find_mac_address(self):
        pass

    @abc.abstractmethod
    def read(self, n_bytes):
        """Read n_bytes from the socket

            Args:
                n_bytes (int): number of bytes to be read

            Returns:
                list of bytes
        """

    @abc.abstractmethod
    def send(self, data):
        """Send data to the device

        Args:
            data (bytearray): Data to be sent
        """

    @staticmethod
    def _check_mac_address(device_name, mac_address):
        return (device_name[-4:-2] == mac_address[-5:-3]) and (device_name[-2:] == mac_address[-2:])


class SDKBtClient(BTClient):
    """ Responsible for Connecting and reconnecting explore devices via bluetooth"""

    def __init__(self, device_name=None, mac_address=None):
        """Initialize Bluetooth connection

        Args:
            device_name(str): Name of the device (either device_name or device address should be given)
            mac_address(str): Devices MAC address
        """
        if (mac_address is None) and (device_name is None):
            raise InputError("Either name or address options must be provided!")
        self.is_connected = False
        self.mac_address = mac_address
        self.device_name = device_name
        self.bt_serial_port_manager = None
        self.device_manager = None

    def connect(self):
        """Connect to the device and return the socket

        Returns:
            socket (bluetooth.socket)
        """
        config_manager = settings_manager.SettingsManager(self.device_name)
        mac_address = config_manager.get_mac_address()

        if mac_address is None:
            self._find_mac_address()
            config_manager.set_mac_address(self.mac_address)
        else:
            self.mac_address = mac_address
            self.device_name = "Explore_" + str(self.mac_address[-5:-3]) + str(self.mac_address[-2:])

        for _ in range(5):
            try:
                self.bt_serial_port_manager = exploresdk.BTSerialPortBinding.Create(self.mac_address, 5)
                return_code = self.bt_serial_port_manager.Connect()
                logger.debug("Return code for connection attempt is : {}".format(return_code))

                if return_code == 0:
                    self.is_connected = True
                    return
                else:
                    self.is_connected = False
                    logger.warning("Could not connect; Retrying in 2s...")
                    time.sleep(2)

            except SystemError as error:
                self.is_connected = False
                logger.debug(
                    "Got an exception while connecting to the device: {} of type: {}".format(error, type(error)))

            except TypeError as error:
                self.is_connected = False
                logger.debug(
                    "Got an exception while connecting to the device: {} of type: {}".format(error, type(error))
                )
                raise ConnectionRefusedError("Please unpair Explore device manually or use a Bluetooth dongle")
            except Exception as error:
                self.is_connected = False
                logger.debug(
                    "Got an exception while connecting to the device: {} of type: {}".format(error, type(error))
                )
                logger.warning("Could not connect; Retrying in 2s...")
                time.sleep(2)

        self.is_connected = False
        raise DeviceNotFoundError(
            "Could not find the device! Please make sure bluetooth is turned on in your computer and the Explore "
            "device is in advertising mode. "
        )

    def reconnect(self):
        """Reconnect to the last used bluetooth socket.

        This function reconnects to the last bluetooth socket. If after 1 minute the connection doesn't succeed,
        program will end.
        """
        self.is_connected = False
        for _ in range(5):
            self.bt_serial_port_manager = exploresdk.BTSerialPortBinding.Create(self.mac_address, 5)
            connection_error_code = self.bt_serial_port_manager.Connect()
            logger.debug("Got an exception while connecting to the device: {}".format(connection_error_code))
            if connection_error_code == 0:
                self.is_connected = True
                logger.info('Connected to the device')
                return self.bt_serial_port_manager
            else:
                self.is_connected = False
                logger.warning("Couldn't connect to the device. Trying to reconnect...")
                time.sleep(2)
        logger.error("Could not reconnect after 5 attempts. Closing the socket.")
        return None

    def disconnect(self):
        """Disconnect from the device"""
        self.is_connected = False
        self.bt_serial_port_manager.Close()

    def _find_mac_address(self):
        self.device_manager = exploresdk.ExploreSDK_Create()
        for _ in range(5):
            available_list = self.device_manager.PerformDeviceSearch()
            logger.debug("Number of devices found: {}".format(len(available_list)))
            for bt_device in available_list:
                if bt_device.name == self.device_name:
                    self.mac_address = bt_device.address
                    return

            logger.warning("No device found with the name: %s, searching again...", self.device_name)
            time.sleep(0.1)
        raise DeviceNotFoundError("No device found with the name: {}".format(self.device_name))

    def read(self, n_bytes):
        """Read n_bytes from the socket

            Args:
                n_bytes (int): number of bytes to be read

            Returns:
                list of bytes
        """
        try:
            read_output = self.bt_serial_port_manager.Read(n_bytes)
            actual_byte_data = read_output.encode('utf-8', errors='surrogateescape')
            return actual_byte_data
        except OverflowError as error:
            if not self.is_connected:
                raise IOError("connection has been closed")
            else:
                logger.debug(
                    "Got an exception while reading data from "
                    "socket which connection is open: {} of type:{}".format(error, type(error)))
                raise ConnectionAbortedError(error)
        except IOError as error:
            if not self.is_connected:
                raise IOError(str(error))
        except (MemoryError, OSError) as error:
            logger.debug("Got an exception while reading data from socket: {} of type:{}".format(error, type(error)))
            raise ConnectionAbortedError(str(error))
        except Exception as error:
            print(error)
            logger.error(
                "unknown error occured while reading bluetooth data by "
                "exploresdk {} of type:{}".format(error, type(error))
            )

    def send(self, data):
        """Send data to the device

        Args:
            data (bytearray): Data to be sent
        """
        self.bt_serial_port_manager.Write(data)


class BLEClient(BTClient):
    """ Responsible for Connecting and reconnecting explore devices via bluetooth"""

    def __init__(self, device_name=None, mac_address=None):
        """Initialize Bluetooth connection

        Args:
            device_name(str): Name of the device (either device_name or device address should be given)
            mac_address(str): Devices MAC address
        """
        super().__init__(device_name=device_name, mac_address=mac_address)

        self.client = None
        self.ble_device = None
        self.eeg_service_uuid = "FFFE0001-B5A3-F393-E0A9-E50E24DCCA9E"
        self.eeg_tx_char_uuid = "FFFE0003-B5A3-F393-E0A9-E50E24DCCA9E"
        self.eeg_rx_char_uuid = "FFFE0002-B5A3-F393-E0A9-E50E24DCCA9E"
        self.rx_char = None
        self.buffer = Queue()
        self.try_disconnect = asyncio.Event()
        self.did_reconnection_fail = asyncio.Event()
        self.notification_thread = None
        self.copy_buffer = bytearray()
        self.read_event = asyncio.Event()
        self.data = None
        self.connection_attempt_counter = 0
        self.result_queue = Queue()

    async def stream(self):
        while True:
            if not self.is_connected:
                break
            if self.try_disconnect.is_set():
                logger.info("scanning for device")
                device = await BleakScanner.find_device_by_name(self.device_name, timeout=3)

                if device is None:
                    logger.info("no device found, wait then scan again")
                    await asyncio.sleep(1)
                    if self.connection_attempt_counter < 2:
                        self.connection_attempt_counter += 1
                        continue
                    else:
                        self.did_reconnection_fail.set()
                        logger.info("ExplorePy initiating disconnection")
                        loop = asyncio.get_running_loop()
                        disconnect_task = loop.create_task(self.client.disconnect())
                        await disconnect_task
                self.connection_attempt_counter = 0

            def disconnection_callback(_: BleakClient):
                logger.debug("Device sent disconnection callback")
                # cancelling all tasks effectively ends the program
                if self.is_connected:
                    self.try_disconnect.set()
                    self.read_event.set()

            self.client = BleakClient(self.ble_device, disconnected_callback=disconnection_callback)
            loop = asyncio.get_running_loop()
            connect_task = loop.create_task(self.client.connect())
            await connect_task

            # async with BleakClient(self.ble_device) as client:
            def handle_packet(sender, bt_byte_array):
                # write packet to buffer
                self.buffer.put(bt_byte_array)

            loop = asyncio.get_running_loop()
            print('1')
            self.notify_task = loop.create_task(self.client.start_notify(self.eeg_tx_char_uuid, handle_packet))
            print('2')
            try:
                print('3')
                await self.notify_task
                print('4')
            except asyncio.CancelledError:
                print("Notify task is cancelled now")
            except Exception as error:
                print('Got exception here: {}'.format(error))

            print('5 ')
            self.rx_char = self.client.services.get_service(self.eeg_service_uuid).get_characteristic(
                self.eeg_rx_char_uuid)
            while True:
                loop = asyncio.get_running_loop()
                try:
                    loop.run_in_executor(None, await self.read_event.wait())
                except Exception:
                    print('Got exception while waiting for read event in BLE thread')
                if self.data is None:
                    if self.try_disconnect.is_set() and self.is_connected:
                        logger.debug('Closing write thread, will attempt reconnection')
                        self.read_event.clear()
                        break
                    logger.debug('Client disconnection requested')
                    self.is_connected = False
                    await self.client.disconnect()
                    break
                await self.client.write_gatt_char(self.rx_char, self.data, response=False)
                self.data = None
                self.read_event.clear()

    def connect(self):
        """Connect to the device and return the socket

        Returns:
            socket (bluetooth.socket)
        """
        self.is_connected = True
        self.notification_thread = threading.Thread(target=self.start_read_loop, daemon=True)
        self.notification_thread.start()
        print('waiting for BLE device to show up..')
        self.result_queue.get()

        if self.ble_device is None:
            logger.info('No device found!!')
            raise DeviceNotFoundError('Could not find device')
        else:
            logger.info('Device is connected')
            self.is_connected = True

            atexit.register(self.disconnect)

    def start_read_loop(self):
        try:
            asyncio.run(self.ble_manager())
        except RuntimeError as error:
            logger.info('Shutting down BLE stream loop with error {}'.format(error))
        except asyncio.exceptions.CancelledError as error:
            logger.debug('asyncio.exceptions.CancelledError from BLE stream thread {}'.format(error))
        except BleDisconnectionError as error:
            print('Got error as {}'.format(error))
            raise error

    def stop_read_loop(self):
        logger.debug('Stopping BLE stream loop')

        self.stream_task.cancel()
        if not self.try_disconnect.is_set():
            self.notification_thread.join()

    async def ble_manager(self):
        try:
            discovery_task = asyncio.create_task(self._discover_device())
            await discovery_task
            logger.debug('Finished device discovery..')
            self.result_queue.put(True)
            self.stream_task = asyncio.create_task(self.stream())
            await self.stream_task
        except DeviceNotFoundError:
            self.result_queue.put(False)
            logger.debug('No matching device found')
        except BleDisconnectionError as error:
            print('Got an BleDisconnectionError {}'.format(error))
            raise error
        except Exception as error:
            logger.debug('Got an BLE exception with error {}'.format(error))

    async def _discover_device(self):
        if self.mac_address:
            self.ble_device = await BleakScanner.find_device_by_address(self.mac_address)
        else:
            logger.info('Commencing device discovery')
            print(self.device_name)
            self.ble_device = await BleakScanner.find_device_by_name(self.device_name, timeout=2)

            if self.ble_device:
                logger.debug('found device!!!!')
        if self.ble_device is None:
            logger.debug('No device found!!!!!')
            raise DeviceNotFoundError(
                "Could not discover the device! Please make sure the device is on and in advertising mode."
            )

    def reconnect(self):
        """Reconnect to the last used bluetooth socket.

        This function reconnects to the last bluetooth socket. If after 1 minute the connection doesn't succeed,
        program will end.
        """
        self.is_connected = False
        return None

    def disconnect(self):
        """Disconnect from the device"""

        self.is_connected = False
        self.notify_task.cancel()
        self.read_event.set()
        time.sleep(1)
        self.stop_read_loop()
        self.ble_device = None
        self.buffer = Queue()
        logger.info('ExplorePy disconnecting from device')

    def _find_mac_address(self):
        raise NotImplementedError

    def read(self, n_bytes):
        """Read n_bytes from the socket

            Args:
                n_bytes (int): number of bytes to be read

            Returns:
                list of bytes
        """
        try:
            if len(self.copy_buffer) < n_bytes:
                get_item = self.buffer.get(timeout=5)
                self.copy_buffer.extend(get_item)
            ret = self.copy_buffer[:n_bytes]
            self.copy_buffer = self.copy_buffer[n_bytes:]
            if len(ret) < n_bytes:
                logger.info('data size mismatch in buffer, raising connection aborted error when trying to read {}'
                            'bytes'.format(n_bytes))
                raise ConnectionAbortedError('Error reading data from BLE stream, too many bytes requested')
            self.try_disconnect.clear()
            return ret
        except queue.Empty:
            self.try_disconnect.set()
            if self.did_reconnection_fail.is_set():
                raise BleDisconnectionError
            if self.try_disconnect.is_set():
                logger.debug('No item in bluetooth buffer, initiating next read() call')
                return self.read(n_bytes)
            else:
                logger.info('disconnection flag is not set, continuing')
        except Exception as error:
            logger.error('Unknown error reading data from BLE stream, error is {}'.format(error))
            raise ConnectionAbortedError(str(error))

    def send(self, data):
        """Send data to the device

        Args:
            data (bytearray): Data to be sent
        """
        self.data = data
        logger.debug('sending data to device')
        self.read_event.set()
