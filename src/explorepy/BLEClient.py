import asyncio
import atexit
import logging
import threading
import time
from queue import (
    Empty,
    Queue
)

import bleak.uuids
from bleak import (
    BleakClient,
    BleakScanner
)

from explorepy._exceptions import (
    BleDisconnectionError,
    DeviceNotFoundError
)
from explorepy.BTClient import BTClient


logger = logging.getLogger(__name__)


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
        self.eeg_service_uuid = bleak.uuids.normalize_uuid_str("FFFE0001-B5A3-F393-E0A9-E50E24DCCA9E")
        self.eeg_tx_char_uuid = bleak.uuids.normalize_uuid_str("FFFE0003-B5A3-F393-E0A9-E50E24DCCA9E")
        self.eeg_rx_char_uuid = bleak.uuids.normalize_uuid_str("FFFE0002-B5A3-F393-E0A9-E50E24DCCA9E")
        self.rx_char = None
        self.buffer = Queue()
        self.notification_thread = None
        self.copy_buffer = bytearray()
        self.read_event = asyncio.Event()
        self.data = None
        self.result_queue = Queue()

    async def stream(self):
        while True:
            if not self.is_connected:
                break

            def disconnection_callback(_: BleakClient):
                logger.debug("Device sent disconnection callback")
                # cancelling all tasks effectively ends the program
                if self.is_connected:
                    self.read_event.set()

            if not self.client:
                self.client = BleakClient(self.ble_device, disconnected_callback=disconnection_callback)

            if not self.client.is_connected:
                loop = asyncio.get_running_loop()
                connect_task = loop.create_task(self.client.connect())
                await connect_task

            # async with BleakClient(self.ble_device) as client:
            def handle_packet(sender, bt_byte_array):
                # write packet to buffer
                self.buffer.put(bt_byte_array)

            loop = asyncio.get_running_loop()
            self.notify_task = loop.create_task(self.client.start_notify(self.eeg_tx_char_uuid, handle_packet))
            try:
                await self.notify_task
            except asyncio.CancelledError:
                print("Notify task is cancelled now")
            except Exception as error:
                print('Got exception here: {}'.format(error))

            self.rx_char = self.client.services.get_service(self.eeg_service_uuid).get_characteristic(
                self.eeg_rx_char_uuid)
            while True:
                loop = asyncio.get_running_loop()
                try:
                    loop.run_in_executor(None, await self.read_event.wait())
                except Exception:
                    print('Got exception while waiting for read event in BLE thread')
                if self.data is None:
                    await self.client.disconnect()
                    self.is_connected = False
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
                get_item = self.buffer.get(timeout=8)
                self.copy_buffer.extend(get_item)
            ret = self.copy_buffer[:n_bytes]
            self.copy_buffer = self.copy_buffer[n_bytes:]
            if len(ret) < n_bytes:
                logger.info('data size mismatch in buffer, raising connection aborted error when trying to read {}'
                            'bytes'.format(n_bytes))
                raise ConnectionAbortedError('Error reading data from BLE stream, too many bytes requested')
            return ret
        except Empty:
            raise ConnectionAbortedError
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
