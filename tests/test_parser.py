import logging
import unittest
from unittest.mock import mock_open, patch, MagicMock, call
import explorepy
from explorepy._exceptions import FletcherError, ReconnectionFlowError
from explorepy.packet import Packet
from src.explorepy.parser import FileHandler, Parser, DeviceInfo

# Define a reusable mock class for Packet
class MockPacket(Packet):
    def __init__(self, timestamp, payload, time_offset=0):
        pass

    def _convert(self, bin_data):
        pass

    def __str__(self):
        return "MockPacket"
    
class TestParser(unittest.TestCase):

    logger = logging.getLogger(__name__)

    @patch('explorepy.btcpp.BLEClient')
    @patch.object(Parser, '_stream')
    def test_start_streaming_ble(self, mock_ble_client, mock_stream):
        # Arrange
        callback = MagicMock()
        parser = Parser(callback=callback)
        device_name = "Explore_AAAA"  # Last four characters are alphabetic
        mac_address = "00:11:22:33:44:55"

        # Act
        parser.start_streaming(device_name, mac_address)
        self.assertEqual(explorepy.get_bt_interface(), "ble")
        parser.stream_interface.connect.assert_called_once()
        mock_stream.assert_called_once()

    @patch('explorepy.serial_client.SerialClient')
    @patch.object(Parser, '_stream')
    @patch('sys.platform', 'darwin')
    def test_start_streaming_macos(self, mock_serial_client, mock_stream):
        # Arrange
        callback = MagicMock()
        parser = Parser(callback=callback)
        device_name = "Explore_AAAA1"  # Last four characters are not alphabetic
        mac_address = "00:11:22:33:44:55"

        # Act
        parser.start_streaming(device_name, mac_address)
        self.assertEqual(explorepy.get_bt_interface(), "pyserial")
        parser.stream_interface.connect.assert_called_once()
        mock_stream.assert_called_once()

    @patch('explorepy.btcpp.SDKBtClient')
    @patch.object(Parser, '_stream')
    @patch('sys.platform', 'win32')
    def test_start_streaming_not_macos(self, mock_sdk_client, mock_stream):
        # Arrange
        callback = MagicMock()
        parser = Parser(callback=callback)
        device_name = "Explore_AAAA1"
        mac_address = "00:11:22:33:44:55"

        # Act
        parser.start_streaming(device_name, mac_address)
        self.assertEqual(explorepy.get_bt_interface(), "sdk")
        parser.stream_interface.connect.assert_called_once()
        mock_stream.assert_called_once()

    # TODO: usb

    @patch('sys.platform', 'win32')
    @patch('explorepy.get_bt_interface', return_value='invalid')
    def test_start_streaming_invalid_interface(self, mock_get_bt_interface):
        # Arrange
        callback = MagicMock()
        parser = Parser(callback=callback)
        device_name = "Explore_AAAA1"
        mac_address = "00:11:22:33:44:55"

        # Act
        self.assertRaises(ValueError, parser.start_streaming, device_name, mac_address)
            

    @patch('explorepy.bt_mock_client.MockBtClient')
    def test_stop_streaming(self, mock_mockbt_client):
        # Arrange
        callback = MagicMock()
        parser = Parser(callback=callback)
        parser._do_streaming = True
        parser.stream_interface = mock_mockbt_client
        parser.usb_marker_port = MagicMock()

        # Act
        parser.stop_streaming()

        # Assert
        self.assertFalse(parser._do_streaming)
        callback.assert_called_once_with(None)
        mock_mockbt_client.disconnect.assert_called_once()
        self.assertIsNone(parser.stream_interface)
        parser.usb_marker_port.close.assert_called_once()

    @patch.object(Parser, '_stream')
    def test_start_reading(self, mock_stream):
        with patch('src.explorepy.parser.FileHandler') as mock_file_handler:
            callback = MagicMock()
            parser = Parser(callback=callback)
            
            parser.start_reading("test_data.bin")

            mock_file_handler.assert_called_once_with("test_data.bin")
            mock_stream.assert_called_once_with(new_thread=True)

    @patch('src.explorepy.parser.Thread')
    def test_stream_with_new_thread(self, mock_thread_class):
        """Test that _stream creates and starts a new thread when new_thread=True."""
        callback = MagicMock()
        parser = Parser(callback=callback)
        parser._stream_loop = MagicMock()

        # Act
        parser._stream(new_thread=True)

        # Assert
        self.assertTrue(parser._do_streaming)
        mock_thread_class.assert_called_once_with(name="ParserThread", target=parser._stream_loop)
        mock_thread_class.return_value.setDaemon.assert_called_once_with(True)
        mock_thread_class.return_value.start.assert_called_once()
        parser._stream_loop.assert_not_called()

    def test_stream_without_new_thread(self):
        # Arrange
        callback = MagicMock()
        parser = Parser(callback=callback)
        parser._stream_loop = MagicMock()

        # Act
        parser._stream(new_thread=False)

        # Assert
        self.assertTrue(parser._do_streaming)
        parser._stream_loop.assert_called_once()

    def test_parse_packet_valid_pid(self):
        with patch('src.explorepy.parser.PACKET_CLASS_DICT', {1: MockPacket}):
            parser = Parser(callback=MagicMock())
            pid = 1
            timestamp = 123456789
            bin_data = b'\x00\x01\x02\xaf\xbe\xad\xde'
            
            packet = parser._parse_packet(pid, timestamp, bin_data)

            self.assertIsInstance(packet, MockPacket)
            self.assertIsNotNone(packet)

    def test_parse_packet_invalid_pid(self):
        with patch('src.explorepy.parser.PACKET_CLASS_DICT', {1: MockPacket}):
            parser = Parser(callback=MagicMock())
            pid = 2
            timestamp = 123456789
            bin_data = b'\x00\x01\x02\xaf\xbe\xad\xde'
            
            with self.assertRaises(FletcherError):
                packet = parser._parse_packet(pid, timestamp, bin_data)
                self.assertIsNone(packet)

    @patch('src.explorepy.parser.binascii.hexlify')
    @patch('src.explorepy.parser.is_usb_mode', return_value=False)
    @patch('src.explorepy.parser.is_explore_pro_device', return_value=False)
    def test_generate_packet(self, mock_is_explore_pro_device, mock_is_usb_mode, mock_hexlify):
        with patch('src.explorepy.parser.PACKET_CLASS_DICT', {1: MockPacket}):
            # Arrange
            callback = MagicMock()
            parser = Parser(callback=callback)
            parser.stream_interface = MagicMock()
            parser.stream_interface.read.side_effect = [
                b'\xaf',  # First byte
                b'\xbe\xad\xde',  # Next three bytes
                b'\x01\x00\x00\x00\x00\x00\x00\x00',  # Raw header (8 bytes)
                b'\x00\x00\x00\x00'  # Payload data
            ]
            mock_hexlify.side_effect = [b'af', b'beadde']
            parser.seek_new_pid.set()

            # Act
            packet, packet_size = parser._generate_packet()

            # Assert
            self.assertIsNotNone(packet)
            self.assertEqual(packet_size, 8 + (0 - 4))  # 8 bytes header + payload size
            callback.assert_called_once()

    @patch('src.explorepy.parser.binascii.hexlify')
    @patch('src.explorepy.parser.is_usb_mode', return_value=False)
    def test_generate_packet_fletcher_error(self, mock_is_usb_mode, mock_hexlify):
        callback = MagicMock()
        parser = Parser(callback=callback)
        parser.stream_interface = MagicMock()
        parser.stream_interface.read.side_effect = [
            b'\xaf',  # First byte
            b'\xbe\xad\xde',  # Next three bytes
            b'\x01\x00\x00\x00\x00\x00\x00\x00',  # Raw header (8 bytes)
            b'\x00\x00\x00\x00'  # Payload data
        ]
        mock_hexlify.side_effect = [b'af', b'beadde']
        parser.seek_new_pid.set()

        # Act & Assert
        with self.assertRaises(FletcherError):
            parser._generate_packet()

    # TODO: This is not working check it again
    @patch('src.explorepy.parser.binascii.hexlify')
    @patch('src.explorepy.parser.is_usb_mode', return_value=False)
    def test_generate_packet_payload_exceeds_limit(self, mock_is_usb_mode, mock_hexlify):
        # Arrange
        callback = MagicMock()
        parser = Parser(callback=callback)
        parser.stream_interface = MagicMock()
        
        # Simulate the sequence of bytes read from the stream interface
        parser.stream_interface.read.side_effect = [
            b'\xaf',  # First byte
            b'\xbe\xad\xde',  # Next three bytes
            b'\x27\x02\x00\x00\x00\x00\x00\x00',  # Raw header with payload size of 551
            b'\x00\x00\x00\x00'  # Payload data
        ]
        
        mock_hexlify.side_effect = [b'af', b'beadde']
        parser.seek_new_pid.set()

        # Act & Assert
        with self.assertRaises(FletcherError):
            parser._generate_packet()

    @patch('src.explorepy.parser.binascii.hexlify')
    @patch('src.explorepy.parser.is_usb_mode', return_value=False)
    def test_generate_packet_reconnection_flow_error(self, mock_is_usb_mode, mock_hexlify):
        # Arrange
        callback = MagicMock()
        parser = Parser(callback=callback)
        parser.stream_interface = MagicMock()
        
        # Set the state to simulate reconnection
        parser._is_reconnecting = True
        parser.seek_new_pid.set()

        # Act & Assert
        with self.assertRaises(ReconnectionFlowError):
            parser._generate_packet()

    @patch.object(Parser, '_generate_packet')
    def test_read_device_info(self, mock_generate_packet):
        with patch('src.explorepy.parser.FileHandler') as mock_file_handler:
            # Setup
            mock_file_handler_instance = mock_file_handler.return_value
            mock_generate_packet.side_effect = [
                (MagicMock(spec=DeviceInfo), None),  # Simulate a DeviceInfo packet
                EOFError()  # Simulate end of file
            ]
            callback = MagicMock()
            parser = Parser(callback=callback)

            # Execute
            parser.read_device_info("test_data.bin")

            # Verify
            mock_file_handler.assert_called_once_with("test_data.bin")
            mock_file_handler_instance.disconnect.assert_called_once()
            callback.assert_called_once()  # Ensure callback was called with a DeviceInfo packet

    @patch.object(Parser, '_generate_packet')
    @patch('src.explorepy.parser.logger')
    def test_read_device_info_error_handling(self, mock_logger, mock_generate_packet):
        with patch('src.explorepy.parser.FileHandler') as mock_file_handler:
            mock_file_handler_instance = mock_file_handler.return_value
            mock_generate_packet.side_effect = FletcherError("Test FletcherError")
            callback = MagicMock()
            parser = Parser(callback=callback)

            with self.assertRaises(FletcherError):
                parser.read_device_info("test_data.bin")

            # Ensure the error was logged
            mock_logger.error.assert_called_once_with('Conversion ended incomplete. The binary file is corrupted.')
            # Ensure disconnect is called
            mock_file_handler_instance.disconnect.assert_called_once()

class TestFileHandler(unittest.TestCase):

    @patch('builtins.open', new_callable=mock_open)
    def test_file_handler_initialization(self, mock_open):
        filename = 'test.bin'

        FileHandler(filename)

        mock_open.assert_called_once_with(filename, mode='rb')

    @patch('builtins.open', new_callable=mock_open, read_data=b'testdata')
    def test_read_invalid_length(self, mock_file):
        file_handler = FileHandler('test.bin')

        with self.assertRaises(ValueError) as context:
            file_handler.read(-1)

        self.assertEqual(str(context.exception), 'Read length must be a positive number!')

    @patch('builtins.open', new_callable=mock_open, read_data=b'\x27\x02\x00\x00\x00\x00\x00\x00')
    def test_read_valid(self, mock_file):
        mock_file.return_value.closed = False
        handler = FileHandler('test.bin')

        read_data = handler.read(4)

        mock_file.return_value.read.assert_called_once_with(4)
        self.assertEqual(read_data, b'\x27\x02\x00\x00')

        # Optionally, you can check that close is called
        handler.disconnect()
        mock_file.return_value.close.assert_called_once()

    @patch('builtins.open', new_callable=mock_open, read_data=b'\x27\x02\x00\x00\x00\x00\x00\x00')
    def test_read_exceeds_file_length(self, mock_file):
        mock_file.return_value.closed = False
        handler = FileHandler('test.bin')

        with self.assertRaises(EOFError):
            handler.read(9)

    @patch('builtins.open', new_callable=mock_open, read_data=b'')
    def test_read_empty_file(self, mock_file):
        mock_file.return_value.closed = False
        handler = FileHandler('test.bin')

        with self.assertRaises(EOFError):
            handler.read(1)

    @patch('builtins.open', new_callable=mock_open, read_data=b'')
    def test_read_closed_file(self, mock_file):
        mock_file.return_value.closed = True  # Mock the closed attribute
        handler = FileHandler('test.bin')

        with self.assertRaises(IOError):
            handler.read(1)

    @patch('builtins.open', new_callable=mock_open)
    def test_disconnect(self, mock_file):
        mock_file.return_value.closed = False
        handler = FileHandler('test.bin')

        handler.disconnect()

        mock_file.return_value.close.assert_called_once()

    @patch('builtins.open', new_callable=mock_open)
    def test_disconnect_closed_file(self, mock_file):
        mock_file.return_value.closed = True
        handler = FileHandler('test.bin')

        handler.disconnect()

        mock_file.return_value.close.assert_not_called()

class TestParserStreamLoop(unittest.TestCase):
    @patch('src.explorepy.parser.Parser._generate_packet', return_value=(MagicMock(), 10))
    def test_stream_loop_normal_flow(self, mock_generate_packet):
        parser = Parser(callback=MagicMock())
        parser._do_streaming = True
        parser.total_packet_size_read = 0

        def one_iteration_side_effect():
            """Stop streaming after one successful iteration."""
            parser._do_streaming = False
            return (MagicMock(), 10)

        mock_generate_packet.side_effect = one_iteration_side_effect

        parser._stream_loop()
        # Assert that _generate_packet was called at least once
        mock_generate_packet.assert_called()
        # Assert the total_packet_size_read is updated
        self.assertEqual(parser.total_packet_size_read, 10)
        # Assert callback was called with a packet
        parser.callback.assert_called_once()

    @patch('src.explorepy.parser.Parser._generate_packet', side_effect=ReconnectionFlowError)
    def test_stream_loop_reconnection_flow(self, mock_generate_packet):
        parser = Parser(callback=MagicMock())
        parser._do_streaming = True

        def stop_after_first():
            parser._do_streaming = False
            raise ReconnectionFlowError

        mock_generate_packet.side_effect = stop_after_first

        # Should not raise, just pass
        parser._stream_loop()
        parser.callback.assert_not_called()  # No packet generated

if __name__ == '__main__':
    unittest.main()