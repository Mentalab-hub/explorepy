import sys
import time
import unittest
from unittest import TestCase
from unittest.mock import patch
import numpy as np
from explorepy.packet import Packet
from explorepy.packet import EEG
from explorepy.packet import EEG94
from explorepy.packet import EEG98


class TestBasePacket(TestCase):

    def test_is_abstract(self):
        """
        Test if explorepy.packet.Packet is abstract
        """
        with self.assertRaises(Exception):
            Packet(12345, b'\x00\x00\x00\x00')

    @patch.multiple(EEG, __abstractmethods__=set())
    def test_int24to32(self):
        res = Packet.int24to32([10, 20, 30, 40, 50, 60])
        np.testing.assert_array_equal(res, [1971210, 3945000])

    @patch.multiple(EEG, __abstractmethods__=set())
    def test_int24to32_signed(self):
        # [111111111111111111111111, 111111111111111111111111]
        res = Packet.int24to32([255, 255, 255, 255, 255, 255])
        np.testing.assert_array_equal(res, [-1, -1])

    @patch.multiple(EEG, __abstractmethods__=set())
    def test_int24to32_zero(self):
        res = Packet.int24to32([0, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(res, [0, 0])

    @patch.multiple(EEG, __abstractmethods__=set())
    def test_int24to32_min(self):
        # [100000000000000000000000, 100000000000000000000000]
        res = Packet.int24to32([0, 0, 128, 0, 0, 128])
        np.testing.assert_array_equal(res, [-8388608, -8388608])


class TestEEGPacket(TestCase):

    def test_is_abstract(self):
        with self.assertRaises(Exception):
            EEG(12345, b'\x00\x00\x00\x00')

    @patch.multiple(EEG, __abstractmethods__=set())
    def test_ptp(self):
        p = EEG(12345, b'\x00\x00\x00\x00')
        p.data = np.array([[40, 3333, 78910, -30],[20, -1000, 10, 30],[10, 2345, 77016, 11],[15, 1234, 70000, 2]])
        res = p.get_ptp()
        np.testing.assert_array_equal(res, [78940, 1030, 77006, 69998])


# The data part of the packet has 507 - 12 bytes of data, with 3 bytes per channel, 4 channels, 1 status msg
# with 4 channels + 1 status message @ 3 bytes each being 15 bytes, so (507-12)/(5*3) = 33 rows of data
# (I think)
# Note that this does *not* test the proper fletcher implementation
class TestEEG94Packet(TestCase):

    @classmethod
    def setUpClass(cls):
        # 99 bytes of data + 4 bytes fletcher (0xDEADBEAF)
        cls.data = b'\x01\x02\x03\x04\x05\x06\x07\x08\x09\x08\x07\x06\x05\x04\x03\x02\x01\x01\x02\x03\x04\x05\x06' \
                  b'\x07\x08\x09\x08\x07\x06\x05\x04\x03\x02\x01\x01\x02\x03\x04\x05\x06\x07\x08\x09\x08\x07\x06' \
                  b'\x05\x04\x03\x02\x01\x01\x02\x03\x04\x05\x06\x07\x08\x09\x08\x07\x06\x05\x04\x03\x02\x01\x01' \
                  b'\x02\x03\x04\x05\x06\x07\x08\x09\x08\x07\x06\x05\x04\x03\x02\x01\x01\x02\x03\x04\x05\x06\x07' \
                  b'\x04\x03\x02\x01\x01\x02\x03'
        cls.fletcher = b'\xaf\xbe\xad\xde'
        cls.eeg94 = EEG94(12345, cls.data + cls.fletcher, 300)

    # Should impedance be none? Is imp_calib_info know at instantiation?
    def test_init_impedance_none(self):
        self.assertIsNone(self.eeg94.imp_data)

    def test_init_timestamp_correct(self):
        self.assertEqual(self.eeg94.timestamp, 301.2345)

    # EEG94 converts the data on initialization,
    # if the packet size is fixed only a specific length of bytes should be accepted
    def test_data_length_too_long(self):
        payload = 2 * self.data + self.fletcher

        with self.assertRaises(Exception):
            EEG94(12345, payload, 300)

    def test_convert(self):
        pass

    def test_str(self):
        pass

    # The Fletcher part of the packets doesn't make sense
    # TODO implement here what Fletcher is supposed to do (32bit)
    def test_fletcher(self):
        pass


class TestEEG98Packet(TestCase):
    pass


class TestEEG99Packet(TestCase):
    pass


class TestEEG99sPacket(TestCase):
    pass


class TestOrientationPacket(TestCase):
    pass


class TestEnvironmentPacket(TestCase):
    pass


class TestTimestampPacket(TestCase):
    pass


class TestEventMarkerPacket(TestCase):
    pass


class TestPushButtonMarkerPacket(TestCase):
    pass


class TestSoftwareMarkerPacket(TestCase):
    pass


class TestTriggerInPacket(TestCase):
    pass


class TestTriggerOutPacket(TestCase):
    pass


class TestDisconnectPacket(TestCase):
    pass


class TestDeviceInfoPacket(TestCase):
    pass


class TestCommandRCVPacket(TestCase):
    pass


class TestCommandStatusPacket(TestCase):
    pass


class TestCalibrationInfoPacket(TestCase):
    pass


if __name__ == '__main__':
    unittest.main()
