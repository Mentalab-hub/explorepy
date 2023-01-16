# import sys
# import time
# import pytest
import unittest
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from explorepy.packet import (
    EEG,
    EEG94,
    EEG98,
    EEG99,
    EEG99s,
    EventMarker,
    Orientation,
    Packet
)


class TestBasePacket(TestCase):

    def test_is_abstract(self):
        """
        Test if explorepy.packet.Packet is abstract
        """
        with self.assertRaises(Exception):
            Packet(12345, b'\x00\x00\x00\x00')

    @patch.multiple(Packet, __abstractmethods__=set())
    def test_init_timestamp_correct(self):
        p = Packet(12345, b'\x00\x00\x00\x00', 300)
        self.assertEqual(p.timestamp, 301.2345)

    def test_int24to32(self):
        res = Packet.int24to32([10, 20, 30, 40, 50, 60])
        np.testing.assert_array_equal(res, [1971210, 3945000])

    def test_int24to32_signed(self):
        # [111111111111111111111111, 111111111111111111111111]
        res = Packet.int24to32([255, 255, 255, 255, 255, 255])
        np.testing.assert_array_equal(res, [-1, -1])

    def test_int24to32_zero(self):
        res = Packet.int24to32([0, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(res, [0, 0])

    def test_int24to32_min(self):
        # [100000000000000000000000, 100000000000000000000000]
        res = Packet.int24to32([0, 0, 128, 0, 0, 128])
        np.testing.assert_array_equal(res, [-8388608, -8388608])


class TestEEGPacket(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.p = EEG(12345, b'\xaf\xbe\xad\xde')
        cls.p.data = np.array(
            [[40, 3333, 78910, -30, 0], [20, -1000, 10, 30, 0], [10, 2345, 77016, 11, 45], [15, 1234, 70000, 2, 44]])

    # Technically redundant because setUpClass will raise an exception if EEG isn't abstract
    def test_is_abstract(self):
        with self.assertRaises(Exception):
            EEG(12345, b'\x00\x00\x00\x00')

    def test_calculate_impedance_no_data(self):
        imp_calib_info = {'slope': None, 'offset': None, 'noise_level': None}
        with self.assertRaises(Exception):
            self.p.calculate_impedance(imp_calib_info)

    def test_get_data(self):
        tv, d = self.p.get_data(250)
        self.assertEqual(len(tv), 4)
        np.testing.assert_array_equal(tv, [1.2345, 1.2385, 1.2425, 1.2465])
        np.testing.assert_array_equal(d, self.p.data)

    def test_get_data_no_sample_rate(self):
        tv, d = self.p.get_data()
        self.assertEqual(tv, 1.2345)
        np.testing.assert_array_equal(d, self.p.data)

    def test_ptp(self):
        res = self.p.get_ptp()
        np.testing.assert_array_equal(res, [78940, 1030, 77006, 69998])

    def test_ptp_no_data(self):
        self.p.data = b'\xaf\xbe\xad\xde'
        with self.assertRaises(Exception):
            self.p.get_ptp()


# The data part of the packet has 507 - 12 bytes of data, with 3 bytes per channel, 4 channels, 1 status msg
# with 4 channels + 1 status message @ 3 bytes each being 15 bytes, so (507-12)/(5*3) = 33 rows of data
# (I think)
# Note that this does *not* test the proper fletcher implementation
class TestEEG94Packet(TestCase):

    @classmethod
    def setUpClass(cls):
        # 99 bytes of data + 4 bytes fletcher (0xDEADBEAF)
        cls.data = bytes([i % 256 for i in range(495)])  # 495 bytes of dummy data, resembles 33 samples in 4+1 channels
        cls.fletcher = b'\xaf\xbe\xad\xde'
        cls.eeg94 = EEG94(12345, cls.data + cls.fletcher, 300)

    # EEG94 converts the data on initialization,
    # if the packet size is fixed only a specific length of bytes should be accepted
    def test_data_length_too_long(self):
        payload = 2 * self.data + self.fletcher
        with self.assertRaises(Exception):
            EEG94(12345, payload, 300)

    def test_convert(self):
        t = np.array([[15673.97, 62732.8, 109791.63],
                      [25085.74, 72144.56, 119203.39],
                      [34497.5, 81556.33, 128615.16],
                      [43909.27, 90968.1, 138026.92]])
        np.testing.assert_array_equal(self.eeg94.data[:, :3], t)

    # TODO implement status message test


class TestEEG98Packet(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = bytes([i % 256 for i in range(432)])
        cls.fletcher = b'\xaf\xbe\xad\xde'
        cls.eeg98 = EEG98(12345, cls.data + cls.fletcher, 300)

    def test_data_too_long(self):
        payload = 2 * self.data + self.fletcher
        with self.assertRaises(Exception):
            EEG98(12345, payload, 300)

    def test_convert(self):
        t = np.array([[15673.97, 100379.86],
                      [25085.74, 109791.63],
                      [34497.5, 119203.39],
                      [43909.27, 128615.16],
                      [53321.03, 138026.92],
                      [62732.8, 147438.69],
                      [72144.56, 156850.45],
                      [81556.33, 166262.22]])
        np.testing.assert_array_equal(self.eeg98.data[:, :2], t)

    # TODO implement status message test


class TestEEG99Packet(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = bytes([i % 256 for i in range(432)])
        cls.fletcher = b'\xaf\xbe\xad\xde'
        cls.eeg99 = EEG99(12345, cls.data + cls.fletcher, 300)

    def test_data_too_long(self):
        payload = 2 * self.data + self.fletcher
        with self.assertRaises(Exception):
            EEG99(12345, payload, 300)

    def test_convert(self):
        t = np.array([[11741.64, 170565.18],
                      [29388.7, 188212.24],
                      [47035.76, 205859.3],
                      [64682.82, 223506.36],
                      [82329.88, 241153.42],
                      [99976.94, 258800.48],
                      [117624.0, 276447.54],
                      [135271.06, 294094.6],
                      [152918.12, 311741.66]])
        np.testing.assert_array_equal(t, self.eeg99.data[:, :2])

    # TODO implement status message check


class TestEEG99sPacket(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = bytes([i % 256 for i in range(432)])
        cls.fletcher = b'\xaf\xbe\xad\xde'
        cls.eeg99s = EEG99s(12345, cls.data + cls.fletcher, 300)

    def test_data_too_long(self):
        payload = 2 * self.data + self.fletcher
        with self.assertRaises(Exception):
            EEG99s(12345, payload, 300)

    def test_convert(self):
        t = np.array([[11741.64, 170565.18],
                      [29388.7, 188212.24],
                      [47035.76, 205859.3],
                      [64682.82, 223506.36],
                      [82329.88, 241153.42],
                      [99976.94, 258800.48],
                      [117624.0, 276447.54],
                      [135271.06, 294094.6],
                      [152918.12, 311741.66]])
        np.testing.assert_array_equal(t, self.eeg99s.data[:, :2])

    # TODO implement status message check


class TestOrientationPacket(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = bytes([i % 256 for i in range(432)])
        cls.fletcher = b'\xaf\xbe\xad\xde'
        cls.orientation = Orientation(12345, cls.data + cls.fletcher, 300)

    def test_convert(self):
        self.fail()

    def test_get_data(self):
        self.fail()

    def test_compute_angle(self):
        self.fail()


class TestEnvironmentPacket(TestCase):
    def test_convert(self):
        self.fail()

    def test_get_data(self):
        self.fail()

    def test_volt_to_percent(self):
        self.fail()


class TestTimestampPacket(TestCase):
    # why does timestamp have a raw data var?

    def test_convert(self):
        self.fail()


class TestEventMarkerPacket(TestCase):
    def test_is_abstract(self):
        with self.assertRaises(Exception):
            EventMarker(12345, b'\xaf\xbe\xad\xde')

    def test_get_data(self):
        self.fail()


class TestPushButtonMarkerPacket(TestCase):
    def test_convert(self):
        self.fail()

    def test_prefix(self):
        self.fail()


class TestSoftwareMarkerPacket(TestCase):
    def test_convert(self):
        self.fail()

    def test_prefix(self):
        self.fail()

    def test_create(self):
        # This is like a second constructor
        self.fail()


class TestTriggerInPacket(TestCase):
    def test_prefix(self):
        self.fail()

    def test_convert(self):
        self.fail()


class TestTriggerOutPacket(TestCase):
    def test_prefix(self):
        self.fail()

    def test_convert(self):
        self.fail()


class TestDisconnectPacket(TestCase):
    def test_convert(self):
        self.fail()


class TestDeviceInfoPacket(TestCase):
    def test_convert(self):
        self.fail()

    def test_get_info(self):
        self.fail()

    def test_get_data(self):
        self.fail()


class TestCommandRCVPacket(TestCase):
    def test_convert(self):
        self.fail()


class TestCommandStatusPacket(TestCase):
    def test_convert(self):
        self.fail()


class TestCalibrationInfoPacket(TestCase):
    def test_convert(self):
        self.fail()

    def test_get_info(self):
        self.fail()


if __name__ == '__main__':
    unittest.main()
