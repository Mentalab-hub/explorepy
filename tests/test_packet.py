import pytest
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


class TestBasePacket:

    def test_is_abstract(self):
        with pytest.raises(Exception):
            Packet(12345, b'\x00\x00\x00\x00')

    def test_init_timestamp_correct(self, mocker):
        if hasattr(Packet, "__abstractmethods__"):
            if len(Packet.__abstractmethods__) != 0:
                mocker.patch.multiple(Packet, __abstractmethods__=set())
        p = Packet(12345, b'\x00\x00\x00\x00', 300)
        assert p.timestamp == 301.2345

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


class TestEEGPacket:

    @pytest.fixture(autouse=True)
    def setup_eeg(self, mocker):
        if hasattr(EEG, "__abstractmethods__"):
            if len(EEG.__abstractmethods__) != 0:
                mocker.patch.multiple(EEG, __abstractmethods__=set())
        self.eeg = EEG(12345, b'\xaf\xbe\xad\xde')
        self.eeg.data = np.array(
            [[40, 3333, 78910, -30, 0], [20, -1000, 10, 30, 0], [10, 2345, 77016, 11, 45], [15, 1234, 70000, 2, 44]])

    def test_is_abstract(self):
        with pytest.raises(Exception):
            EEG(12345, b'\x00\x00\x00\x00')

    def test_calculate_impedance_no_data(self):
        imp_calib_info = {'slope': None, 'offset': None, 'noise_level': None}
        with pytest.raises(Exception):
            self.eeg.calculate_impedance(imp_calib_info)

    def test_get_data(self):
        tv, d = self.eeg.get_data(250)
        assert len(tv) == 5
        np.testing.assert_array_equal(tv, [1.2345, 1.2385, 1.2425, 1.2465])
        np.testing.assert_array_equal(d, self.eeg.data)

    def test_get_data_no_sample_rate(self):
        tv, d = self.eeg.get_data()
        assert tv == 1.2345
        np.testing.assert_array_equal(d, self.eeg.data)

    def test_ptp(self):
        res = self.eeg.get_ptp()
        np.testing.assert_array_equal(res, [78940, 1030, 77006, 69998])

    def test_ptp_no_data(self):
        self.eeg.data = b'\xaf\xbe\xad\xde'
        with pytest.raises(Exception):
            self.eeg.get_ptp()


# The data part of the packet has 507 - 12 bytes of data, with 3 bytes per channel, 4 channels, 1 status msg
# with 4 channels + 1 status message @ 3 bytes each being 15 bytes, so (507-12)/(5*3) = 33 rows of data
# (I think)
# Note that this does *not* test the proper fletcher implementation
class TestEEG94Packet:

    @pytest.fixture(autouse=True)
    def setup_eeg94(self):
        # 99 bytes of data + 4 bytes fletcher (0xDEADBEAF)
        self.data = bytes(
            [i % 256 for i in range(495)])  # 495 bytes of dummy data, resembles 33 samples in 4+1 channels
        self.fletcher = b'\xaf\xbe\xad\xde'
        self.eeg94 = EEG94(12345, self.data + self.fletcher, 300)

    # EEG94 converts the data on initialization,
    # if the packet size is fixed only a specific length of bytes should be accepted
    def test_data_length_too_long(self):
        payload = 2 * self.data + self.fletcher
        with pytest.raises(Exception):
            EEG94(12345, payload, 300)

    def test_convert_data(self):
        t = np.array([[15673.97, 62732.8, 109791.63],
                      [25085.74, 72144.56, 119203.39],
                      [34497.5, 81556.33, 128615.16],
                      [43909.27, 90968.1, 138026.92]])
        np.testing.assert_array_equal(self.eeg94.data[:, :3], t)

    @pytest.mark.skip
    def test_convert_status(self):
        t = np.array([15673.97, 62732.8, 109791.63])
        np.testing.assert_array_equal(self.eeg94.data_status[:3], t)


class TestEEG98Packet:

    @pytest.fixture(autouse=True)
    def setup_eeg98_real(self, eeg8_test_samples, eeg8_test_timestamp, eeg8_test_fletcher, eeg8_test_status):
        self.eeg98_real = EEG98(eeg8_test_timestamp, eeg8_test_samples + eeg8_test_fletcher, 0)

    @pytest.fixture(autouse=True)
    def setup_eeg98_fake(self):
        self.fake_data = bytes([i % 256 for i in range(432)])
        self.fake_timestamp = 12345
        self.fake_fletcher = b'\xaf\xbe\xad\xde'
        self.fake_status = ('0x0', '0x1', '0x2')
        self.eeg98_fake = EEG98(self.fake_timestamp, self.fake_data + self.fake_fletcher, 0)

    def test_data_too_long(self):
        payload = 2 * self.fake_data + self.fake_fletcher
        with pytest.raises(Exception):
            EEG98(self.fake_timestamp, payload, 300)

    def test_convert_fake(self, eeg8_expected_samples_fake):
        # t = np.array([[15673.97, 100379.86],
        #               [25085.74, 109791.63],
        #               [34497.5, 119203.39],
        #               [43909.27, 128615.16],
        #               [53321.03, 138026.92],
        #               [62732.8, 147438.69],
        #               [72144.56, 156850.45],
        #               [81556.33, 166262.22]])
        print(self.eeg98_fake.data)
        print("Shape")
        print(self.eeg98_fake.data.shape)
        for i in range(self.eeg98_fake.data.shape[0]):
            for j in range(self.eeg98_fake.data.shape[1]):
                if self.eeg98_fake.data[i][j] != eeg8_expected_samples_fake[i][j]:
                    print(f"Samples are different at ({i}, {j})")
                    print(f"Internally: {self.eeg98_fake.data[i][j]}, from json: {eeg8_expected_samples_fake[i][j]}")
        np.testing.assert_array_almost_equal(self.eeg98_fake.data, eeg8_expected_samples_fake)

    def test_convert_status_fake(self, eeg8_expected_status_fake):
        assert eeg8_expected_status_fake == self.eeg98_fake.status

    def test_convert_real(self, eeg8_expected_samples):
        np.testing.assert_array_equal(self.eeg98_real.data, eeg8_expected_samples)

    def test_convert_status_real(self, eeg8_expected_status):
        # Note: the only status message that is actually considered is the first one
        # (despite there being a status message per 8 channel samples
        # so, 16 status messages for an EEG98 packet
        assert eeg8_expected_status == self.eeg98_real.status


class TestEEG99Packet:

    @pytest.fixture(autouse=True)
    def setup_eeg99(self):
        self.data = bytes([i % 256 for i in range(432)])
        self.fletcher = b'\xaf\xbe\xad\xde'
        self.eeg99 = EEG99(12345, self.data + self.fletcher, 300)

    def test_data_too_long(self):
        payload = 2 * self.data + self.fletcher
        with pytest.raises(Exception):
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


class TestEEG99sPacket:

    @pytest.fixture(autouse=True)
    def setup_eeg99s(self):
        self.data = bytes([i % 256 for i in range(432)])
        self.fletcher = b'\xaf\xbe\xad\xde'
        self.eeg99s = EEG99s(12345, self.data + self.fletcher, 300)

    def test_data_too_long(self):
        payload = 2 * self.data + self.fletcher
        with pytest.raises(Exception):
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


class TestOrientationPacket:

    @pytest.fixture(autouse=True)
    def setup_orientation(self):
        self.data = bytes([i % 256 for i in range(432)])
        self.fletcher = b'\xaf\xbe\xad\xde'
        self.orientation = Orientation(12345, self.data + self.fletcher, 300)

    @pytest.mark.skip(reason="Not implemented")
    def test_convert(self):
        self.fail()

    @pytest.mark.skip(reason="Not implemented")
    def test_get_data(self):
        self.fail()

    @pytest.mark.skip(reason="Not implemented")
    def test_compute_angle(self):
        self.fail()


@pytest.mark.skip(reason="Not implemented")
class TestEnvironmentPacket:
    def test_convert(self):
        self.fail()

    def test_get_data(self):
        self.fail()

    def test_volt_to_percent(self):
        self.fail()


@pytest.mark.skip(reason="Not implemented")
class TestTimestampPacket:
    # why does timestamp have a raw data var?

    def test_convert(self):
        self.fail()


class TestEventMarkerPacket:

    def test_is_abstract(self):
        with pytest.raises(Exception):
            EventMarker(12345, b'\xaf\xbe\xad\xde')

    @pytest.mark.skip(reason="Not implemented")
    def test_get_data(self):
        self.fail()


@pytest.mark.skip(reason="Not implemented")
class TestPushButtonMarkerPacket:
    def test_convert(self):
        self.fail()

    def test_prefix(self):
        self.fail()


@pytest.mark.skip(reason="Not implemented")
class TestSoftwareMarkerPacket:
    def test_convert(self):
        self.fail()

    def test_prefix(self):
        self.fail()

    def test_create(self):
        # This is like a second constructor
        self.fail()


@pytest.mark.skip(reason="Not implemented")
class TestTriggerInPacket:
    def test_prefix(self):
        self.fail()

    def test_convert(self):
        self.fail()


@pytest.mark.skip(reason="Not implemented")
class TestTriggerOutPacket:
    def test_prefix(self):
        self.fail()

    def test_convert(self):
        self.fail()


@pytest.mark.skip(reason="Not implemented")
class TestDisconnectPacket:
    def test_convert(self):
        self.fail()


@pytest.mark.skip(reason="Not implemented")
class TestDeviceInfoPacket:
    def test_convert(self):
        self.fail()

    def test_get_info(self):
        self.fail()

    def test_get_data(self):
        self.fail()


@pytest.mark.skip(reason="Not implemented")
class TestCommandRCVPacket:
    def test_convert(self):
        self.fail()


@pytest.mark.skip(reason="Not implemented")
class TestCommandStatusPacket:
    def test_convert(self):
        self.fail()


@pytest.mark.skip(reason="Not implemented")
class TestCalibrationInfoPacket:
    def test_convert(self):
        self.fail()

    def test_get_info(self):
        self.fail()
