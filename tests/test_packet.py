import numpy as np
import pytest

import explorepy.packet
from explorepy.packet import Packet


def read_bin_to_byte_string(path):
    f = open(path, "rb")
    byte_string = f.read()
    f.close()

    return byte_string


def get_first_status_as_tuple(status_list):
    first_byte = hex(int(status_list[0][:2], 16))
    second_byte = hex(int(status_list[0][2:4], 16))
    third_byte = hex(int(status_list[0][4:6], 16))
    return first_byte, second_byte, third_byte


def test_is_abstract(parametrized_abstract_packets):
    with pytest.raises(Exception):
        parametrized_abstract_packets(1234, b'\xff\xff\xff\xff', 0)


def test_abstract_timestamp_correct(mocker, parametrized_abstract_packets):
    if explorepy.packet.TIMESTAMP_SCALE != 10000:
        pytest.xfail(
            f"packet.py's TIMESTAMP_SCALE has changed. Expected: 10000, got: {explorepy.packet.TIMESTAMP_SCALE}")
    if hasattr(parametrized_abstract_packets, "__abstractmethods__"):
        if len(parametrized_abstract_packets.__abstractmethods__) != 0:
            mocker.patch.multiple(parametrized_abstract_packets, __abstractmethods__=set())
    p = parametrized_abstract_packets(12345, b'\x00\x00\x00\x00', 300)
    assert p.timestamp == 301.2345


def test_int24to32(parametrized_int24toint32_in_out):
    list_in = Packet.int24to32(parametrized_int24toint32_in_out[0])
    list_out = parametrized_int24toint32_in_out[1]
    np.testing.assert_array_equal(list_in, list_out)


def test_calculate_impedance_no_info(mocked_eeg_base):
    imp_calib_info = {'slope': None, 'offset': None, 'noise_level': None}
    with pytest.raises(Exception):
        mocked_eeg_base.calculate_impedance(imp_calib_info)


def test_get_data(mocked_eeg_base):
    tv, d = mocked_eeg_base.get_data(250)
    assert len(tv) == 5
    np.testing.assert_array_equal(tv, [1.2345, 1.2385, 1.2425, 1.2465, 1.2505])
    np.testing.assert_array_equal(d, mocked_eeg_base.data)


def test_get_data_no_sample_rate(mocked_eeg_base):
    tv, d = mocked_eeg_base.get_data()
    assert tv == 1.2345
    np.testing.assert_array_equal(d, mocked_eeg_base.data)


def test_ptp(mocked_eeg_base):
    res = mocked_eeg_base.get_ptp()
    np.testing.assert_array_equal(res, [78940, 1030, 77006, 69998])


def test_ptp_no_data(mocked_eeg_base):
    mocked_eeg_base.data = b'\xaf\xbe\xad\xde'
    with pytest.raises(Exception):
        mocked_eeg_base.get_ptp()


def test_status(parametrized_eeg_in_out):
    """
    Tests if the status messages have been converted correctly.
    Currently expected to fail for every EEG packet due to only the first three status bytes being saved.
    For EEG94, the property isn't called status but data_status. I could test for this, but the better solution is to
    unify the name of the property. The test will still fail if data_status in EEG94 is renamed to status, since
    the status messages are saved as numbers instead of i.e. strings or byte strings.
    """
    eeg = parametrized_eeg_in_out['eeg_instance']
    eeg_out = parametrized_eeg_in_out['eeg_out']

    if not hasattr(eeg, 'status'):
        class_type = parametrized_eeg_in_out['eeg_class']
        pytest.xfail(f"{str(class_type)} object has no property called \'status\'")

    if len(eeg.status) < len(eeg_out['status']):
        class_type = parametrized_eeg_in_out['eeg_class']
        pytest.xfail(f"{str(class_type)} object's status property doesn't contain all status messages")

    assert eeg.data_status == eeg_out['status']


def test_convert(parametrized_eeg_in_out):
    eeg = parametrized_eeg_in_out['eeg_instance']
    eeg_out = parametrized_eeg_in_out['eeg_out']
    np.testing.assert_array_equal(eeg.data, eeg_out['samples'])


def test_check_fletcher(parametrized_eeg_in_out):
    eeg = parametrized_eeg_in_out['eeg_instance']
    eeg_out = parametrized_eeg_in_out['eeg_out']
    eeg._check_fletcher(bytes.fromhex(eeg_out['fletcher']))
