import numpy as np
import pytest

import explorepy.packet
from explorepy.packet import (
    EEG,
    Environment,
    EventMarker,
    ExternalMarker,
    Packet,
    SoftwareMarker
)


EXPECTED_TIMESCALE = 10000


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


def xfail_on_unexpected_timescale():
    if explorepy.packet.TIMESTAMP_SCALE != EXPECTED_TIMESCALE:
        pytest.xfail(
            f"packet.py's TIMESTAMP_SCALE has changed. Expected: 10000, got: {explorepy.packet.TIMESTAMP_SCALE}")


def test_is_abstract(parametrized_abstract_packets):
    with pytest.raises(Exception):
        parametrized_abstract_packets(1234, b'\xff\xff\xff\xff', 0)


def test_abstract_timestamp_correct(mocker, parametrized_abstract_packets):
    xfail_on_unexpected_timescale()
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


def test_is_eeg(parametrized_eeg_in_out):
    eeg_instance = parametrized_eeg_in_out['eeg_instance']
    assert isinstance(eeg_instance, EEG)


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


def test_convert_orn(orientation_in_out):
    orn = orientation_in_out['orn_instance']
    orn_out = orientation_in_out['orn_out']
    np.testing.assert_array_equal(orn.acc, orn_out['acc'])
    np.testing.assert_array_equal(orn.gyro, orn_out['gyr'])
    np.testing.assert_array_equal(orn.mag, orn_out['mag'])


def test_get_data_orn(orientation_in_out):
    xfail_on_unexpected_timescale()
    orn = orientation_in_out['orn_instance']
    orn_out = orientation_in_out['orn_out']
    ts, samples = orn.get_data()
    assert [orn_out['raw_timestamp'] / EXPECTED_TIMESCALE] == ts
    ls = []
    ls.extend(orn_out['acc'])
    ls.extend(orn_out['gyr'])
    ls.extend(orn_out['mag'])
    np.testing.assert_array_almost_equal(samples, ls)


def test_compute_angle(compute_angle_in_out):
    test_object = compute_angle_in_out['orn_instance']
    print(f"Passed matrix: {compute_angle_in_out['matrix']}")
    theta_out, axis_out = test_object.compute_angle(compute_angle_in_out['matrix'])
    assert theta_out == compute_angle_in_out['theta']
    np.testing.assert_array_almost_equal(axis_out, compute_angle_in_out['axis'])


def test_check_fletcher_orn(orientation_in_out):
    orn = orientation_in_out['orn_instance']
    orn_out = orientation_in_out['orn_out']
    orn._check_fletcher(bytes.fromhex(orn_out['fletcher']))


def test_convert_env_temperature(env_in_out):
    assert env_in_out['env_instance'].temperature == env_in_out['env_out']['temperature']


def test_convert_env_light(env_in_out):
    assert env_in_out['env_instance'].light == env_in_out['env_out']['light']


def test_convert_env_battery(env_in_out):
    assert env_in_out['env_instance'].battery == env_in_out['env_out']['battery']


def test_volt_to_percent(env_in_out):
    expected = env_in_out['env_out']['battery_percentage']
    res = Environment._volt_to_percent(env_in_out['env_out']['battery'])
    assert res == int(expected)


def test_get_data_env(env_in_out):
    expected = {
        "battery": [int(env_in_out['env_out']['battery_percentage'])],
        "temperature": [env_in_out['env_out']['temperature']],
        "light": [env_in_out['env_out']['light']]
    }
    assert env_in_out['env_instance'].get_data() == expected


def test_check_fletcher_env(env_in_out):
    env = env_in_out['env_instance']
    env_out = env_in_out['env_out']
    env._check_fletcher(bytes.fromhex(env_out['fletcher']))


def test_convert_ts(ts_in_out):
    ts = ts_in_out['ts_instance']
    ts_out = ts_in_out['ts_out']
    assert ts.host_timestamp == ts_out['host_timestamp']


def test_check_fletcher_ts(ts_in_out):
    ts = ts_in_out['ts_instance']
    ts_out = ts_in_out['ts_out']
    ts._check_fletcher(bytes.fromhex(ts_out['fletcher']))


def test_marker_is_eventmarker(marker_in_out):
    marker = marker_in_out['marker_instance']
    assert isinstance(marker, EventMarker)


def test_convert_marker(marker_in_out):
    marker = marker_in_out['marker_instance']
    marker_out = marker_in_out['marker_out']
    assert marker.code == marker_out['marker']


def test_label_prefix_marker(marker_in_out):
    marker = marker_in_out['marker_instance']
    marker_out = marker_in_out['marker_out']
    assert marker._label_prefix == marker_out['label_prefix']


def test_check_fletcher_marker(marker_in_out):
    marker = marker_in_out['marker_instance']
    marker_out = marker_in_out['marker_out']
    marker._check_fletcher(bytes.fromhex(marker_out['fletcher']))


# TODO could change these to be one "create marker"?
# TODO possibly split in 3
@pytest.mark.parametrize("input_values,valid", [((12345, 0), True),
                                                ((0, 65535), True),
                                                ((42.42, 65536), False),
                                                ((12345, -1), False)])
def test_create_software_marker(input_values, valid):
    if not valid:
        with pytest.raises(Exception):
            SoftwareMarker.create(input_values[0], input_values[1])
    else:
        out = SoftwareMarker.create(input_values[0], input_values[1])
        assert out.code == input_values[1]
        assert out._label_prefix == "sw_"
        assert out.timestamp == input_values[0]


# TODO possibly split in 3
@pytest.mark.parametrize("input_values,valid", [((12345, "Experiment 0"), True),
                                                ((42.42, "Short marker"), True),
                                                ((12345, "Exp_1"), True),
                                                ((0, -1), False),
                                                ((0, "Marker that is way too long"), False),
                                                ((0, ""), False)])
def test_create_external_marker(input_values, valid):
    if not valid:
        with pytest.raises(Exception):
            ExternalMarker.create(input_values[0], input_values[1])
    else:
        out = ExternalMarker.create(input_values[0], input_values[1])
        assert out.code == input_values[1]
        assert out._label_prefix == "ext_"
        assert out.timestamp == input_values[0]


@pytest.mark.skip(reason="TriggerIn and TriggerOut not in use, no packets available")
def test_triggers_is_eventmarker(triggers_in_out):
    assert isinstance(triggers_in_out['triggers_instance'], EventMarker)


# TODO possibly split in 3
@pytest.mark.skip(reason="TriggerIn and TriggerOut not in use, no packets available")
def test_convert_triggers(triggers_in_out):
    trigger_instance = triggers_in_out['triggers_instance']
    trigger_out = triggers_in_out['triggers_out']
    assert trigger_instance.timestamp == trigger_out['timestamp']
    assert trigger_instance.code == trigger_out['code']
    assert trigger_instance.mac_address == trigger_out['mac_address']


@pytest.mark.skip(reason="TriggerIn and TriggerOut not in use, no packets available")
def test_triggers_check_fletcher(triggers_in_out):
    trigger_instance = triggers_in_out['triggers_instance']
    trigger_instance._check_fletcher(bytes.fromhex(triggers_in_out['fletcher']))


@pytest.mark.skip(reason="No Disconnect test data available")
def test_disconnect_check_fletcher(disconnect_in_out):
    disconnect_instance = disconnect_in_out['disconnect_instance']
    disconnect_out = disconnect_in_out['disconnect_out']
    disconnect_instance._check_fletcher(bytes.fromhex(disconnect_out['fletcher']))


def test_convert_device_info_fw(device_info_in_out):
    dev_info_instance = device_info_in_out['dev_info_instance']
    dev_info_out = device_info_in_out['dev_info_out']
    assert dev_info_instance.firmware_version == dev_info_out['fw_version']


def test_convert_device_info_sr(device_info_in_out):
    dev_info_instance = device_info_in_out['dev_info_instance']
    dev_info_out = device_info_in_out['dev_info_out']
    assert dev_info_instance.sampling_rate == dev_info_out['data_rate']


def test_convert_device_info_adc_mask(device_info_in_out):
    dev_info_instance = device_info_in_out['dev_info_instance']
    dev_info_out = device_info_in_out['dev_info_out']
    assert dev_info_instance.adc_mask == dev_info_out['adc_mask']


def test_device_info_get_data(device_info_in_out):
    dev_info_instance = device_info_in_out['dev_info_instance']
    dev_info_out = device_info_in_out['dev_info_out']
    assert dev_info_instance.get_data() == {'firmware_version': [dev_info_out['fw_version']]}


def test_convert_device_info_v2_board_id(device_info_v2_in_out):
    dev_info_v2_instance = device_info_v2_in_out['dev_info_v2_instance']
    dev_info_v2_out = device_info_v2_in_out['dev_info_v2_out']
    assert dev_info_v2_instance.board_id == dev_info_v2_out['board_id']


def test_convert_device_info_v2_fw(device_info_v2_in_out):
    dev_info_v2_instance = device_info_v2_in_out['dev_info_v2_instance']
    dev_info_v2_out = device_info_v2_in_out['dev_info_v2_out']
    assert dev_info_v2_instance.firmware_version == dev_info_v2_out['fw_version']


def test_convert_device_info_v2_sr(device_info_v2_in_out):
    dev_info_v2_instance = device_info_v2_in_out['dev_info_v2_instance']
    dev_info_v2_out = device_info_v2_in_out['dev_info_v2_out']
    assert dev_info_v2_instance.sampling_rate == dev_info_v2_out['data_rate']


def test_convert_device_info_v2_adc_mask(device_info_v2_in_out):
    dev_info_v2_instance = device_info_v2_in_out['dev_info_v2_instance']
    dev_info_v2_out = device_info_v2_in_out['dev_info_v2_out']
    assert dev_info_v2_instance.adc_mask == dev_info_v2_out['adc_mask']


def test_convert_device_info_v2_memory_available(device_info_v2_in_out):
    dev_info_v2_instance = device_info_v2_in_out['dev_info_v2_instance']
    dev_info_v2_out = device_info_v2_in_out['dev_info_v2_out']
    assert dev_info_v2_instance.is_memory_available == dev_info_v2_out['memory']


def test_device_info_v2_get_info(device_info_v2_in_out):
    dev_info_v2_instance = device_info_v2_in_out['dev_info_v2_instance']
    dev_info_v2_out = device_info_v2_in_out['dev_info_v2_out']
    out_dict = {
        'firmware_version': dev_info_v2_out['fw_version'],
        'adc_mask': dev_info_v2_out['adc_mask'],
        'sampling_rate': dev_info_v2_out['data_rate'],
        'board_id': dev_info_v2_out['board_id'],
        'memory_info': dev_info_v2_out['memory']
    }
    assert dev_info_v2_instance.get_info() == out_dict


def test_device_info_v2_get_data(device_info_v2_in_out):
    dev_info_v2_instance = device_info_v2_in_out['dev_info_v2_instance']
    dev_info_v2_out = device_info_v2_in_out['dev_info_v2_out']
    assert dev_info_v2_instance.get_data() == {'firmware_version': [dev_info_v2_out['fw_version']]}


def test_convert_cmd_rcv(cmd_rcv_in_out):
    cmd_rcv_instance = cmd_rcv_in_out['cmd_rcv_instance']
    cmd_rcv_out = cmd_rcv_in_out['cmd_rcv_out']
    assert cmd_rcv_instance.opcode == cmd_rcv_out['received_opcode']


def test_cmd_rcv_check_fletcher(cmd_rcv_in_out):
    cmd_rcv_instance = cmd_rcv_in_out['cmd_rcv_instance']
    cmd_rcv_out = cmd_rcv_in_out['cmd_rcv_out']
    cmd_rcv_instance._check_fletcher(bytes.fromhex(cmd_rcv_out['fletcher']))


def test_convert_cmd_stat_opcode(cmd_stat_in_out):
    cmd_stat_instance = cmd_stat_in_out['cmd_stat_instance']
    cmd_stat_out = cmd_stat_in_out['cmd_stat_out']
    assert cmd_stat_instance.opcode == cmd_stat_out['received_opcode']


def test_convert_cmd_stat_status(cmd_stat_in_out):
    cmd_stat_instance = cmd_stat_in_out['cmd_stat_instance']
    cmd_stat_out = cmd_stat_in_out['cmd_stat_out']
    assert cmd_stat_instance.status == cmd_stat_out['status']


def test_cmd_stat_check_fletcher(cmd_stat_in_out):
    cmd_stat_instance = cmd_stat_in_out['cmd_stat_instance']
    cmd_stat_out = cmd_stat_in_out['cmd_stat_out']
    cmd_stat_instance._check_fletcher(bytes.fromhex(cmd_stat_out['fletcher']))


def test_convert_calib_info_slope(calibration_info_in_out):
    calib_info_instance = calibration_info_in_out['calib_info_instance']
    calib_info_out = calibration_info_in_out['calib_info_out']
    assert calib_info_instance.slope == calib_info_out['slope']


def test_convert_calib_info_offset(calibration_info_in_out):
    calib_info_instance = calibration_info_in_out['calib_info_instance']
    calib_info_out = calibration_info_in_out['calib_info_out']
    assert float(calib_info_instance.offset) == calib_info_out['offset']


def test_calib_info_get_info(calibration_info_in_out):
    dict_out = {'slope': calibration_info_in_out['calib_info_out']['slope'],
                'offset': calibration_info_in_out['calib_info_out']['offset']}
    assert calibration_info_in_out['calib_info_instance'].get_info() == dict_out


def test_calib_info_check_fletcher(calibration_info_in_out):
    fletcher_out = bytes.fromhex(calibration_info_in_out['calib_info_out']['fletcher'])
    calibration_info_in_out['calib_info_instance']._check_fletcher(fletcher_out)


def test_calib_info_usbc_convert_slope(calibration_info_usbc_in_out):
    calib_info_usbc_instance = calibration_info_usbc_in_out['calib_info_usbc_instance']
    calib_info_usbc_out = calibration_info_usbc_in_out['calib_info_usbc_out']
    assert calib_info_usbc_instance.slope == calib_info_usbc_out['slope']


def test_calib_info_usbc_convert_offset(calibration_info_usbc_in_out):
    calib_info_usbc_instance = calibration_info_usbc_in_out['calib_info_usbc_instance']
    calib_info_usbc_out = calibration_info_usbc_in_out['calib_info_usbc_out']
    assert pytest.approx(calib_info_usbc_instance.offset) == calib_info_usbc_out['offset']


def test_calib_info_usbc_get_info(calibration_info_usbc_in_out):
    dict_out = {
        'slope': calibration_info_usbc_in_out['calib_info_usbc_out']['slope'],
        'offset': calibration_info_usbc_in_out['calib_info_usbc_out']['offset']
    }
    assert calibration_info_usbc_in_out['calib_info_usbc_instance'].get_info() == dict_out


def test_calib_info_usbc_check_fletcher(calibration_info_usbc_in_out):
    fletcher_out = bytes.fromhex(calibration_info_usbc_in_out['calib_info_usbc_out']['fletcher'])
    calibration_info_usbc_in_out['calib_info_usbc_instance']._check_fletcher(fletcher_out)
