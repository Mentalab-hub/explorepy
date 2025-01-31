import copy

import numpy as np
import pytest

import explorepy.packet
from explorepy.packet import (
    EEG,
    EEG_BLE,
    EEG32_BLE,
    Environment,
    TimeStamp,
    EventMarker,
    ExternalMarker,
    Packet,
    PacketBIN,
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


def test_is_abstract(parametrized_abstract_packets):
    with pytest.raises(Exception):
        parametrized_abstract_packets(1234, b'\xff\xff\xff\xff', 0)


def test_abstract_timestamp_correct(mocker, parametrized_abstract_packets):
    if hasattr(parametrized_abstract_packets, "__abstractmethods__"):
        if len(parametrized_abstract_packets.__abstractmethods__) != 0:
            mocker.patch.multiple(parametrized_abstract_packets, __abstractmethods__=set())
    start = 12345
    offset = 300
    if parametrized_abstract_packets == EEG:
        p = parametrized_abstract_packets(start, b'\x00\x00\x00\xaf\xbe\xad\xde', offset, v_ref=2.4, n_packet=1)
    else:
        p = parametrized_abstract_packets(start, b'\xaf\xbe\xad\xde', offset)
    assert p.timestamp == (start+offset)


def test_int24to32(parametrized_int24toint32_in_out):
    list_in = Packet.int24to32(parametrized_int24toint32_in_out[0])
    list_out = parametrized_int24toint32_in_out[1]
    np.testing.assert_array_equal(list_in, list_out)


def test_calculate_impedance_no_info(mocked_eeg_base):
    imp_calib_info = {'slope': None, 'offset': None, 'noise_level': None}
    with pytest.raises(Exception):
        mocked_eeg_base.calculate_impedance(imp_calib_info)

def test_calculate_impedance(parametrized_eeg_in_out):
    out = parametrized_eeg_in_out["eeg_out"]
    if "slope" not in out or "offset" not in out or "noise_level" not in out or "impedances" not in out:
        pytest.xfail("Impedance calibration parameters and/or expected output are not known for this test EEG packet.")
    imp_calib_info = {'slope': out["slope"], 'offset': out["offset"], 'noise_level': out["noise_level"]}
    eeg = copy.deepcopy(parametrized_eeg_in_out["eeg_instance"])
    eeg.calculate_impedance(imp_calib_info)
    np.testing.assert_allclose(eeg.get_impedances(), out["impedances"])

def test_get_impedances_none(parametrized_eeg_in_out):
    eeg = parametrized_eeg_in_out["eeg_instance"]
    assert eeg.get_impedances() is None


def test_get_data(mocked_eeg_base):
    tv, d = mocked_eeg_base.get_data(250)
    assert len(tv) == 5
    np.testing.assert_allclose(tv, [12345., 12345.004, 12345.008, 12345.012, 12345.016])
    np.testing.assert_array_equal(d, mocked_eeg_base.data)


def test_get_data_no_sample_rate(mocked_eeg_base):
    tv, d = mocked_eeg_base.get_data()
    assert tv == 12345
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


def test_packetbin_bin_data(parametrized_eeg_in_out):
    raw_data = parametrized_eeg_in_out['eeg_in']
    packetbin = PacketBIN(raw_data)
    assert raw_data == packetbin.bin_data


def test_packetbin_convert(parametrized_eeg_in_out):
    raw_data = parametrized_eeg_in_out['eeg_in']
    packetbin = PacketBIN(raw_data)
    packetbin._convert(raw_data)
    assert not hasattr(packetbin, "data")
    assert raw_data == packetbin.bin_data


def test_packetbin_str(parametrized_eeg_in_out):
    raw_data = parametrized_eeg_in_out['eeg_in']
    packet_bin = PacketBIN(raw_data)
    packet_bin_out = parametrized_eeg_in_out['eeg_out']
    if 'bin_string' in packet_bin_out:
        assert bytes(packet_bin_out['bin_string'], "utf-8").__str__() == packet_bin.__str__()


def test_byteorder_data(parametrized_eeg_in_out):
    eeg = parametrized_eeg_in_out['eeg_instance']
    if isinstance(eeg, EEG_BLE):
        assert eeg.byteorder_data == "big"
    else:
        assert eeg.byteorder_data == "little"


def test_convert_errors_v_ref(parametrized_eeg_in_out):
    eeg = copy.deepcopy(parametrized_eeg_in_out["eeg_instance"])
    eeg.v_ref = None
    with pytest.raises(ValueError, match="v_ref or n_packet cannot be null for conversion!"):
        eeg._convert(parametrized_eeg_in_out["eeg_in"][8:-4])

def test_convert_errors_n_packet(parametrized_eeg_in_out):
    eeg = copy.deepcopy(parametrized_eeg_in_out["eeg_instance"])
    eeg.n_packet = None
    with pytest.raises(ValueError, match="v_ref or n_packet cannot be null for conversion!"):
        eeg._convert(parametrized_eeg_in_out["eeg_in"][8:-4])


@pytest.mark.parametrize("byte_order", [(None, TypeError),
                                        (0, TypeError),
                                        ("LITTLE", ValueError),
                                        ("BIG", ValueError),
                                        ("l", ValueError),
                                        ("b", ValueError),
                                        (-1, TypeError),
                                        (True, TypeError),
                                        (False, TypeError)])
def test_convert_errors_byteorder(parametrized_eeg_in_out, byte_order):
    eeg = copy.deepcopy(parametrized_eeg_in_out["eeg_instance"])
    eeg.byteorder_data = byte_order[0]
    with pytest.raises(byte_order[1]):
        eeg._convert(parametrized_eeg_in_out["eeg_in"][8:-4])


def test_status(parametrized_eeg_in_out):
    eeg = parametrized_eeg_in_out['eeg_instance']
    eeg_out = parametrized_eeg_in_out['eeg_out']
    if isinstance(eeg, EEG_BLE):
        with pytest.raises(AttributeError):
            eeg.status
    else:
        status_out = {
            'ads': eeg_out['status_ads'],
            'empty': eeg_out['status_empty'],
            'sr': eeg_out['status_sr']
        }
        np.testing.assert_array_equal(eeg.status['ads'], status_out['ads'])
        np.testing.assert_array_equal(eeg.status['empty'], status_out['empty'])
        np.testing.assert_array_equal(eeg.status['sr'], status_out['sr'])

def test_convert(parametrized_eeg_in_out):
    eeg = parametrized_eeg_in_out['eeg_instance']
    eeg_out = parametrized_eeg_in_out['eeg_out']
    np.testing.assert_array_equal(eeg.data, eeg_out['samples'])


def test_check_fletcher(parametrized_eeg_in_out):
    eeg = parametrized_eeg_in_out['eeg_instance']
    eeg_out = parametrized_eeg_in_out['eeg_out']
    eeg._check_fletcher(bytes.fromhex(eeg_out['fletcher']))


@pytest.mark.parametrize('fletcher', [b"\x01\x02\x03\x04",
                                      b"\x00\x00\x00\x00",
                                      b"\xff\xff\xff\xff"])
def test_check_fletcher_invalid(parametrized_eeg_in_out, fletcher):
    with pytest.raises(Exception):
        eeg = parametrized_eeg_in_out['eeg_instance']
        eeg._check_fletcher(fletcher)


def test_convert_orn(orientation_in_out):
    orn = orientation_in_out['orn_instance']
    orn_out = orientation_in_out['orn_out']
    np.testing.assert_array_equal(orn.acc, orn_out['acc'])
    np.testing.assert_array_equal(orn.gyro, orn_out['gyr'])
    np.testing.assert_array_equal(orn.mag, orn_out['mag'])


def test_get_data_orn(orientation_in_out):
    orn = orientation_in_out['orn_instance']
    orn_out = orientation_in_out['orn_out']
    ts, samples = orn.get_data()
    assert [orn_out['raw_timestamp']] == ts
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


def test_create_software_marker_invalid(sw_marker_inputs_invalid):
    with pytest.raises(Exception):
        SoftwareMarker.create(sw_marker_inputs_invalid[0], sw_marker_inputs_invalid[1])


def test_create_software_marker_code(sw_marker_inputs_valid):
    out = SoftwareMarker.create(sw_marker_inputs_valid[0], sw_marker_inputs_valid[1])
    assert out.code == sw_marker_inputs_valid[1]


def test_create_software_marker_prefix(sw_marker_inputs_valid):
    out = SoftwareMarker.create(sw_marker_inputs_valid[0], sw_marker_inputs_valid[1])
    assert out._label_prefix == "sw_"


def test_create_software_marker_ts(sw_marker_inputs_valid):
    out = SoftwareMarker.create(sw_marker_inputs_valid[0], sw_marker_inputs_valid[1])
    assert out.timestamp == sw_marker_inputs_valid[0] * 10000


def test_create_external_marker_invalid(ext_marker_inputs_invalid):
    with pytest.raises(Exception):
        ExternalMarker.create(ext_marker_inputs_invalid[0], ext_marker_inputs_invalid[1])


def test_create_external_marker_code(ext_marker_inputs_valid):
    out = ExternalMarker.create(ext_marker_inputs_valid[0], ext_marker_inputs_valid[1])
    assert out.code == ext_marker_inputs_valid[1]


def test_create_external_marker_prefix(ext_marker_inputs_valid):
    out = ExternalMarker.create(ext_marker_inputs_valid[0], ext_marker_inputs_valid[1])
    assert out._label_prefix == "sw_"


def test_create_external_marker_ts(ext_marker_inputs_valid):
    out = ExternalMarker.create(ext_marker_inputs_valid[0], ext_marker_inputs_valid[1])
    assert out.timestamp == ext_marker_inputs_valid[0]


@pytest.mark.skip(reason="TriggerIn and TriggerOut not in use, no packets available")
def test_triggers_is_eventmarker(triggers_in_out):
    assert isinstance(triggers_in_out['triggers_instance'], EventMarker)


@pytest.mark.skip(reason="TriggerIn and TriggerOut not in use, no packets available")
def test_convert_triggers_ts(triggers_in_out):
    trigger_instance = triggers_in_out['triggers_instance']
    trigger_out = triggers_in_out['triggers_out']
    assert trigger_instance.timestamp == trigger_out['timestamp']


@pytest.mark.skip(reason="TriggerIn and TriggerOut not in use, no packets available")
def test_convert_triggers_code(triggers_in_out):
    trigger_instance = triggers_in_out['triggers_instance']
    trigger_out = triggers_in_out['triggers_out']
    assert trigger_instance.code == trigger_out['code']


@pytest.mark.skip(reason="TriggerIn and TriggerOut not in use, no packets available")
def test_convert_triggers_mac_address(triggers_in_out):
    trigger_instance = triggers_in_out['triggers_instance']
    trigger_out = triggers_in_out['triggers_out']
    assert trigger_instance.mac_address == trigger_out['mac_address']


@pytest.mark.skip(reason="TriggerIn and TriggerOut not in use, no packets available")
def test_triggers_check_fletcher(triggers_in_out):
    trigger_instance = triggers_in_out['triggers_instance']
    trigger_instance._check_fletcher(bytes.fromhex(triggers_in_out['fletcher']))


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


@pytest.mark.skip("get_data has been removed from DeviceInfo")
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
        'memory_info': dev_info_v2_out['memory'],
        'is_imp_mode': False
    }
    assert dev_info_v2_instance.get_info() == out_dict


@pytest.mark.skip("get_data has been removed from DeviceInfoV2")
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
    assert pytest.approx(calibration_info_usbc_in_out['calib_info_usbc_instance'].get_info()) == dict_out


def test_calib_info_usbc_check_fletcher(calibration_info_usbc_in_out):
    fletcher_out = bytes.fromhex(calibration_info_usbc_in_out['calib_info_usbc_out']['fletcher'])
    calibration_info_usbc_in_out['calib_info_usbc_instance']._check_fletcher(fletcher_out)
