import json
import os.path
import struct

import numpy as np
import pytest

from explorepy.packet import (
    EEG,
    EEG32,
    EEG94,
    EEG98,
    EEG98_USBC,
    EEG98_BLE,
    CalibrationInfo,
    CalibrationInfo_USBC,
    CommandRCV,
    CommandStatus,
    DeviceInfo,
    DeviceInfoV2,
    Disconnect,
    Environment,
    EventMarker,
    ExternalMarker,
    Orientation,
    Packet,
    PushButtonMarker,
    SoftwareMarker,
    TimeStamp,
    TriggerIn,
    TriggerOut
)


IN = "in"
OUT = "out"

EEG94_IN = os.path.join(IN, "eeg94")
EEG98_IN = os.path.join(IN, "eeg98")
EEG98_USBC_IN = os.path.join(IN, "eeg98_usbc")
EEG98_USBC_IN_2 = os.path.join(IN, "eeg98_usbc_2")
EEG32_IN = os.path.join(IN, "eeg32")
EEG98_BLE_IN = os.path.join(IN, "eeg98_ble")

ORN_IN = os.path.join(IN, "orn")
CMD_STAT_IN = os.path.join(IN, "cmd_stat")
DEV_INFO_IN = os.path.join(IN, "device_info")
DEV_INFO_V2_IN = os.path.join(IN, "device_info_v2")
ENV_IN = os.path.join(IN, "env")
TS_IN = os.path.join(IN, "ts")  # Doesn't exist
PUSH_MARKER_IN = os.path.join(IN, "push_marker")
SOFTWARE_MARKER_IN = os.path.join(IN, "software_marker")  # Doesn't exist
EXTERNAL_MARKER_IN = os.path.join(IN, "external_marker")  # Doesn't exist
TRIGGER_IN_IN = os.path.join(IN, "trigger_in")  # Doesn't exist
TRIGGER_OUT_IN = os.path.join(IN, "trigger_out")  # Doesn't exist
DISCONNECT_IN = os.path.join(IN, "disconnect")  # Doesn't exist
CMD_RCV_IN = os.path.join(IN, "cmd_rcv")
CALIB_INFO_IN = os.path.join(IN, "calibration_info")
CALIB_INFO_USBC_IN = os.path.join(IN, "calibration_info_usbc")

MATRIX_IN = os.path.join(IN, "orn_matrix.txt")

EEG94_OUT = os.path.join(OUT, "eeg94_out.txt")
EEG98_OUT = os.path.join(OUT, "eeg98_out.txt")
EEG98_USBC_OUT = os.path.join(OUT, "eeg98_usbc_out.txt")
EEG98_USBC_OUT_2 = os.path.join(OUT, "eeg98_usbc_out_2.txt")
EEG32_OUT = os.path.join(OUT, "eeg32_out.txt")
EEG98_BLE_OUT = os.path.join(OUT, "eeg98_ble_out.txt")

ORN_OUT = os.path.join(OUT, "orn_out.txt")
CMD_STAT_OUT = os.path.join(OUT, "cmd_stat_out.txt")
DEV_INFO_OUT = os.path.join(OUT, "device_info_out.txt")
DEV_INFO_V2_OUT = os.path.join(OUT, "device_info_v2_out.txt")
ENV_OUT = os.path.join(OUT, "env_out.txt")
TS_OUT = os.path.join(OUT, "ts_out.txt")  # Doesn't exist
PUSH_MARKER_OUT = os.path.join(OUT, "push_marker_out.txt")
SOFTWARE_MARKER_OUT = os.path.join(OUT, "software_marker_out.txt")  # Doesn't exist
EXTERNAL_MARKER_OUT = os.path.join(OUT, "external_marker_out.txt")
TRIGGER_IN_OUT = os.path.join(OUT, "trigger_in_out.txt")  # Doesn't exist
TRIGGER_OUT_OUT = os.path.join(OUT, "trigger_out_out.txt")  # Doesn't exist
DISCONNECT_OUT = os.path.join(OUT, "disconnect_out.txt")  # Doesn't exist
CMD_RCV_OUT = os.path.join(OUT, "cmd_rcv_out.txt")
CALIB_INFO_OUT = os.path.join(OUT, "calibration_info_out.txt")
CALIB_INFO_USBC_OUT = os.path.join(OUT, "calibration_info_usbc_out.txt")

MATRIX_OUT = os.path.join(OUT, "axis_and_angle.txt")

EEG_IN_OUT_LIST = [
    (EEG94, EEG94_IN, EEG94_OUT),
    (EEG98, EEG98_IN, EEG98_OUT),
    (EEG98_USBC, EEG98_USBC_IN, EEG98_USBC_OUT),
    (EEG98_USBC, EEG98_USBC_IN_2, EEG98_USBC_OUT_2),
    (EEG32, EEG32_IN, EEG32_OUT),
    (EEG98_BLE, EEG98_BLE_IN, EEG98_BLE_OUT)
]


def get_res_path(filename):
    parent_dir = os.path.dirname(os.path.realpath(__file__))
    res_path = os.path.join(parent_dir, 'res', filename)
    return res_path


# To check how explorepy reads, see explorepy.parser._generate_packet()
def read_bin_to_byte_string(path):
    f = open(path, "rb")
    byte_string = f.read()
    f.close()
    return byte_string


def string_to_byte_string(input_string):
    return bytes.fromhex(input_string)


def read_json_to_dict(source):
    with open(get_res_path(source), "r") as expected_output_file:
        return json.load(expected_output_file)


def string_list_to_hex_tuple(string_list):
    first_string = hex(int(string_list[0], 16))
    second_string = hex(int(string_list[1], 16))
    third_string = hex(int(string_list[2], 16))
    return first_string, second_string, third_string


def get_timestamp_from_byte_string(source):
    timestamp_bytes = source[4:8]
    timestamp_floating_point = struct.unpack('<I', timestamp_bytes)[0]
    return timestamp_floating_point


def eeg_in_out_list():
    return EEG_IN_OUT_LIST


def data_from_files(path_in, path_out, class_name, field_names, offset=0):
    bin_in = read_bin_to_byte_string(get_res_path(path_in))
    as_instance = class_name(get_timestamp_from_byte_string(bin_in), bin_in[8:], offset)
    dict_out = read_json_to_dict(get_res_path(path_out))
    data = dict()
    if 'instance' in field_names:
        data[field_names['instance']] = as_instance
    if 'out' in field_names:
        data[field_names['out']] = dict_out
    if 'in' in field_names:
        data[field_names['in']] = bin_in
    if 'class_name' in field_names:
        data[field_names['class_name']] = class_name
    return data


@pytest.fixture(params=[Packet, EEG, EventMarker], scope="module")
def parametrized_abstract_packets(request):
    return request.param


@pytest.fixture(scope="function")
def mocked_eeg_base(mocker):
    if hasattr(EEG, "__abstractmethods__"):
        if len(EEG.__abstractmethods__) != 0:
            mocker.patch.multiple(EEG, __abstractmethods__=set())
    eeg = EEG(12345, b'\x00\x00\x00\xaf\xbe\xad\xde', v_ref=2.4, n_packet=1)
    eeg.data = np.array(
        [[40, 3333, 78910, -30, 0],
         [20, -1000, 10, 30, 0],
         [10, 2345, 77016, 11, 45],
         [15, 1234, 70000, 2, 44]])
    return eeg


@pytest.fixture(params=[
    ([10, 20, 30, 40, 50, 60], [1971210, 3945000]),
    ([255, 255, 255, 255, 255, 255], [-1, -1]),  # [111111111111111111111111, 111111111111111111111111]
    ([0, 0, 0, 0, 0, 0], [0, 0]),
    ([0, 0, 128, 0, 0, 128], [-8388608, -8388608])],  # [100000000000000000000000, 100000000000000000000000]
    scope="module")
def parametrized_int24toint32_in_out(request):
    return request.param


@pytest.fixture(params=eeg_in_out_list(), scope="module")
def parametrized_eeg_in_out(request):
    """Provides objects containing an instance of an EEG packet, input data, expected output data and class name.
    Note that the instances distributed with this method have scope "module", meaning changes to the instance are
    persistent throughout all following tests!
    """
    field_names = {'class_name': 'eeg_class',
                   'in': 'eeg_in',
                   'instance': 'eeg_instance',
                   'out': 'eeg_out'}
    return data_from_files(request.param[1], request.param[2], request.param[0], field_names)


@pytest.fixture(params=[(ORN_IN, ORN_OUT)], scope="module")
def orientation_in_out(request):
    field_names = {'instance': 'orn_instance',
                   'out': 'orn_out'}
    return data_from_files(request.param[0], request.param[1], Orientation, field_names)


# Note that ORN_IN is only necessary because compute_angle is not static,
# but the instance isn't used in the method
# compute_angle could be split into a static compute_angle function and a set_angle method
@pytest.fixture(params=[(MATRIX_IN, MATRIX_OUT, ORN_IN)], scope="module")
def compute_angle_in_out(request):
    matrix_in = read_json_to_dict(get_res_path(request.param[0]))['matrix']
    angle_out = read_json_to_dict(get_res_path(request.param[1]))
    orn_in = read_bin_to_byte_string(get_res_path(request.param[2]))
    orn_instance = Orientation(get_timestamp_from_byte_string(orn_in), orn_in[8:], 0)
    data = {
        'matrix': matrix_in,
        'orn_instance': orn_instance,
        'axis': angle_out['axis'],
        'theta': angle_out['theta']
    }
    return data


@pytest.fixture(params=[(ENV_IN, ENV_OUT)])
def env_in_out(request):
    field_names = {'instance': 'env_instance',
                   'out': 'env_out'}
    return data_from_files(request.param[0], request.param[1], Environment, field_names)


@pytest.fixture(params=[(TS_IN, TS_OUT)])
def ts_in_out(request):
    try:
        field_names = {'instance': 'ts_instance',
                       'out': 'ts_out'}
        return data_from_files(request.param[0], request.param[1], TimeStamp, field_names)
    except FileNotFoundError:
        pytest.skip("TimeStamp input or output file not available")


@pytest.fixture(params=[(PushButtonMarker, PUSH_MARKER_IN, PUSH_MARKER_OUT),
                        (SoftwareMarker, SOFTWARE_MARKER_IN, SOFTWARE_MARKER_OUT),
                        (ExternalMarker, EXTERNAL_MARKER_IN, EXTERNAL_MARKER_OUT)])
def marker_in_out(request):
    try:
        field_names = {'instance': 'marker_instance',
                       'out': 'marker_out'}
        return data_from_files(request.param[1], request.param[2], request.param[0], field_names)
    except FileNotFoundError:
        pytest.skip(f"Input or output file not available for {request.param[0]}")


@pytest.fixture(params=[(12345, 0),
                        (0, 65535)])
def sw_marker_inputs_valid(request):
    return request.param


@pytest.fixture(params=[(42.42, 65536),
                        (12345, -1)])
def sw_marker_inputs_invalid(request):
    return request.param


@pytest.fixture(params=[(12345, "Exp 0"),
                        (42.42, "Short"),
                        (12345, "A")])
def ext_marker_inputs_valid(request):
    return request.param


@pytest.fixture(params=[(0, -1),
                        (0, "Marker that is way too long"),
                        (0, "")])
def ext_marker_inputs_invalid(request):
    return request.param


@pytest.fixture(params=[(TriggerIn, TRIGGER_IN_IN, TRIGGER_IN_OUT),
                        (TriggerOut, TRIGGER_OUT_IN, TRIGGER_OUT_OUT)])
def triggers_in_out(request):
    field_names = {'instance': 'triggers_instance',
                   'out': 'triggers_out'}
    return data_from_files(request.param[1], request.param[2], request.param[0], field_names)


@pytest.fixture(params=[(DISCONNECT_IN, DISCONNECT_OUT)])
def disconnect_in_out(request):
    field_names = {'instance': 'disconnect_instance',
                   'out': 'disconnect_out'}
    return data_from_files(request.param[0], request.param[1], Disconnect, field_names)


@pytest.fixture(params=[(DEV_INFO_IN, DEV_INFO_OUT)])
def device_info_in_out(request):
    field_names = {'instance': 'dev_info_instance',
                   'out': 'dev_info_out'}
    return data_from_files(request.param[0], request.param[1], DeviceInfo, field_names)


@pytest.fixture(params=[(DEV_INFO_V2_IN, DEV_INFO_V2_OUT)])
def device_info_v2_in_out(request):
    field_names = {'instance': 'dev_info_v2_instance',
                   'out': 'dev_info_v2_out'}
    return data_from_files(request.param[0], request.param[1], DeviceInfoV2, field_names)


@pytest.fixture(params=[(CMD_RCV_IN, CMD_RCV_OUT)])
def cmd_rcv_in_out(request):
    field_names = {
        'instance': 'cmd_rcv_instance',
        'out': 'cmd_rcv_out'
    }
    return data_from_files(request.param[0], request.param[1], CommandRCV, field_names)


@pytest.fixture(params=[(CMD_STAT_IN, CMD_STAT_OUT)])
def cmd_stat_in_out(request):
    field_names = {
        'instance': 'cmd_stat_instance',
        'out': 'cmd_stat_out'
    }
    return data_from_files(request.param[0], request.param[1], CommandStatus, field_names)


@pytest.fixture(params=[(CALIB_INFO_IN, CALIB_INFO_OUT)])
def calibration_info_in_out(request):
    field_names = {
        'instance': 'calib_info_instance',
        'out': 'calib_info_out'
    }
    return data_from_files(request.param[0], request.param[1], CalibrationInfo, field_names)


@pytest.fixture(params=[(CALIB_INFO_USBC_IN, CALIB_INFO_USBC_OUT)])
def calibration_info_usbc_in_out(request):
    field_names = {
        'instance': 'calib_info_usbc_instance',
        'out': 'calib_info_usbc_out'
    }
    return data_from_files(request.param[0], request.param[1], CalibrationInfo_USBC, field_names)
