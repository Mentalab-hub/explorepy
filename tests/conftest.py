import json
import os
import struct

import numpy as np
import pytest

import pandas as pd

from explorepy.packet import (
    DeviceInfo,
    DeviceInfoV2,
    Disconnect,
    EEG,
    EEG32,
    EEG94,
    EEG98,
    EEG98_USBC,
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

MATRIX_IN = os.path.join(IN, "orn_matrix.txt")

EEG94_OUT = os.path.join(OUT, "eeg94_out.txt")
EEG98_OUT = os.path.join(OUT, "eeg98_out.txt")
EEG98_USBC_OUT = os.path.join(OUT, "eeg98_usbc_out.txt")
EEG98_USBC_OUT_2 = os.path.join(OUT, "eeg98_usbc_out_2.txt")
EEG32_OUT = os.path.join(OUT, "eeg32_out.txt")

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

MATRIX_OUT = os.path.join(OUT, "axis_and_angle.txt")

EEG_IN_OUT_LIST = [
    (EEG94, EEG94_IN, EEG94_OUT),
    (EEG98, EEG98_IN, EEG98_OUT),
    (EEG98_USBC, EEG98_USBC_IN, EEG98_USBC_OUT),
    (EEG98_USBC, EEG98_USBC_IN_2, EEG98_USBC_OUT_2),
    (EEG32, EEG32_IN, EEG32_OUT)
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
    eeg = EEG(12345, b'\xaf\xbe\xad\xde')
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
    class_type = request.param[0]
    path_in = request.param[1]
    path_out = request.param[2]
    eeg_in = read_bin_to_byte_string(get_res_path(path_in))
    eeg_out = read_json_to_dict(get_res_path(path_out))
    ts = get_timestamp_from_byte_string(eeg_in)
    eeg_instance = class_type(ts, eeg_in[8:], 0)
    data = {'eeg_class': class_type,
            'eeg_instance': eeg_instance,
            'eeg_in': eeg_in,
            'eeg_out': eeg_out}
    return data


@pytest.fixture(params=[(ORN_IN, ORN_OUT)], scope="module")
def orientation_in_out(request):
    orn_in = read_bin_to_byte_string(get_res_path(request.param[0]))
    orn_out = read_json_to_dict(get_res_path(request.param[1]))
    orn_instance = Orientation(get_timestamp_from_byte_string(orn_in), orn_in[8:], 0)
    data = {
        'orn_in': orn_in,
        'orn_instance': orn_instance,
        'orn_out': orn_out
    }
    return data


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
    env_in = read_bin_to_byte_string(get_res_path(request.param[0]))
    env_instance = Environment(get_timestamp_from_byte_string(env_in), env_in[8:], 0)
    env_out = read_json_to_dict(get_res_path(request.param[1]))
    data = {
        'env_in': env_in,
        'env_instance': env_instance,
        'env_out': env_out
    }
    return data


@pytest.fixture(params=[(TS_IN, TS_OUT)])
def ts_in_out(request):
    try:
        ts_in = read_bin_to_byte_string(get_res_path(request.param[0]))
        ts_instance = TimeStamp(get_timestamp_from_byte_string(ts_in), ts_in[8:], 0)
        ts_out = read_json_to_dict(get_res_path(request.param[1]))
        data = {
            'ts_in': ts_in,
            'ts_instance': ts_instance,
            'ts_out': ts_out
        }
        return data
    except FileNotFoundError:
        pytest.skip("TimeStamp input or output file not available")


@pytest.fixture(params=[(PushButtonMarker, PUSH_MARKER_IN, PUSH_MARKER_OUT),
                        (SoftwareMarker, SOFTWARE_MARKER_IN, SOFTWARE_MARKER_OUT),
                        (ExternalMarker, EXTERNAL_MARKER_IN, EXTERNAL_MARKER_OUT)])
def marker_in_out(request):
    try:
        marker_in = read_bin_to_byte_string(get_res_path(request.param[1]))
        marker_instance = request.param[0](get_timestamp_from_byte_string(marker_in), marker_in[8:], 0)
        marker_out = read_json_to_dict(get_res_path(request.param[2]))
        data = {
            'marker_in': marker_in,
            'marker_instance': marker_instance,
            'marker_out': marker_out
        }
        return data
    except FileNotFoundError:
        pytest.skip(f"Input or output file not available for {request.param[0]}")


@pytest.fixture(params=[(TriggerIn, TRIGGER_IN_IN, TRIGGER_IN_OUT),
                        (TriggerOut, TRIGGER_OUT_IN, TRIGGER_OUT_OUT)])
def triggers_in_out(request):
    triggers_in = read_bin_to_byte_string(get_res_path(request.param[1]))
    triggers_instance = request.param[0](get_timestamp_from_byte_string(triggers_in), triggers_in[8:], 0)
    triggers_out = read_json_to_dict(get_res_path(request.param[2]))
    data = {
        'triggers_in': triggers_in,
        'triggers_instance': triggers_instance,
        'triggers_out': triggers_out
    }
    return data


@pytest.fixture(params=[(DISCONNECT_IN, DISCONNECT_OUT)])
def disconnect_in_out(request):
    disconnect_in = read_bin_to_byte_string(get_res_path(request.param[0]))
    disconnect_instance = Disconnect(get_timestamp_from_byte_string(disconnect_in), disconnect_in[8:], 0)
    disconnect_out = read_json_to_dict(get_res_path(request.param[1]))
    data = {
        'disconnect_in': disconnect_in,
        'disconnect_instance': disconnect_instance,
        'disconnect_out': disconnect_out
    }
    return data


@pytest.fixture(params=[(DEV_INFO_IN, DEV_INFO_OUT)])
def device_info_in_out(request):
    dev_info_in = read_bin_to_byte_string(get_res_path(request.param[0]))
    dev_info_instance = DeviceInfo(get_timestamp_from_byte_string(dev_info_in), dev_info_in[8:], 0)
    dev_info_out = read_json_to_dict(get_res_path(request.param[1]))
    data = {
        'dev_info_in': dev_info_in,
        'dev_info_instance': dev_info_instance,
        'dev_info_out': dev_info_out
    }
    return data


@pytest.fixture(params=[(DEV_INFO_V2_IN, DEV_INFO_V2_OUT)])
def device_info_v2_in_out(request):
    field_names = {'instance': 'dev_info_v2_instance',
                   'out': 'dev_info_v2_out'}
    return data_from_files(request.param[0], request.param[1], DeviceInfoV2, field_names)


#@pytest.fixture(params=[(CMD_RCV_IN, CMD_RCV_OUT)])
#def cmd_rcv_in_out(request):
#    return None
