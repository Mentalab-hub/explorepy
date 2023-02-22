import os
import struct
import json

import numpy as np

import pytest

from explorepy.packet import (
    EEG,
    EEG94,
    EEG98,
    EEG98_USBC,
    EEG32,
    EEG99,
    EEG99s,
    EventMarker,
    Orientation,
    Packet
)

IN = "in"
OUT = "out"

EEG94_IN = os.path.join(IN, "eeg94")
EEG98_IN = os.path.join(IN, "eeg98")
EEG98_USBC_IN = os.path.join(IN, "eeg98_usbc")
EEG98_USBC_IN_2 = os.path.join(IN, "eeg98_usbc_2")
EEG32_IN = os.path.join(IN, "eeg32")

EEG94_OUT = os.path.join(OUT, "eeg94_out.txt")
EEG98_OUT = os.path.join(OUT, "eeg98_out.txt")
EEG98_USBC_OUT = os.path.join(OUT, "eeg98_usbc_out.txt")
EEG98_USBC_OUT_2 = os.path.join(OUT, "eeg98_usbc_out_2.txt")
EEG32_OUT = os.path.join(OUT, "eeg32_out.txt")

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


@pytest.fixture(params=[Packet, EEG], scope="module")
def parametrized_abstract_packets(request):
    return request.param


@pytest.fixture(scope="function")
def mocked_eeg_base(mocker):
    if hasattr(EEG, "__abstractmethods__"):
        if len(EEG.__abstractmethods__) != 0:
            mocker.patch.multiple(EEG, __abstractmethods__=set())
    eeg = EEG(12345, b'\xaf\xbe\xad\xde')
    eeg.data = np.array(
             [[40, 3333, 78910, -30, 0], [20, -1000, 10, 30, 0], [10, 2345, 77016, 11, 45], [15, 1234, 70000, 2, 44]])
    return eeg


@pytest.fixture(params=[
    ([10, 20, 30, 40, 50, 60], [1971210, 3945000]),
    ([255, 255, 255, 255, 255, 255], [-1, -1]),         # [111111111111111111111111, 111111111111111111111111]
    ([0, 0, 0, 0, 0, 0], [0, 0]),
    ([0, 0, 128, 0, 0, 128], [-8388608, -8388608])],    # [100000000000000000000000, 100000000000000000000000]
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
