import os
import struct
import json
import binascii

import pytest

IN = "in"
OUT = "out"
EEG8_CHANNEL_PACKET_BIN = os.path.join(IN, "eeg8_channel_packet")
EEG8_EXPECTED_REAL_OUTPUT = os.path.join(OUT, "expected_eeg8_output.txt")
EEG8_EXPECTED_FAKE_OUTPUT = os.path.join(OUT, "expected_fake_eeg8_output.txt")


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
    return binascii.unhexlify(input_string)


def read_json_to_dict(source):
    with open(get_res_path(source), "r") as expected_output_file:
        return json.load(expected_output_file)


def string_list_to_hex_tuple(string_list):
    first_string = hex(int(string_list[0], 16))
    second_string = hex(int(string_list[1], 16))
    third_string = hex(int(string_list[2], 16))
    return first_string, second_string, third_string


@pytest.fixture(scope="module")
def eeg8_test_whole_packet():
    return read_bin_to_byte_string(get_res_path(EEG8_CHANNEL_PACKET_BIN))


@pytest.fixture(scope="module")
def eeg8_test_samples(eeg8_test_whole_packet):
    return eeg8_test_whole_packet[8:-4]


@pytest.fixture(scope="module")
def eeg8_test_fletcher(eeg8_test_whole_packet):
    return eeg8_test_whole_packet[-4:]


@pytest.fixture(scope="module")
def eeg8_test_timestamp(eeg8_test_whole_packet):
    timestamp_bytes = eeg8_test_whole_packet[4:8]
    timestamp_floating_point = struct.unpack('<I', timestamp_bytes)[0]
    return timestamp_floating_point


@pytest.fixture(scope="module")
def eeg8_test_status(eeg8_test_whole_packet):
    status = read_bin_to_byte_string(get_res_path(EEG8_CHANNEL_PACKET_BIN))[8:11]
    status = (hex(status[0]), hex(status[1]), hex(status[2]))
    return status


@pytest.fixture(scope="module")
def eeg8_expected_results_whole():
    return read_json_to_dict(EEG8_EXPECTED_REAL_OUTPUT)


@pytest.fixture(scope="module")
def eeg8_expected_samples(eeg8_expected_results_whole):
    return eeg8_expected_results_whole["samples"]


@pytest.fixture(scope="module")
def eeg8_expected_fletcher(eeg8_expected_results_whole):
    fletcher_string = eeg8_expected_results_whole["fletcher"]
    return string_to_byte_string(fletcher_string)


@pytest.fixture(scope="module")
def eeg8_expected_timestamp(eeg8_expected_results_whole):
    return eeg8_expected_results_whole['raw_timestamp']


@pytest.fixture(scope="module")
def eeg8_expected_status(eeg8_expected_results_whole):
    return string_list_to_hex_tuple(eeg8_expected_results_whole['status'])


@pytest.fixture(scope="module")
def eeg8_expected_results_fake_whole():
    with open(get_res_path(EEG8_EXPECTED_FAKE_OUTPUT), "r") as expected_output_file:
        return json.load(expected_output_file)


@pytest.fixture(scope="module")
def eeg8_expected_samples_fake(eeg8_expected_results_fake_whole):
    return eeg8_expected_results_fake_whole["samples"]


@pytest.fixture(scope="module")
def eeg8_expected_fletcher_fake(eeg8_expected_results_fake_whole):
    fletcher_string = eeg8_expected_results_fake_whole["fletcher"]
    return string_to_byte_string(fletcher_string)


@pytest.fixture(scope="module")
def eeg8_expected_timestamp_fake(eeg8_expected_results_fake_whole):
    return eeg8_expected_results_fake_whole['raw_timestamp']


@pytest.fixture(scope="module")
def eeg8_expected_status_fake(eeg8_expected_results_fake_whole):
    status_list = eeg8_expected_results_fake_whole['status']
    first_string = hex(int(status_list[0], 16))
    second_string = hex(int(status_list[1], 16))
    third_string = hex(int(status_list[2], 16))
    as_tuple = first_string, second_string, third_string
    return as_tuple


@pytest.fixture(scope="module")
def eeg98_usbc_test_samples():
    pass
