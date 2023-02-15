import numpy as np
import pytest


def get_first_status_as_tuple(status_list):
    first_byte = hex(int(status_list[0][:2], 16))
    second_byte = hex(int(status_list[0][2:4], 16))
    third_byte = hex(int(status_list[0][4:6], 16))
    return first_byte, second_byte, third_byte


def test_status(parametrized_eeg_in_out):
    eeg = parametrized_eeg_in_out['eeg_instance']
    eeg_out = parametrized_eeg_in_out['eeg_out']
    expected_status = get_first_status_as_tuple(eeg_out['status'])
    assert eeg.status == expected_status


def test_convert_real(parametrized_eeg_in_out):
    eeg = parametrized_eeg_in_out['eeg_instance']
    eeg_out = parametrized_eeg_in_out['eeg_out']
    np.testing.assert_array_equal(eeg.data, eeg_out['samples'])


def test_check_fletcher(parametrized_eeg_in_out):
    eeg = parametrized_eeg_in_out['eeg_instance']
    eeg_out = parametrized_eeg_in_out['eeg_out']
    eeg._check_fletcher(bytes.fromhex(eeg_out['fletcher']))
