import numpy as np
import pytest

from explorepy.filters import ExGFilter


@pytest.fixture
def exg_filter():
    return ExGFilter(cutoff_freq=30, filter_type='lowpass', s_rate=250, n_chan=8)


def test_get_lowpass_coeffs():
    a, b, zi = ExGFilter.get_lowpass_coeffs(30, 125, 8, 5)
    assert len(a) == 6
    assert len(b) == 6
    assert zi.shape == (8, 5)


def test_get_highpass_coeffs():
    a, b, zi = ExGFilter.get_highpass_coeffs(1, 125, 8, 5)
    assert len(a) == 6
    assert len(b) == 6
    assert zi.shape == (8, 5)


def test_get_bandpass_coeffs():
    a, b, zi = ExGFilter.get_bandpass_coeffs(1, 40, 125, 8, 5)
    assert len(a) == 11
    assert len(b) == 11
    assert zi.shape == (8, 10)


def test_get_notch_coeffs():
    a, b, zi = ExGFilter.get_notch_coeffs(50, 125, 8, 5)
    assert len(a) == 3
    assert len(b) == 3
    assert zi.shape == (8, 2)


def test_get_notch_coeffs_imp_mode():
    a, b, zi = ExGFilter.get_notch_coeffs(50, 125, 8, 5, imp_mode=True)
    assert len(a) == 11
    assert len(b) == 11
    assert zi.shape == (8, 10)


def test_apply(exg_filter):
    raw_data = np.random.randn(8, 1000)
    filtered_data = exg_filter.apply(raw_data, in_place=False)
    assert filtered_data.shape == raw_data.shape


def test_apply_in_place(exg_filter):
    raw_data = np.random.randn(8, 1000)
    filtered_data = exg_filter.apply(raw_data, in_place=True)
    assert filtered_data.shape == raw_data.shape


def test_apply_packet(exg_filter):
    class MockPacket:
        def get_data(self, s_rate):
            return None, np.random.randn(8, 1000)
    packet = MockPacket()
    filtered_packet = exg_filter.apply(packet, in_place=False)
    assert filtered_packet.data.shape == (8, 1000)


def test_apply_to_raw_data(exg_filter):
    raw_data = np.random.randn(8, 1000)
    filtered_data = exg_filter._apply_to_raw_data(raw_data)
    assert filtered_data.shape == raw_data.shape
