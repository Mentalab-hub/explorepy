import unittest

import numpy as np
from scipy.signal import butter

from explorepy.filters import ExGFilter


class TestExGFilter(unittest.TestCase):
    def test_no_frequencies_passed(self):
        with self.assertRaises(Exception):
            ExGFilter.get_filter_coeffs(None, 'notch', 50, 4, 5)

    def test_wrong_frequency_shape_bandpass(self):
        with self.assertRaises(Exception):
            ExGFilter.get_filter_coeffs(5, 'bandpass', 50, 4, 5)

    def test_wrong_frequency_shape_lowpass(self):
        with self.assertRaises(Exception):
            ExGFilter.get_filter_coeffs([5, 20], 'lowpass', 50, 4, 5)


# TODO add tests for highpass, bandpass and notch filters
class TestExGApply(unittest.TestCase):

    # Set up a common filter for every test
    @classmethod
    def setUpClass(cls):
        cls.filter = ExGFilter(10, 'lowpass', 50, 1, 5)

    # Tests if in_place changing of ndarray works
    def test_apply_in_place_ndarray(self):
        unfiltered_data = np.array([200, 200, 200, 200, 200])
        self.filter.apply(unfiltered_data, in_place=True)
        np.testing.assert_raises(AssertionError,
                                 np.testing.assert_array_equal,
                                 unfiltered_data,
                                 np.array([200, 200, 200, 200, 200]))

    def test_apply_input_none(self):
        with self.assertRaises(Exception):
            self.filter.apply(None)

    # If no values are entered I expect the array to remain unchanged
    def test_apply_input_empty_unchanged(self):
        res = self.filter.apply(np.empty([0]))
        np.testing.assert_array_equal(res, np.empty([0]))

    def test_apply_packet(self):
        pass

    def test_apply_raw(self):
        pass


class TestExGLowpassFilter(unittest.TestCase):
    # TODO figure out how to test whether the filter result makes sense...?

    def test_negative_nyquist(self):
        with self.assertRaises(Exception):
            ExGFilter.get_lowpass_coeffs(10, -20, 4, 5)

    def test_negative_order(self):
        with self.assertRaises(Exception):
            ExGFilter.get_lowpass_coeffs(10, 25, 4, -5)

    def test_negative_cutoff(self):
        with self.assertRaises(Exception):
            ExGFilter.get_lowpass_coeffs(-10, 25, 4, 5)

    def test_no_channels(self):
        with self.assertRaises(Exception):
            ExGFilter.get_lowpass_coeffs(10, 25, 0, 5)

    # Tests if a cutoff that's too high gets clipped to nyquist-1/nyquist
    def test_lowpass_coefficients_cutoff_too_high(self):
        test_params = {'cutoff': 50, 'sampling_rate': 50, 'order': 2, 'n_channels': 4}
        nyquist = np.float(test_params['sampling_rate']) / 2.0
        cutoff = (nyquist - 1.0) / nyquist
        b, a = butter(N=test_params['order'], Wn=cutoff,
                      btype='lowpass')
        filter_coeffs = ExGFilter.get_lowpass_coeffs(test_params['cutoff'],
                                                     nyquist,
                                                     test_params['n_channels'],
                                                     test_params['order'])
        np.testing.assert_array_equal(filter_coeffs[0], a)
        np.testing.assert_array_equal(filter_coeffs[1], b)

    def test_lowpass_coefficients_normal_inputs(self):
        test_params = {'cutoff': 20, 'sampling_rate': 50, 'order': 2, 'n_channels': 4}
        b, a = butter(N=test_params['order'], Wn=test_params['cutoff'], fs=test_params['sampling_rate'],
                      btype='lowpass')
        filter_coeffs = ExGFilter.get_lowpass_coeffs(test_params['cutoff'],
                                                     test_params['sampling_rate'] / 2.0,
                                                     test_params['n_channels'],
                                                     test_params['order'])
        np.testing.assert_array_equal(filter_coeffs[0], a)
        np.testing.assert_array_equal(filter_coeffs[1], b)


if __name__ == '__main__':
    unittest.main()
