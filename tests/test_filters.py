import unittest
from explorepy.filters import ExGFilter
import numpy as np
from scipy.signal import butter

class TestExGFilter(unittest.TestCase):
    def test_no_frequencies_passed(self):
        with self.assertRaises(Exception):
            ExGFilter(None, 'notch', 50, 4, 5)

    def test_wrong_frequency_shape_bandpass(self):
        with self.assertRaises(Exception):
            ExGFilter(5, 'bandpass', 50, 4, 5)

    def test_wrong_frequency_shape_lowpass(self):
        with self.assertRaises(Exception):
            ExGFilter([5, 20], 'lowpass', 50, 4, 5)

# TODO add tests for highpass, bandpass and notch filters

class TestExGLowpassFilter(unittest.TestCase):

    # Tests if in_place changing of ndarray works
    def test_apply_in_place_ndarray(self):
        f = ExGFilter(10, 'lowpass', 50, 1, 5)
        unfiltered_data = np.array([200, 200, 200, 200, 200])
        f.apply(unfiltered_data, in_place=True)
        np.testing.assert_raises(AssertionError,
                                 np.testing.assert_array_equal,
                                 unfiltered_data,
                                 np.array([200, 200, 200, 200, 200]))

    def test_negative_sampling_rate(self):
        with self.assertRaises(Exception):
            ExGFilter(10, 'lowpass', -20, 4, 5)

    def test_negative_order(self):
        with self.assertRaises(Exception):
            ExGFilter(10, 'lowpass', 50, 4, -5)

    def test_negative_cutoff(self):
        with self.assertRaises(Exception):
            ExGFilter(-10, 'lowpass', 50, 4, 5)

    def test_no_channels(self):
        with self.assertRaises(Exception):
            ExGFilter(10, 'lowpass', 50, 0, 5)

    # Tests if a cutoff that's too high gets clipped to nyquist-1/nyquist
    def test_lowpass_coefficients_cutoff_too_high(self):
        test_params = {'cutoff': 50, 'sampling_rate': 50, 'order': 2, 'n_channels': 4}
        nyquist = np.float(test_params['sampling_rate']) / 2.0
        cutoff = (nyquist - 1.0) / nyquist
        b, a = butter(N=test_params['order'], Wn=cutoff,
                      btype='lowpass')
        filter_coeffs = ExGFilter(test_params['cutoff'],
                  'lowpass',
                  test_params['sampling_rate'],
                  test_params['n_channels'],
                  test_params['order']).filter_param

    def test_lowpass_coefficients_normal_inputs(self):
        test_params = {'cutoff': 20, 'sampling_rate': 50, 'order': 2, 'n_channels': 4}
        b, a = butter(N=test_params['order'], Wn=test_params['cutoff'], fs=test_params['sampling_rate'],
                      btype='lowpass')
        filter_coeffs = ExGFilter(test_params['cutoff'],
                                      'lowpass',
                                      test_params['sampling_rate'],
                                      test_params['n_channels'],
                                      test_params['order']).filter_param
        self.assertEqual(filter_coeffs['a'].all(), a.all())
        self.assertEqual(filter_coeffs['b'].all(), b.all())

if __name__ == '__main__':
    unittest.main()
