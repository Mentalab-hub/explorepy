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

    def test_refactored_equal_nonrefactored_lowpass(self):
        e1 = ExGFilter(5, 'lowpass', 50, 4, 5)
        e2 = ExGFilter(5, 'lowpass', 50, 4, 5, use_new=True)
        np.testing.assert_equal(e1.filter_param, e2.filter_param)

    def test_refactored_equal_nonrefactored_notch(self):
        e1 = ExGFilter(5, 'notch', 50, 4, 5)
        e2 = ExGFilter(5, 'notch', 50, 4, 5, use_new=True)
        np.testing.assert_equal(e1.filter_param, e2.filter_param)

    def test_refactored_equal_nonrefactored_highpass(self):
        e1 = ExGFilter(5, 'highpass', 50, 4, 5)
        e2 = ExGFilter(5, 'highpass', 50, 4, 5, use_new=True)
        np.testing.assert_equal(e1.filter_param, e2.filter_param)

    def test_refactored_equal_nonrefactored_bandpass(self):
        e1 = ExGFilter((5, 10), 'bandpass', 50, 4, 5)
        e2 = ExGFilter((5, 10), 'bandpass', 50, 4, 5, use_new=True)
        np.testing.assert_equal(e1.filter_param, e2.filter_param)


class TestExGLowpassFilter(unittest.TestCase):
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
