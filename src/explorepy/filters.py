# -*- coding: utf-8 -*-
"""Filter module"""

import numpy as np
from scipy.signal import butter, lfilter, iirfilter


class ExGFilter:
    """Filter class for ExG signals"""
    def __init__(self, cutoff_freq, filter_type, s_rate, n_chan, order=5):
        """
        Args:
            cutoff_freq (Union[float, tuple]): cutoff frequency (frequencies) for the filter
            filter_type (str): Fitler type ['lowpass', 'highpass', 'bandpass', 'notch']
            s_rate (Union[float, int]): sampling rate of the signal
            order (int): Filter order (default value: 5)
            n_chan (int): Number of channels
        """
        self.s_rate = np.float(s_rate)
        nyq_freq = self.s_rate / 2.
        self.filter_type = filter_type
        self.filter_param = None
        if filter_type is 'lowpass':
            hc_freq = cutoff_freq / nyq_freq
            b, a, _ = butter(order, hc_freq, btype='lowpass')
            zi = np.zeros((n_chan, order))

        elif filter_type is 'highpass':
            lc_freq = cutoff_freq / nyq_freq
            b, a, _ = butter(order, lc_freq, btype='highpass')
            zi = np.zeros((n_chan, order))

        elif filter_type is 'bandpass':
            if cutoff_freq[0] >= cutoff_freq[1]:
                raise ValueError("High cutoff frequency must be larger than low cutoff frequency.")
            lc_freq = cutoff_freq[0] / nyq_freq
            hc_freq = cutoff_freq[1] / nyq_freq
            if lc_freq <= 0.003:
                raise ValueError('Transient band for Low Frequency of bandpass filter is too narrow')
            b, a, _ = butter(order, [lc_freq, hc_freq], btype='band')
            zi = np.zeros((n_chan, order * 2))

        elif filter_type is 'notch':
            lc_freq = (cutoff_freq - 2) / nyq_freq
            hc_freq = (cutoff_freq + 2) / nyq_freq
            b, a, _ = iirfilter(5, [lc_freq, hc_freq], btype='bandstop', ftype='butter')
            zi = np.zeros((n_chan, 10))
        else:
            raise ValueError('Unknown filter type: {}'.format(filter_type))
        self.filter_param = {'a': a, 'b': b, 'zi': zi}

    def apply(self, packet):
        """Apply filter in-place

        Args:
            packet (explorepy.packet.EEG): ExG packet to be filtered

        Returns:
            packet: filtered packet
        """
        _, raw_data = packet.get_data(self.s_rate)
        if len(raw_data.shape) < 2:
            raw_data = np.array(raw_data)[np.newaxis, :]
        filtered_data, self.filter_param['zi'] = lfilter(self.filter_param['b'],
                                                         self.filter_param['a'],
                                                         raw_data,
                                                         zi=self.filter_param['zi'])
        packet.data = filtered_data
        return packet
