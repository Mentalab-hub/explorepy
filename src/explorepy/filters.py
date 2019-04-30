import numpy as np
from scipy.signal import butter, lfilter, iirnotch, iirfilter


class Filter:
    def __init__(self, l_freq, h_freq, line_freq=50, order=5):
        self.low_cutoff_freq = l_freq
        self.high_cutoff_freq = h_freq
        self.line_freq = line_freq
        self.sample_frequency = 250.0
        self.order = order
        self.bp_param = None
        self.notch_param = None

    def _design_filter(self, nchan):
        nyq = 0.5 * self.sample_frequency
        low_freq = self.low_cutoff_freq / nyq
        high_freq = self.high_cutoff_freq / nyq
        b, a = butter(self.order, [low_freq, high_freq], btype='band')
        zi = np.zeros((nchan, self.order*2))
        self.bp_param = {'a': a, 'b': b, 'zi': zi}

    def _design_notch_filter(self, nchan):
        # Q = 50
        # b, a = iirnotch(self.line_freq, Q, self.sample_frequency)
        nyq = 0.5 * self.sample_frequency
        low_freq = (self.line_freq-2) / nyq
        high_freq = (self.line_freq+2) / nyq
        b, a = iirfilter(5, [low_freq, high_freq], btype='bandstop', ftype='butter')
        zi = np.zeros((nchan, 10))
        self.notch_param = {'a': a, 'b': b, 'zi': zi}

    def apply_bp_filter(self, raw_data):
        if len(raw_data.shape) < 2:
            raw_data = np.array(raw_data)[np.newaxis, :]
        if self.bp_param is None:
            self._design_filter(nchan=raw_data.shape[0])

        filtered_data, zi = lfilter(self.bp_param['b'], self.bp_param['a'], raw_data, zi=self.bp_param['zi'])
        self.bp_param['zi'] = zi
        return filtered_data

    def apply_notch_filter(self, raw_data):
        if self.notch_param is None:
            self._design_notch_filter(nchan=raw_data.shape[0])

        filtered_data, zi = lfilter(self.notch_param['b'], self.notch_param['a'], raw_data, zi=self.notch_param['zi'])
        self.notch_param['zi'] = zi
        return filtered_data


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    n_chunk = 15
    n_sample = 33 * n_chunk
    t = np.linspace(0, n_chunk * 33./250., n_sample)
    omega = 2 * np.pi * 25
    x = 0.5 * np.sin(omega * t)
    x = np.repeat(x[np.newaxis,:], 4, axis=0)
    x_noisy = x + .2 * np.random.rand(4, n_sample) + np.cos(2 * np.pi * 50 * t) + .5

    filt = Filter(l_freq=20, h_freq=30, line_freq=50)
    x_filt = filt.apply_notch_filter(filt.apply_bp_filter(x_noisy))

    # test real-time filtering
    x_filt_realtime = np.zeros((4, 0))
    for i in range(n_chunk):
        x_filt_realtime = np.concatenate((x_filt_realtime,
                                          filt.apply_notch_filter(filt.apply_bp_filter(x_noisy[:, i * 33:(i + 1) * 33])))
                                         , axis=1)
        # Only notch filter
        # x_filt_realtime = np.concatenate((x_filt_realtime,
        #                                   filt.apply_notch_filter(x_noisy[:, i * 33:(i + 1) * 33]))
        #                                  , axis=1)

    plt.figure(1)
    plt.clf()
    plt.plot(t, x[1,:], label='original signal')
    plt.plot(t, x_noisy[1,:], label='noisy signal')
    plt.plot(t, x_filt[1,:], label='offline filtered')
    plt.plot(t, x_filt_realtime[1, :], label="real-time filtered")
    plt.legend()
    plt.show()

    print("finish")

