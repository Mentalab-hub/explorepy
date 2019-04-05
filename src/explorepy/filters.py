import numpy as np
from scipy.signal import butter, lfilter


class Filter:
    def __init__(self, l_freq, h_freq):
        self.low_cutoff_freq = l_freq
        self.high_cutoff_freq = h_freq
        self.sample_frequency = 250.0
        self.order = 5
        self.param = None

    def _design_filter(self, nchan):
        nyq = 0.5 * self.sample_frequency
        low_freq = self.low_cutoff_freq / nyq
        high_freq = self.high_cutoff_freq / nyq
        b, a = butter(self.order, [low_freq, high_freq], btype='band')
        zi = np.zeros((nchan, 10))  # lfiltic(b, a, (0.,))
        self.param = {'a': a, 'b': b, 'zi': zi}

    def apply_bp_filter(self, raw_data):
        if self.param is None:
            self._design_filter(nchan=raw_data.shape[0])

        filtered_data, zi = lfilter(self.param['b'], self.param['a'], raw_data, zi=self.param['zi'])
        self.param['zi'] = zi
        return filtered_data


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    n_chunk = 15
    n_sample = 33 * n_chunk
    t = np.linspace(0, n_chunk * 33./250., n_sample)
    omega = 2 * np.pi * 25
    x = 0.5 * np.sin(omega * t)
    x = np.repeat(x[np.newaxis,:], 4, axis=0)
    x_noisy = x + .4 * np.random.rand(4, n_sample) + .4  # np.cos(500 * np.pi * t) + .5

    filt = Filter(l_freq=20, h_freq=30)
    x_filt = filt.apply_bp_filter(x_noisy)

    # test real-time filtering
    x_filt_realtime = np.zeros((4, 0))
    for i in range(n_chunk):
        x_filt_realtime = np.concatenate((x_filt_realtime, filt.apply_bp_filter(x_noisy[:, i * 33:(i + 1) * 33])), axis=1)

    plt.figure(1)
    plt.clf()
    plt.plot(t, x[1,:], label='original signal')
    plt.plot(t, x_noisy[1,:], label='noisy signal')
    plt.plot(t, x_filt[1,:], label='offline filtered')
    plt.plot(t, x_filt_realtime[1, :], label="real-time filtered")
    plt.legend()
    plt.show()

    print("finish")

