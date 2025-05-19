import argparse
import os.path
import threading
import time

import mne.io

import explorepy
import matplotlib.pyplot as plt
import numpy as np
import serial

from explorepy.tools import SettingsManager
from explorepy.packet import EEG, EEG32_BLE, EEG16_BLE, EEG98_BLE
from explorepy.stream_processor import TOPICS
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from mne.time_frequency import psd_array_multitaper
from numpy.typing import NDArray
from scipy.integrate import simpson
from scipy.signal import periodogram, welch


class EEGBuffer:
    def __init__(self, channels: int = 8, length: int = 2000):
        self.buffer = [np.zeros(length, dtype=np.float64) for _ in range(channels)]

    def update(self, packet: EEG):
        _, data = packet.get_data()
        for i in range(len(data)):
            self.buffer[i] = np.roll(self.buffer[i], -len(data[i]))  # Circular buffer
            self.buffer[i][-len(data[i]):] = data[i]

    def get_data(self) -> NDArray[np.float64]:
        return np.array(self.buffer)


class BandpowerCalculator:
    def __init__(self, fs: float = 250.0):
        self.fs = fs
        self.bands = {
            'Delta': (0.5, 4),
            'Theta': (4, 8),
            'Alpha': (8, 13),
            'Beta': (13, 30),
            'Gamma': (30, 50)
        }

    def compute(self, data: NDArray[np.float64], method: str = "welch", relative: bool = True) -> dict:
        bandpowers = {}
        for name, band in self.bands.items():
            bp = self._bandpower(data, self.fs, method, band, relative)  # array of channels
            bandpowers[name] = np.mean(bp)
        return bandpowers

    def _bandpower(self, data, fs, method, band, relative):
        if method == "periodogram":
            freqs, psd = periodogram(data, fs, window='hann')
        elif method == "welch":
            freqs, psd = welch(data, fs, window='hann', nperseg=500, noverlap=250)
        elif method == "multitaper":
            psd, freqs = psd_array_multitaper(data, fs, verbose="ERROR")
        else:
            raise ValueError(f"Unknown method: {method}")

        freq_res = freqs[1] - freqs[0]
        idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
        if psd.ndim == 1:
            bp = simpson(psd[idx_band], dx=freq_res)
        else:
            bp = simpson(psd[:, idx_band], dx=freq_res)

        if relative:
            idx_total = np.logical_and(freqs >= 0.5, freqs <= 50.0)
            if psd.ndim == 1:
                total = simpson(psd[idx_total], dx=freq_res)
            else:
                total = simpson(psd[:, idx_total], dx=freq_res)
            with np.errstate(divide='ignore', invalid='ignore'):
                bp = np.divide(bp, total, out=np.zeros_like(bp), where=total != 0)

        return bp


class BandpowerSmoother:
    """Class that holds a number of calculated bandpower values and returns the mean of them (thereby smoothing larger
    changes)."""
    def __init__(self, window_size=5):
        self.history = []
        self.window_size = window_size

    def smooth(self, new_values: dict):
        """Add new values to the internal history and return a smoothed value per dict key"""
        self.history.append(new_values)
        if len(self.history) > self.window_size:
            self.history.pop(0)
        return {
            key: np.mean([h[key] for h in self.history])
            for key in new_values
        }


class ThresholdTracker:
    """Class that uses the 10th and 90th percentile of a window of bandpower values to determine a moving threshold to
    detect when the bandpower value is rising."""
    def __init__(self, window_size: int = 120):
        self.values = []
        self.window_size = window_size
        self.min_spread = 0.1

    def add_value(self, value: float):
        self.values.append(value)
        if len(self.values) > self.window_size:
            self.values.pop(0)

    def get_threshold(self) -> float:
        """Calculates a value threshold that is approximately the middle of the value range represented in the current
        window of values"""
        if len(self.values) < 10:
            return 0.5
        q10 = np.percentile(self.values, 10)
        q90 = np.percentile(self.values, 90)
        spread = max(q90 - q10, self.min_spread)
        return q10 + 0.5 * spread


class EEGVisualizer:
    def __init__(self, calculator: BandpowerCalculator, tracker: ThresholdTracker, serial_port=None,
                 simulate_lamp=False, lamp_max=10.0):
        """Initializes a bandpower visualizer with the given parameters and creates a plot that is updated regularly.
        Args:
            calculator: A calculator that takes EEG data on updates and calculates bandpowers based on it
            (according to its own dict of frequency bands)
            tracker: A threshold tracker that holds a window of values and determines a suitable threshold on updates
            for when a band is rising or high in power
            serial_port: A serial port to write messages based on whether the power of a frequency band is above the
            current threshold or not
            simulate_lamp: Whether to add a second plot to the window that simulates a lamp increasing or decreasing in
            brightness
            lamp_max: The maximum value for the lamp above which values above the threshold stop influencing the
            brightness of the lamp (only for the simulated lamp plot)
        """
        self.calculator = calculator
        self.tracker = tracker
        self.smoother = BandpowerSmoother()
        self.serial_port = serial_port

        self.simulate_lamp = simulate_lamp
        self.lamp_value = 0.0
        self.lamp_max = lamp_max

        n_cols = 1
        if self.simulate_lamp:
            n_cols = 2
        self.fig, self.ax = plt.subplots(1, n_cols)

        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        ax = self.ax[0] if self.simulate_lamp else self.ax
        self.bars = ax.bar(calculator.bands.keys(), [0] * len(calculator.bands), color=self.colors)
        self.labels = [ax.text(bar.get_x() + bar.get_width() / 2, 0, '',
                                    ha='center', va='bottom') for bar in self.bars]

        ax.set_ylim(0, 1)
        ax.set_ylabel('Relative Bandpower')
        ax.set_title('Real-Time EEG Bandpower')
        if self.simulate_lamp:
            # Create a colourmap that attempts to look like the on and off states of a lamp (and matches the Arduino
            # code vaguely)
            cmap = LinearSegmentedColormap.from_list("lamp",
                                                     [(0.0, 0.2, 0.14), (1.0, 0.78, 0.0)],
                                                     N=int(self.lamp_max))

            self.lamp = self.ax[1].imshow([[0.0]], cmap=cmap, vmin=0.0, vmax=self.lamp_max)
            self.ax[1].set_title("Simulated lamp")

    def update(self, frame, eeg_buffer: EEGBuffer):
        data = eeg_buffer.get_data()

        raw_bandpowers = self.calculator.compute(data)
        bandpowers = self.smoother.smooth(raw_bandpowers)

        alpha_rel = bandpowers["Alpha"]
        self.tracker.add_value(alpha_rel)
        threshold = self.tracker.get_threshold()

        above = alpha_rel > threshold
        print(f"Current alpha (%): {alpha_rel:.2f}, "
              f"current threshold: {threshold:.2f}, "
              f"above threshold: {"yes" if above else "no"}")

        if self.serial_port and self.serial_port.is_open:
            try:
                # write 1 to the port if the threshold is crossed, 0 otherwise
                self.serial_port.write(b'1\n' if above else b'0\n')
            except Exception as e:
                print(f"Failed to write to port: {e}")

        if self.simulate_lamp:
            self.lamp_value = self.lamp_value + 1.0 if above else self.lamp_value - 1.0
            self.lamp_value = max(min(self.lamp_max, self.lamp_value), 0.0)  # clamp, not actually necessary
            self.lamp.set_data([[self.lamp_value]])

        for bar, label, (name, value) in zip(self.bars, self.labels, bandpowers.items()):
            bar.set_height(value)
            label.set_y(value + 0.01)
            label.set_text(f'{value:.2f}')

        return list(self.bars.patches) + self.labels


class DataServer:
    def __init__(self, data_file, callback=None):
        out = mne.io.read_raw_bdf(data_file)

        self.sr = out.info["sfreq"]
        self.update_rate = 1. / self.sr
        self.data = out.get_data()
        self.n_ch = self.data.shape[0] - 1

        self.dummy_packet = None
        dummy_data = bytearray(b'\x80\x00\x00' * self.n_ch + b'\xaf\xbe\xad\xde')
        if self.n_ch == 8:
            self.dummy_packet = EEG98_BLE(0.0, dummy_data)
        elif self.n_ch == 16:
            self.dummy_packet = EEG16_BLE(0.0, dummy_data)
        elif self.n_ch == 32:
            self.dummy_packet = EEG32_BLE(0.0, dummy_data)
        else:
            raise ValueError("Got unexpected channel count")

        self.callback = None
        self.acquire_thread = None
        self.set_callback(callback)

    def set_callback(self, callback):
        if self.callback or self.acquire_thread:
            raise NotImplementedError("Changing the callback is not supported.")
        if not callback:
            return
        self.callback = callback
        self.acquire_thread = threading.Thread(target=self.acquire, daemon=True)
        self.acquire_thread.start()

    def acquire(self):
        it = 0
        while True:
            packet = self.dummy_packet
            packet.data = self.data[1:, it].reshape(self.n_ch, 1)
            packet.timestamp = self.data[0, it]
            it+=1
            self.callback(packet)
            time.sleep(self.update_rate)

    def get_channel_count(self):
        return self.n_ch

    def get_sampling_rate(self):
        return self.sr


class EEGApp:
    def __init__(self, device_name, sr=None, port=None, notch=None, bandpass=None, simulate_lamp=False, lamp_max=10.0,
                 file=None):
        self.serial_port = None
        if port:
            print("Serial port supplied, attempting to open for communication...")
            try:
                self.serial_port = serial.Serial(port, 9600)
                print("Serial port opened for communication with Arduino")
            except Exception as e:
                print(f"Couldn't open serial port: {e}")

        if file and os.path.exists(file) and file.split(".")[-1] == "bdf":
            print("Simulating from file")
            self.data_server = DataServer(file)
            self.buffer = EEGBuffer(self.data_server.get_channel_count())
            self.data_server.set_callback(self.buffer.update)

            self.calculator = BandpowerCalculator(fs=250)
            self.tracker = ThresholdTracker()

            self.visualizer = EEGVisualizer(self.calculator, self.tracker, self.serial_port,
                                            simulate_lamp=simulate_lamp,
                                            lamp_max=lamp_max)
            return

        self.device = explorepy.Explore()
        self.device.connect(device_name=device_name)
        if sr: self.device.set_sampling_rate(sr)

        self.settings = SettingsManager(device_name)
        self.buffer = EEGBuffer(self.settings.get_channel_count())
        self.calculator = BandpowerCalculator(fs=self.settings.get_sampling_rate())
        self.tracker = ThresholdTracker()

        self.visualizer = EEGVisualizer(self.calculator, self.tracker, self.serial_port, simulate_lamp=simulate_lamp,
                                        lamp_max=lamp_max)

        if notch: self.device.stream_processor.add_filter(cutoff_freq=notch, filter_type="notch")
        if bandpass: self.device.stream_processor.add_filter(cutoff_freq=(bandpass[0], bandpass[1]),
                                                             filter_type="bandpass")

        self.device.stream_processor.subscribe(callback=self.buffer.update, topic=TOPICS.filtered_ExG)

    def run(self):
        plt_animation = FuncAnimation(self.visualizer.fig, self.visualizer.update, fargs=(self.buffer,), interval=500,
                            cache_frame_data=False)
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="EEG2Lamp",
                                     description="This script connects to an Arduino running the accompanying example "
                                                 ".ino to control the brightness and colour of an LED strip based on "
                                                 "the increasing and decreasing power in the alpha band (8, 13)")
    parser.add_argument("-n", "--name", required=True, type=str, nargs=1,
                        help="The name of the Explore device (i.e. Explore_ABCD")
    parser.add_argument("-sr", "--sampling_rate", default=None, type=int, nargs=1,
                        help="The sampling rate to set the device to (disabled by default)")
    parser.add_argument("-p", "--port", default=None, type=str, nargs=1,
                        help="The port used to connect and communicate with the Arduino controlling the LEDs")
    parser.add_argument("-fn", "--notch", default=None, type=float, nargs=1,
                        help="A value for the notch filter (disabled by default)")
    parser.add_argument("-fbp", "--bandpass", default=None, type=float, nargs=2,
                        help="Values for the bandpass filter (disabled by default)")
    parser.add_argument("--simulate_lamp", action="store_true",
                        help="Whether to add a second plot to the bandpower plot that simulates the lamp's colour "
                             "change")
    parser.add_argument("-f", "--from_file", default=None, type=str, nargs=1,
                        help="Supplying this argument causes the script to ignore the arguments related to device "
                             "streaming and simulate the changing of the bandpowers and lamp brightness from the "
                             "supplied .bdf file")

    args = parser.parse_args()

    sr = args.sampling_rate[0] if args.sampling_rate else None
    port = args.port[0] if args.port else None
    notch = args.notch[0] if args.notch else None
    bp = (args.bandpass[0], args.bandpass[1]) if args.bandpass else None
    file = args.from_file[0] if args.from_file else None

    app = EEGApp(device_name=args.name[0], sr=sr, port=port, notch=notch,
                 bandpass=bp, simulate_lamp=args.simulate_lamp,  lamp_max=20, file=file)

    app.run()
