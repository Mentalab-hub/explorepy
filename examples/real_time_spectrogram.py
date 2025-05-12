import vispy.plot

import argparse

import explorepy
from explorepy import Explore
from explorepy.packet import EEG
from explorepy.stream_processor import TOPICS
from vispy import app
from scipy.signal import stft
from scipy.fft import fft
import numpy as np
from vispy.visuals.transforms import STTransform

valid_colormaps = ["autumn", "blues", "cool", "greens", "reds", "spring", "summer", "fire", "grays", "hot", "ice",
                   "winter", "light_blues", "orange", "viridis", "coolwarm", "PuGr", "GrBu", "GrBu_d", "RdBu",
                   "cubehelix", "single_hue", "hsl", "husl", "diverging", "RdYeBuCy"]  # taken from vispy


class SpectrogramPlot:
    """Class containing a vispy plot and logic to get ExG data from the Explore device, calculate the STFT on it and
    display it in real time."""
    def __init__(self, device_sr=250, update_window=0.5, time_window=10, mode="overlapping", colormap="viridis",
                 config=None, f_cutoff=70):
        """
        Args:
            device_sr (int): The sampling rate of the device
            update_window (float): Time in seconds between updates of the canvas, for the mode "overlapping", this determines
            (together with the sampling rate) how many values are dropped before recalculating the STFT. For the mode
            "moving_stft, this determines how many values are considered for a calculation of the STFT
            time_window (int): The time window to display in seconds. For the mode "overlapping", this determines how many
            values are considered for a calculation of the STFT, for the mode "moving", this determines the total time
            window of the plot
            mode (str): The drawing mode of the plot, "overlapping" for a plot that updates completely every frame,
            "moving_stft" or "moving_fft" for a plot that adds a new segment to the plot for every frame using either
            the STFT (calculating chunks) or FFT (calculated for a single timestamp). Note that "moving" doesn't overlap
            values when recalculating the STFT and thus needs an increased update rate for better resolution!
            colormap (str): The initial colormap to use for the plot, needs to be one of
            vispy.color.colormap.get_colormaps() (see "valid_colormaps")
            config (dict): A config containing parameter values for the stft (when using moving_stft or overlapping)
        """
        self.fig = vispy.plot.Fig(size=(800, 600), show=False)
        self.fig.events.connect(self.on_event)
        self.plot = self.fig[0, 0]
        self.plot.bgcolor = "#FFFFFF"
        self.img = np.full(shape=(512, 512), fill_value=0.0, dtype=np.float32)
        self.plot_image = None
        self.plot_colorbar = None
        self.colormap = colormap

        self.colormap_index = 0

        self.config = config

        self.clim_max = 128
        self.f_cutoff = f_cutoff

        self.valid_modes = ["moving_fft", "moving_stft", "overlapping"]
        self.mode = mode

        if self.mode not in self.valid_modes:
            raise ValueError(f"Encountered unexpected drawing mode: {self.mode}, valid modes: {self.valid_modes}")

        self.drop_num = -1
        self.update_window = update_window
        if self.mode == "moving_stft" or self.mode == "moving_fft":
            self.num_vals = 2 ** np.ceil(np.log2(device_sr * update_window))
        elif self.mode == "overlapping":
            self.num_vals = 2 ** np.ceil(np.log2(device_sr * time_window))
            self.drop_num = int(device_sr * update_window)

        self.device_sr = device_sr
        self.time_window = time_window  # in s, this is a suggested time window, not fixed

        self.current_index = 0
        self.data_buf = []

        self.current_size = None
        self.timer = app.Timer(start=True, interval=1. / 30., connect=self.create_window)

    def on_event(self, event):
        """Catches events and processes them, used to catch presses of the up and down key to adjust the maximum value
        of the plotted image. Additionally, pressing the c key will cycle between available colormaps"""
        if type(event) is not vispy.app.canvas.KeyEvent:
            return
        if event.type == "key_press":
            if event.key == "Up":
                self.clim_max += 1.0
            elif event.key == "Down":
                self.clim_max = max(1.0, self.clim_max - 1.0)
        if event.type == "key_release" and event.key == "C":
            new_cmap = valid_colormaps[self.colormap_index]
            print(f"Switching to colourmap: {new_cmap}")
            self.plot_image.cmap = new_cmap
            self.plot_colorbar.cmap = new_cmap
            self.colormap_index += 1
            self.colormap_index %= len(valid_colormaps)

    def create_window(self, event):
        """Used to instantiate the window initially, has to be called from the main thread"""
        if self.current_size is not None:
            self.plot_image = self.plot.image(self.img, cmap=self.colormap, clim=(0.0, 4.0),)

            scale_factor = self.time_window / self.img.shape[1]
            self.plot_image.transform = STTransform(scale=(scale_factor, 1, 1))

            self.plot.camera.set_range(margin=0)
            self.plot.camera.aspect = self.img.shape[0] / self.time_window

            self.plot.xaxis.axis.axis_label = "Time (s)"
            self.plot.yaxis.axis.axis_label = "Frequency (Hz)"
            self.plot.yaxis.axis.axis_label_margin = 50

            self.plot_colorbar = self.plot.colorbar(position="right", cmap=self.colormap, clim=("0", "4.0"))
            self.fig.show()
            self.timer.stop()

    def on_exg(self, packet: EEG):
        """Called by explorepy from a thread that isn't the main thread, triggers plot updates"""
        ret = packet.get_data()[1]
        self.data_buf.append(ret)
        if len(self.data_buf) >= self.num_vals:
            Zxx_channels = []
            if self.mode == "moving_stft" or self.mode == "overlapping":
                for i in range(len(self.data_buf[0])):
                    _, _, Zxx = stft(np.asarray(self.data_buf)[:, i, 0], fs=self.device_sr) if self.config is None else (
                        stft(np.asarray(self.data_buf)[:, i, 0],
                            fs=self.device_sr,
                            window=self.config["window"],
                            nperseg=self.config["nperseg"],
                         noverlap=self.config["noverlap"]))
                    Zxx_channels.append(Zxx)
                Zxx = np.mean(Zxx_channels, axis=0)
            elif self.mode == "moving_fft":
                Zxx = np.abs(fft(np.asarray(self.data_buf)[:, :, 0].T))
                Zxx = np.mean(Zxx, axis=0)
                Zxx = Zxx[1:Zxx.shape[0]//2]
                Zxx = np.reshape(Zxx, shape=(Zxx.shape[0], 1))

            y_max = Zxx.shape[0]
            y_max = min(y_max, self.f_cutoff)
            x_max = Zxx.shape[1]

            if self.current_size is None:
                self.current_size = self.get_initial_size(x_max, y_max)

                # Set the numpy array to the same size as the calculated window size
                self.img = np.full(shape=(self.current_size[1], self.current_size[0]), fill_value=0.0, dtype=np.float32)

            # Overwrite the next section in the numpy array with the new STFT values
            self.img[0:y_max, self.current_index:self.current_index+x_max] = np.abs(Zxx[:y_max, :])

            if self.mode == "moving_stft" or self.mode == "moving_fft":
                self.current_index += x_max

                # Draw a rudimentary "swipe line" (the colour is single-valued, so we're limited in choices here)
                # Note that self.current_index + 1 can be outside the array boundary!
                self.img[:, self.current_index:self.current_index+1] = 2**16

                if self.current_size is not None:
                    self.current_index %=self.current_size[0]

            if self.plot_image is not None:
                self.plot_image.set_data(self.img)
                self.plot_image.clim = (0.0, self.clim_max)
                if self.plot_colorbar is not None:
                    self.plot_colorbar.clim = ("0.0", f"{self.clim_max}")
                self.fig.update()

            if self.mode == "moving_stft" or self.mode == "moving_fft":
                self.data_buf = []
            elif self.mode == "overlapping":
                self.data_buf = self.data_buf[self.drop_num:]

    def get_initial_size(self, x_max, y_max):
        """Calculates the size of the numpy array according to time window, update rate and shape of the STFT / FFT
        result"""
        if self.mode == "moving_stft" or self.mode == "moving_fft":
            return int(x_max * (self.time_window // self.update_window)), y_max
        elif self.mode == "overlapping":
            return x_max, y_max
        else:
            raise ValueError(f"Encountered unexpected drawing mode while trying to get initial size: {self.mode}")


from explorepy.tools import SettingsManager

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="Real-time spectrogram example",
                                     description="This program connects to an Explore device, optionally sets its "
                                                 "sampling rate and adds filters to the stream and then displays the "
                                                 "spectrogram of the signals in real-time.")
    parser.add_argument("-n", "--name", nargs=1, required=True, type=str,
                        help="The device name, i.e. Explore_ABCD (required)")
    parser.add_argument("-sr", "--sampling_rate", nargs=1, type=int,
                        help="The sampling rate to set the device to after connecting")
    parser.add_argument("-uw", "--update_window", nargs=1, type=float,
                        help="The time window between updates of the plot")
    parser.add_argument("-tw", "--time_window", nargs=1, type=float,
                        help="The total time window displayed by the plot")
    parser.add_argument("-usb", action="store_true",
                        help="Whether to connect via USB")
    parser.add_argument("-fn", "--notch", nargs=1, default=None, type=float,
                        help="The frequency for the notch filter")
    parser.add_argument("-fbp", "--bandpass", nargs=2, default=None, type=float,
                        help="The frequencies for the bandpass filter")
    parser.add_argument("-m", "--drawing_mode", nargs=1,
                        choices=["moving_fft", "overlapping", "moving_stft"], default="moving_fft", type=str,
                        help="The mode to use for drawing the plot (default: %(default)s")
    args = parser.parse_args()

    uw = args.update_window[0]
    if uw is None:
        if args.drawing_mode[0] == "overlapping":
            uw = 1./10.
        else:
            uw = 1./2.

    tw = args.time_window[0]
    if tw is None:
        tw = 10

    vispy.app.use_app("glfw")

    config = {"window": "hann", "nperseg": 128, "noverlap": 0}
    # config for the stft, used for modes overlapping and moving_stft, set to None to use default values

    if args.usb: explorepy.set_bt_interface("usb")

    explore_device = Explore()
    explore_device.connect(device_name=args.name[0])
    if args.sampling_rate: explore_device.set_sampling_rate(args.sampling_rate[0])

    dev_sr = SettingsManager(args.name[0]).get_sampling_rate()

    rt_spectrogram = SpectrogramPlot(device_sr=dev_sr,
                                     update_window=uw,
                                     time_window=tw,
                                     mode=args.drawing_mode[0],
                                     colormap="viridis",
                                     config=config,
                                     f_cutoff=70)

    if args.notch: explore_device.stream_processor.add_filter(cutoff_freq=args.notch[0], filter_type="notch")
    if args.bandpass: explore_device.stream_processor.add_filter(cutoff_freq=(args.bandpass[0], args.bandpass[1]), filter_type="bandpass")

    explore_device.stream_processor.subscribe(rt_spectrogram.on_exg, topic=TOPICS.filtered_ExG)
    app.run()
