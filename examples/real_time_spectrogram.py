import vispy.plot

import explorepy
from explorepy import Explore
from explorepy.packet import EEG
from explorepy.stream_processor import TOPICS
from vispy import app
from vispy.visuals import ImageVisual
from vispy import visuals
from scipy.signal import stft
import numpy as np

class SpectrogramCanvas(app.Canvas):
    def __init__(self, device_sr=250, update_window=0.5, time_window=10, mode="overlapping"):
        super().__init__()

        self.last_value = 0
        self.max_value = 0

        self.valid_modes = ["moving", "overlapping"]
        self.mode = mode

        if self.mode not in self.valid_modes:
            raise ValueError(f"Encountered unexpected drawing mode: {self.mode}, valid modes: {self.valid_modes}")

        self.drop_num = -1
        self.update_window = update_window
        if self.mode == "moving":
            self.num_vals = 2 ** np.ceil(np.log2(device_sr * update_window))
        elif self.mode == "overlapping":
            self.num_vals = 2 ** np.ceil(np.log2(device_sr * time_window))
            self.drop_num = int(device_sr * update_window)
        print(f"Attempting to update every {self.num_vals} values")

        self.device_sr = device_sr

        self.time_window = time_window  # in s, this is a suggested time window, not fixed

        self.current_index = 0
        self.data_buf = []

        self.current_size = None

        self.size = (1024, 1024)
        self.img = np.full(shape=(1024, 1024), fill_value=0.5, dtype=np.float32)
        # TODO: set clim correctly
        # TODO: adapt cmap
        self.plot = ImageVisual(self.img, clim=[0.0, 1.0])

        self.timer = app.Timer(start=True, interval=1./30., connect=self.create_window)

    def create_window(self, event):
        """Used to instantiate the window initially, has to be called from the main thread!"""
        if self.current_size is not None:
            self.size = self.current_size
            self.show()
            self.timer.stop()

    def on_resize(self, event):
        if self.current_size is None:
            return
        transform_system = visuals.transforms.TransformSystem()
        transform_system.configure(canvas=self)
        self.plot.transforms = transform_system

    def on_exg(self, packet: EEG):
        """Called by explorepy from a thread that isn't the main thread, triggers canvas updates"""
        ret = packet.get_data()[1]
        self.data_buf.append(ret)
        if len(self.data_buf) >= self.num_vals:
            Zxx_channels = []
            for i in range(len(self.data_buf[0])):
                # f contains the frequencies (y-axis)
                # t contains the segment timestamps
                # Zxx contains the corresponding magnitude
                f, t, Zxx = stft(np.asarray(self.data_buf)[:, i, 0], fs=self.device_sr, nperseg=256)
                self.max_value = max(np.max(np.abs(Zxx)), self.max_value)
                Zxx_channels.append(Zxx)
            Zxx = np.mean(Zxx_channels, axis=0)
            Zxx = np.rot90(np.rot90(Zxx))
            y_max = Zxx.shape[0]
            x_max = Zxx.shape[1]

            if self.current_size is None:
                self.current_size = self.get_initial_size(x_max, y_max)

                # Set the numpy array to the same size as the calculated window size
                self.img = np.full(shape=(self.current_size[1], self.current_size[0]), fill_value=0.5, dtype=np.float32)

            # Overwrite the next section in the numpy array with the new STFT values
            self.img[0:y_max, self.current_index:self.current_index+x_max] = np.abs(Zxx)
            self.last_value += 1

            if self.mode == "moving":
                self.current_index += x_max

                # Draw a rudimentary "swipe line" (the colour is single-valued, so we're limited in choices here)
                # Note that self.current_index + 2 can be outside the array boundary!
                self.img[:, self.current_index:self.current_index+2] = 0.0

                if self.current_size is not None:
                    self.current_index %=self.current_size[0]

            self.plot.set_data(self.img)
            self.plot.clim = (0.0, self.max_value)
            self.plot.update()
            self.update()
            if self.mode == "moving":
                self.data_buf = []
            elif self.mode == "overlapping":
                self.data_buf = self.data_buf[self.drop_num:]

    def get_initial_size(self, x_max, y_max):
        if self.mode == "moving":
            return int(x_max * (self.time_window // self.update_window)), y_max
        elif self.mode == "overlapping":
            return x_max, y_max
        else:
            raise ValueError(f"Encountered unexpected drawing mode while trying to get initial size: {self.mode}")

    def on_draw(self, event):
        self.plot.draw()


class SpectrogramPlot:
    def __init__(self, device_sr=250, update_window=0.5, time_window=10, mode="overlapping"):
        self.fig = vispy.plot.Fig(size=(800, 600), show=False)
        self.plot = self.fig[0, 0]
        self.plot.bgcolor = "#FFFFFF"
        self.img = np.full(shape=(512, 512), fill_value=0.0, dtype=np.float32)
        self.plot_image = None

        self.last_value = 0
        self.max_value = 0

        self.valid_modes = ["moving", "overlapping"]
        self.mode = mode

        if self.mode not in self.valid_modes:
            raise ValueError(f"Encountered unexpected drawing mode: {self.mode}, valid modes: {self.valid_modes}")

        self.drop_num = -1
        self.update_window = update_window
        if self.mode == "moving":
            self.num_vals = 2 ** np.ceil(np.log2(device_sr * update_window))
        elif self.mode == "overlapping":
            self.num_vals = 2 ** np.ceil(np.log2(device_sr * time_window))
            self.drop_num = int(device_sr * update_window)
        print(f"Attempting to update every {self.num_vals} values")

        self.device_sr = device_sr

        self.time_window = time_window  # in s, this is a suggested time window, not fixed

        self.current_index = 0
        self.data_buf = []

        self.current_size = None
        self.timer = app.Timer(start=True, interval=1. / 30., connect=self.create_window)

    def create_window(self, event):
        """Used to instantiate the window initially, has to be called from the main thread!"""
        if self.current_size is not None:
            self.plot_image = self.plot.image(self.img, cmap="coolwarm", clim=(0.0, 4.0))
            self.plot.colorbar(position="left", label=" ", cmap="coolwarm", clim=("0", "4.0"))
            self.fig.show()
            self.timer.stop()

    def on_exg(self, packet: EEG):
        """Called by explorepy from a thread that isn't the main thread, triggers canvas updates"""
        ret = packet.get_data()[1]
        self.data_buf.append(ret)
        if len(self.data_buf) >= self.num_vals:
            Zxx_channels = []
            for i in range(len(self.data_buf[0])):
                # f contains the frequencies (y-axis)
                # t contains the segment timestamps
                # Zxx contains the corresponding magnitude
                f, t, Zxx = stft(np.asarray(self.data_buf)[:, i, 0], fs=self.device_sr, nperseg=128)
                self.max_value = max(np.max(np.abs(Zxx)), self.max_value)
                Zxx_channels.append(Zxx)
            Zxx = np.mean(Zxx_channels, axis=0)
            #Zxx = np.rot90(np.rot90(Zxx))
            y_max = Zxx.shape[0]
            x_max = Zxx.shape[1]

            if self.current_size is None:
                self.current_size = self.get_initial_size(x_max, y_max)

                # Set the numpy array to the same size as the calculated window size
                self.img = np.full(shape=(self.current_size[1], self.current_size[0]), fill_value=0.5, dtype=np.float32)

            # Overwrite the next section in the numpy array with the new STFT values
            self.img[0:y_max, self.current_index:self.current_index+x_max] = np.abs(Zxx)
            self.last_value += 1

            if self.mode == "moving":
                self.current_index += x_max

                # Draw a rudimentary "swipe line" (the colour is single-valued, so we're limited in choices here)
                # Note that self.current_index + 2 can be outside the array boundary!
                self.img[:, self.current_index:self.current_index+2] = 0.0

                if self.current_size is not None:
                    self.current_index %=self.current_size[0]

            #self.plot.set_data(self.img)
            if self.plot_image is not None:
                self.plot_image.set_data(self.img)
                self.plot_image.clim = (0.0, self.max_value)
                self.fig.update()

            if self.mode == "moving":
                self.data_buf = []
            elif self.mode == "overlapping":
                self.data_buf = self.data_buf[self.drop_num:]

    def get_initial_size(self, x_max, y_max):
        if self.mode == "moving":
            return int(x_max * (self.time_window // self.update_window)), y_max
        elif self.mode == "overlapping":
            return x_max, y_max
        else:
            raise ValueError(f"Encountered unexpected drawing mode while trying to get initial size: {self.mode}")




if __name__ == '__main__':
    device_name = "Explore_DABB"
    device_sr = 250
    update_window = 0.5 # in seconds, determines the update rate but also how many values are considered for the STFT
    time_window = 10 # in seconds, determines the time window shown (approximately)
    # Note that the window will always show a little more than the time window chosen!

    drawing_mode = "plot"
    plot_class_dict = {"canvas": SpectrogramCanvas, "plot": SpectrogramPlot}

    rt_spectrogram = plot_class_dict[drawing_mode](device_sr=device_sr, update_window=update_window, time_window=time_window,
                                       mode="overlapping")

    explorepy.set_bt_interface("ble")  # Remove for Bluetooth connection

    explore_device = Explore()
    explore_device.connect(device_name=device_name)
    explore_device.set_sampling_rate(device_sr)
    explore_device.stream_processor.subscribe(rt_spectrogram.on_exg, topic=TOPICS.raw_ExG)
    app.run()
