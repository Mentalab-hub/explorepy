import time
from argparse import ArgumentParser
from collections.abc import Callable
from typing import Union

from vispy import app, gloo, scene
import numpy as np
import pylsl
from pylsl import StreamInlet
import threading

class SignalBuffer:
    _DEFAULT_SCALE = 100.  # in uV
    _OFFSET_MODES = ["mean", "fixed"]
    _DEFAULT_OFFSET_MODE = _OFFSET_MODES[0]
    def __init__(self, name, ch_count, max_samples, offset_mode="mean", fixed_offset=0.0):
        self.name = name
        self.ch_count = ch_count
        self.max_samples = max_samples

        self.offset_mode = offset_mode if offset_mode in self._OFFSET_MODES else self._DEFAULT_OFFSET_MODE

        self.offsets = [fixed_offset] * ch_count
        self.scales = [self._DEFAULT_SCALE] * ch_count

        self.current_index = 0

        self.buffer = np.empty(shape=self.max_samples,
                               dtype=[("samples", np.float64, ch_count),
                                      ("timestamps", np.float64, 1)])

    def insert_sample(self, samples, timestamp):
        self.buffer["samples"][self.current_index] = samples
        self.buffer["timestamps"][self.current_index] = timestamp
        self.current_index += 1
        self.current_index %= self.max_samples
        if self.offset_mode == "mean":
            for i in range(self.buffer["samples"].shape[1]):
                self.offsets[i] = -self.buffer["samples"][:, i].mean()

    def get_all(self, sorted: bool = True):
        if sorted:
            return np.roll(self.buffer, -self.current_index)
        else:
            return self.buffer

    def get_all_as_pos(self):
        """Return the internal buffer as buffer of positions"""
        rolled = self.get_all()
        as_pos = np.empty(shape=(self.max_samples, self.ch_count), dtype=[("pos", np.float64, 2)])
        for i in range(self.ch_count):
            as_pos["pos"][:, i] = np.stack((rolled["timestamps"][:].flatten(), rolled["samples"][:, i]), axis=1)
        return as_pos

    def get_scales(self):
        return self.scales

    def set_scales(self, new_scales):
        self.scales = new_scales

    def get_offsets(self):
        return self.offsets

    def set_offsets(self, new_offsets):
        self.offsets = new_offsets

class LslModule:
    def __init__(self,
                 inlet_type: str = "ExG",
                 on_quit: Union[Callable, None] = None,
                 timeout: float = 10.,
                 pull_timeout: float = .2,
                 devices=None,
                 offset_mode="mean",
                 offset=0.0,
                 max_sr=1000,
                 use_hypersync=True):
        self.resolution_timeout = timeout
        self.pull_timeout = pull_timeout
        self.inlet_type = inlet_type
        self.on_quit = on_quit

        self.use_hypersync = use_hypersync

        self.devices = devices
        self.offset_mode = offset_mode
        self.fixed_offset = offset

        self.signal_buffers = {}
        self.max_samples = 10 * max_sr  # assume 10s at 250Hz

        self.inlet_dicts: list[dict[str, StreamInlet]] = []
        self.inlet_finder = threading.Thread(target=self.find_inlets)
        self.inlet_finder.start()

    def quit(self):
        if self.on_quit:
            self.on_quit()

    def find_inlets(self):
        print("Attempting to find stream inlets...")
        found_streams = pylsl.resolve_streams(self.resolution_timeout)
        if len(found_streams) <= 0:
            print("Could not find any streams.")
            self.quit()
        for s in found_streams:
            stream_name = s.name()
            stream_found = s.type() == self.inlet_type
            if self.use_hypersync:
                stream_found = stream_found and "HyperSync" in stream_name
            else:
                stream_found = stream_found and "HyperSync" not in stream_name

            if stream_found:
                print(f"Found stream that matches inlet type (ExG): {stream_name}")
                stream_found = False

                if not self.devices:
                    stream_found = True
                else:
                    for device_id in self.devices:
                        if device_id in stream_name:
                            stream_found = True

                if stream_found:
                    self.inlet_dicts.append({"name": s.name(),
                                             "type": s.type(),
                                             "ch_count": s.channel_count(),
                                             "inlet": StreamInlet(s, processing_flags=pylsl.proc_clocksync)})
                    self.signal_buffers[s.name()] = SignalBuffer(name=s.name(),
                                                                 ch_count=s.channel_count(),
                                                                 max_samples=self.max_samples,
                                                                 offset_mode=self.offset_mode,
                                                                 fixed_offset=self.fixed_offset)
        if len(self.inlet_dicts) <= 0:
            self.quit()

    def pull_data(self):
        for inlet_dict in self.inlet_dicts:
            samples, timestamps = inlet_dict["inlet"].pull_chunk(timeout=self.pull_timeout)
            for i in range(len(timestamps)):
                self.signal_buffers[inlet_dict["name"]].insert_sample(samples[i], timestamps[i])

    def on_timer(self, event):
        if self.inlet_finder.is_alive():
            return
        if len(self.inlet_dicts) <= 0:
            return
        self.pull_data()

    def get_signal_buffers(self):
        return self.signal_buffers


class SignalViewer:
    _LINE_COLOURS = [(.8, .1, .1, 1.0),
                     (.1, .8, .1, 1.0),
                     (.1, .1, .8, 1.0),
                     (.8, .8, .1, 1.0),
                     (.1, .8, .8, 1.0),
                     (.8, .1, .8, 1.0),
                     (.8, .1, .8, 1.0),]
    def __init__(self, lsl_module: LslModule):
        self.lsl_module = lsl_module
        self.time_window = 10.
        self.bg_colour = (0.1, 0.1, 0.1, 1.)  # black
        self.line_colour = (0., 0., 1., 1.)  # blue, default
        self.canvas = scene.SceneCanvas(bgcolor=self.bg_colour)
        self.canvas.show()
        self.lines = {}
        self.line_scales = {}
        self.lsl_time = 0.0
        self.device_colors = {}

    def add_line(self, name, pos, offset=0.0, scale=1.0, col=None):
        if not col:
            col = self.line_colour
        self.lines[name] = []
        pos[:, 0] -= self.lsl_time - self.time_window
        pos[:, 1] += offset
        line = scene.visuals.Line(pos=pos, color=col, parent=self.canvas.scene)
        line.transform = scene.transforms.STTransform()
        self.lines[name] = line
        self.line_scales[name] = scale

    def update_line(self, name, pos, offset=0.0, scale=1.0, col=None):
        if not col:
            col = self.line_colour
        # start at 0 (x), center at 0 (y)
        # this mitigates loss of precision from the float32 limit from vispy / OpenGL, to preserve precision totally,
        # scaling could also be performed here instead of with the line's transform
        # however, if this is run on the GPU, leaving scaling up to the transform will be a LOT more efficient than
        # doing it here (this holds true for offsetting the line as well but slower offsetting is preferable to
        # precision loss...)
        pos[:, 0] -= self.lsl_time - self.time_window
        pos[:, 1] += offset
        self.lines[name].set_data(pos=pos, color=col)
        self.line_scales[name] = scale

    def update_transforms(self):
        line_keys = list(self.lines)
        n_keys = len(line_keys)
        for i in range(len(line_keys)):
            self.lines[line_keys[i]].transform.scale = [1. / self.time_window * self.canvas.size[0],  # x (timestamp)
                                                        1./self.line_scales[line_keys[i]]]  # y (signal)
            self.lines[line_keys[i]].transform.translate = [0.0, # x (timestamp)
                                                            ((i+1)/(n_keys+1)) * self.canvas.size[1]]  # y (signal)

    def set_device_color(self, name, n):
        self.device_colors[name] = self._LINE_COLOURS[n%len(self._LINE_COLOURS)]

    def on_timer(self, event):
        buffers = self.lsl_module.get_signal_buffers()
        self.lsl_time = pylsl.local_clock()  # get local clock as common reference for all signals
        for name in buffers.keys():
            if name not in self.device_colors:
                self.set_device_color(name, len(self.device_colors.keys()))
            lines = buffers[name].get_all_as_pos()
            offsets = buffers[name].get_offsets()
            scales = buffers[name].get_scales()
            n_lines = lines.shape[1]
            for i in range(n_lines):
                line = lines["pos"][:, i]
                line_name = f"{name}_{i}"
                if line_name not in self.lines.keys():
                    self.add_line(name=line_name, pos=line, offset=offsets[i], scale=scales[i], col=self.device_colors[name])
                else:
                    self.update_line(name=line_name, pos=line, offset=offsets[i], scale=scales[i], col=self.device_colors[name])
        self.update_transforms()


class Communicator:
    def __init__(self, scale=100., devices=None, offset_mode="mean", offset=0.0, hypersync=True):
        app.use_app("glfw")
        self.canvas_refresh_rate = 1. / 20.
        self.lsl_refresh_rate = 1. / 20.
        self.status_refresh_rate = 1. / 10.
        self.quit_flag = False

        self.lsl_module = LslModule(inlet_type="ExG", on_quit=self.quit, devices=devices, offset_mode=offset_mode,
                                    offset=offset, use_hypersync=hypersync)
        self.signal_module = SignalViewer(self.lsl_module)
        # note that the signal module is limited in update rate by LSL, not the other way around!

        self.canvas_timer = app.Timer(start=True, interval=self.canvas_refresh_rate, connect=self.signal_module.on_timer)
        self.lsl_timer = app.Timer(start=True, interval=self.lsl_refresh_rate, connect=self.lsl_module.on_timer)
        self.run_timer = app.Timer(start=True, interval=self.status_refresh_rate, connect=self.run)

    def quit(self):
        self.quit_flag = True

    def run(self, event):
        if self.quit_flag:
            print("Got request to quit...")
            self.canvas_timer.stop()
            self.lsl_timer.stop()
            app.quit()

def main():
    args = ArgumentParser()
    args.add_argument("-s", "--scale", default=100.0, type=float, required=False,
                      help="The scale (in uV) to apply to the plots (the plots will scale to +- scale, meaning that "
                           "the range it will be scaled to is twice the chosen scale value)")
    args.add_argument("-d", "--devices", nargs='*', required=False, default=None, type=str,
                      help="The device IDs to connect to (default is None, meaning all found Explore HyperSync streams "
                           "will be connected to)")
    args.add_argument("--offset_mode", default="mean", type=str, choices=["fixed", "mean"],
                      help="The mode to calculate the offset for the signals to center them in the plot. If \"fixed\", "
                           "a fixed value should be supplied to add to the signals. If \"mean\", the mean of the "
                           "visible signal will be subtracted from the signal to center it in the plot.")
    args.add_argument("--offset", default=0.0, type=float, required=False,
                      help="The offset to add to the signal values if the offset mode is \"fixed\"")
    if __debug__:
        args.add_argument("--hypersync", default=1, type=int, required=False,
                          help="Whether to fetch hypersync streams or \"raw\" streams")
    args = args.parse_args()
    use_hypersync = True
    if __debug__:
        use_hypersync = bool(args.hypersync)
        print(f"Hypersync is {use_hypersync}")

    c = Communicator(scale=args.scale, devices=args.devices, offset_mode=args.offset_mode, offset=args.offset,
                     hypersync=use_hypersync)
    app.run()

if __name__ == '__main__':
    main()
