import time
from collections.abc import Callable
from typing import Union

from vispy import app, gloo, scene
import numpy as np
import pylsl
from pylsl import StreamInlet
import threading

class SignalBuffer:
    def __init__(self, name, ch_count, max_samples):
        self.name = name
        self.max_samples = max_samples
        self.ch_count = ch_count

        self.current_index = 0

        self.buffer = np.empty(shape=self.max_samples,
                               dtype=[("samples", np.float32, ch_count),
                                      ("timestamps", np.float32, 1)])

    def insert_sample(self, samples, timestamp):
        self.buffer["samples"][self.current_index] = samples
        self.buffer["timestamps"][self.current_index] = timestamp
        self.current_index += 1
        self.current_index %= self.max_samples

    def get_all(self, sorted: bool = True):
        if sorted:
            return np.roll(self.buffer, -self.current_index)
        else:
            return self.buffer

    def get_all_as_pos(self):
        """Return the internal buffer as buffer of positions"""
        rolled = self.get_all()
        as_pos = np.empty(shape=(self.max_samples, self.ch_count), dtype=[("pos", np.float32, 2)])
        for i in range(self.ch_count):
            as_pos["pos"][:, i] = np.stack((rolled["timestamps"][:].flatten(), rolled["samples"][:, i]), axis=1)
        return as_pos

class LslModule:
    def __init__(self,
                 inlet_type: str = "ExG",
                 on_quit: Union[Callable, None] = None,
                 timeout: float = 10.,
                 pull_timeout: float = .1):
        self.resolution_timeout = timeout
        self.pull_timeout = pull_timeout
        self.inlet_type = inlet_type
        self.on_quit = on_quit

        self.signal_buffers = {}
        self.max_samples = 10 * 250  # assume 10s at 250Hz

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
            if s.type() == self.inlet_type and "HyperSync" in s.name():
                self.inlet_dicts.append({"name": s.name(),
                                         "type": s.type(),
                                         "ch_count": s.channel_count(),
                                         "inlet": StreamInlet(s, processing_flags=pylsl.proc_clocksync)})
                self.signal_buffers[s.name()] = SignalBuffer(name=s.name(),
                                                             ch_count=s.channel_count(),
                                                             max_samples=self.max_samples)
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
    def __init__(self, lsl_module: LslModule):
        self.lsl_module = lsl_module
        self.time_window = 10.
        self.bg_colour = (0., 0., 0., 1.)  # black
        self.line_colour = (1., 0., 0., 1.)  # red
        self.canvas = scene.SceneCanvas(bgcolor=self.bg_colour)
        self.canvas.show()
        self.lines = {}

    def add_line(self, name, pos, col=None):
        if not col:
            col = self.line_colour
        self.lines[name] = []
        line = scene.visuals.Line(pos=pos, color=col, parent=self.canvas.scene)
        line.transform = scene.transforms.STTransform()
        self.lines[name] = line

    def update_line(self, name, pos, col=None):
        if not col:
            col = self.line_colour
        self.lines[name].set_data(pos=pos, color=col)

    def update_transforms(self):
        right_x = pylsl.local_clock()
        left_x = right_x - self.time_window
        line_keys = list(self.lines)
        n_keys = len(line_keys)
        for i in range(len(line_keys)):
            self.lines[line_keys[i]].transform.scale = [1. / self.time_window * self.canvas.size[0], 1.]
            # translate is applied *after* scaling - which includes scaling with window size!
            self.lines[line_keys[i]].transform.translate = [-left_x/self.time_window * self.canvas.size[0],
                                                    400000. + ((i+1)/(n_keys+1)) * self.canvas.size[1]]  # mean (or baseline) + line offset in [0., 1.] * self.canvas.size[0]

    def on_timer(self, event):
        buffers = self.lsl_module.get_signal_buffers()
        for name in buffers.keys():
            lines = buffers[name].get_all_as_pos()
            n_lines = lines.shape[1]
            for i in range(n_lines):
                line = lines["pos"][:, i]
                line_name = f"{name}_{i}"
                if line_name not in self.lines.keys():
                    self.add_line(name=line_name, pos=line)
                else:
                    self.update_line(name=line_name, pos=line)
        self.update_transforms()


class Communicator:
    def __init__(self):
        app.use_app("glfw")
        self.canvas_refresh_rate = 1. / 30.
        self.lsl_refresh_rate = 1. / 30.
        self.status_refresh_rate = 1. / 10.
        self.quit_flag = False

        self.lsl_module = LslModule(inlet_type="ExG", on_quit=self.quit)
        self.signal_module = SignalViewer(self.lsl_module)
        # note that the signal module is limited in update rate by LSL!

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
    c = Communicator()
    app.run()

if __name__ == '__main__':
    main()
