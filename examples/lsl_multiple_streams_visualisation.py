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
            as_pos["pos"][:, i] = np.stack((rolled["samples"][:, i], rolled["timestamps"][:].flatten()), axis=1)

class LslModule:
    def __init__(self,
                 inlet_type: str = "ExG",
                 on_quit: Union[Callable, None] = None,
                 timeout: float = 10.,
                 pull_timeout: float = 1.):
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
                                         "inlet": StreamInlet(s)})
                self.signal_buffers[s.name()] = SignalBuffer(name=s.name(),
                                                             ch_count=s.channel_count(),
                                                             max_samples=self.max_samples)

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


class SignalViewer:
    def __init__(self):
        self.canvas = scene.SceneCanvas
        self.lines = {}

    def add_line(self, name, pos, col=(1.,1.,1.,1.)):
        self.lines[name] = []
        line = scene.visuals.Line(pos=pos, color=col, parent=self.canvas.scene)
        line.transform = scene.transforms.STTransform()
        self.lines[name].append(line)

    def update_line(self, name, pos, col=(1.,1.,1.,1.)):
        x_width = pos[-1, 1] - pos[0, 1]
        self.lines[name].set_data(pos=pos, color=col)
        self.lines[name].transform.scale = [1./x_width, 1.]

    def on_timer(self):
        pass


class Communicator:
    def __init__(self):
        app.use_app("glfw")
        self.canvas_refresh_rate = 1. / 5.
        self.lsl_refresh_rate = 1. / 5.
        self.status_refresh_rate = 1. / 5.
        self.quit_flag = False

        self.lsl_module = LslModule(inlet_type="ExG", on_quit=self.quit)
        self.signal_module = SignalViewer()

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
