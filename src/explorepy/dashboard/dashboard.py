import numpy as np
import time
from functools import partial
from threading import Thread

from bokeh.layouts import column
from bokeh.models import ColumnDataSource, WheelZoomTool, ResetTool
from bokeh.plotting import figure
from bokeh.server.server import Server
from bokeh.themes import Theme
from bokeh.models.widgets import Slider

from tornado import gen

EEG_SRATE = 250  # Hz
WIN_LENGTH = 10  # Seconds
CHAN_LIST = ['Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5', 'Ch6', 'Ch7', 'Ch8']
DEFAULT_SCALE = 10**-3  # Volt


class Dashboard:
    """Explorepy dashboard class"""

    def __init__(self, n_chan):
        self.n_chan = n_chan
        self.y_unit = DEFAULT_SCALE
        self.offsets = np.arange(self.n_chan)[::-1][:, np.newaxis].astype(float)
        self.chan_key_list = ['Ch' + str(i + 1) for i in range(self.n_chan)]
        exg_temp = self.offsets
        init_data = dict(zip(self.chan_key_list, exg_temp))
        init_data['t'] = np.array([0.])
        self.source = ColumnDataSource(data=init_data)
        self.server = None

    def start_server(self):
        self.server = Server({'/': self._init_doc}, num_procs=1)
        self.server.start()

    def start_loop(self):
        self.server.io_loop.add_callback(self.server.show, "/")
        self.server.io_loop.start()

    def _init_doc(self, doc):
        self.doc = doc

        # Create plot
        # wheel_zoom_tool = WheelZoomTool()
        # wheel_zoom_tool.dimensions = 'height'
        # wheel_zoom_tool.maintain_focus = False
        plot = figure(y_range=(-1, 4), y_axis_label='Voltage (v)', x_axis_label='Time (s)', title="EEG signal", plot_height=600, plot_width=1200,
                      tools=[ResetTool()], active_scroll=None, active_drag=None,
                      active_inspect=None, active_tap=None)

        for i in range(self.n_chan):
            plot.line(x='t', y=CHAN_LIST[i], source=self.source, legend='Channel '+ CHAN_LIST[i][-1], line_width=2)
        plot.x_range.follow = "end"
        plot.x_range.follow_interval = WIN_LENGTH

        # Create scale slider
        scale_slider = Slider(start=0, end=6, value=-np.log10(DEFAULT_SCALE), step=.5, title="Amplitude scale")
        scale_slider.on_change('value', self._change_scale)
        scale_slider.orientation = "horizontal"
        scale_slider.show_value = True
        scale_slider.tooltips = True

        # Add widgets to the doc
        self.doc.add_root(column(scale_slider, plot))

        # Set the theme
        self.doc.theme = Theme(filename="theme.yaml")

    @gen.coroutine
    def update(self, time_vector, ExG):
        ExG = self.offsets + ExG / self.y_unit
        new_data = dict(zip(self.chan_key_list, ExG))
        new_data['t'] = time_vector
        self.source.stream(new_data, rollover=EEG_SRATE * WIN_LENGTH)

    def _change_scale(self, attr, old, new):
        new, old = float(new), float(old)
        old_unit = 10 ** (-old)
        self.y_unit = 10**(-new)

        for ch, value in self.source.data.items():
            if ch in CHAN_LIST:
                temp_offset = self.offsets[CHAN_LIST.index(ch)]
                self.source.data[ch] = (value - temp_offset) * (old_unit/self.y_unit) + temp_offset


if __name__ == '__main__':
    print('Opening Bokeh application on http://localhost:5006/')
    m_dashboard = Dashboard(n_chan=4)
    m_dashboard.start_server()


    def my_loop():
        T = 0
        time.sleep(2)
        while True:
            time_vector = np.linspace(T, T + .2, 50)
            T += .2
            EEG = (np.random.rand(4, 50)-.5) * .0002
            m_dashboard.doc.add_next_tick_callback(partial(m_dashboard.update, time_vector=time_vector, ExG=EEG))
            time.sleep(0.2)


    thread = Thread(target=my_loop)
    thread.start()
    m_dashboard.start_loop()
