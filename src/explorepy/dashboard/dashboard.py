import numpy as np
import time
from functools import partial
from threading import Thread

from bokeh.layouts import column
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.server.server import Server
from bokeh.themes import Theme
from tornado import gen


EEG_SRATE = 250
T = 0
EEG = np.array([[0.], [1.], [2.], [3.]])


class Dashboard:
    """Explorepy dashboard class"""
    def __init__(self, n_chan):
        self.n_chan = n_chan
        self.win_length = 10
        data = {'t': np.array([0.]),
                'ch1': EEG[0, :],
                'ch2': EEG[1, :],
                'ch3': EEG[2, :],
                'ch4': EEG[3, :],
                }
        self.source = ColumnDataSource(data=data)

    def start_server(self):
        self.server = Server({'/': self._init_doc}, num_procs=1)
        self.server.start()

    def start_loop(self):
        self.server.io_loop.add_callback(self.server.show, "/")
        self.server.io_loop.start()

    def _init_doc(self, doc):
        self.doc = doc
        global T, EEG
        plot = figure(y_range=(0, 4), y_axis_label='Voltage (v)', title="EEG signal", plot_height=700, plot_width=900)
        plot.line(x='t', y='ch1', source=self.source, legend='Channel 1', line_width=2)
        plot.line(x='t', y='ch2', source=self.source, legend='Channel 2', line_width=2)
        plot.line(x='t', y='ch3', source=self.source, legend='Channel 3', line_width=2)
        plot.line(x='t', y='ch4', source=self.source, legend='Channel 4', line_width=2)
        plot.x_range.follow = "end"
        plot.x_range.follow_interval = self.win_length
        self.doc.add_root(column(plot))
        self.doc.theme = Theme(filename="theme.yaml")
        # self.doc.add_periodic_callback(self.update_periodically, 200)

    def update_periodically(self):
        pass

    @gen.coroutine
    def update(self, time_vector, EEG):
        # global T, EEG

        # time_vector = np.linspace(T, T+.2, 50)
        # T += .2
        # EEG = np.random.rand(self.n_chan, 50)*.5
        new_data = {'t': time_vector,
                    'ch1': EEG[0, :],
                    'ch2': EEG[1, :]+1,
                    'ch3': EEG[2, :]+2,
                    'ch4': EEG[3, :]+3,
                    }
        self.source.stream(new_data, rollover=EEG_SRATE * self.win_length)


if __name__ == '__main__':
    print('Opening Bokeh application on http://localhost:5006/')
    m_dashboard = Dashboard(n_chan=4)
    m_dashboard.start_server()

    def my_loop():
        time.sleep(1)
        global T, EEG
        while True:
            time.sleep(0.2)
            time_vector = np.linspace(T, T+.2, 50)
            T += .2
            EEG = np.random.rand(4, 50)*.5
            m_dashboard.doc.add_next_tick_callback(partial(m_dashboard.update, time_vector=time_vector, EEG=EEG))


    thread = Thread(target=my_loop)
    thread.start()
    m_dashboard.start_loop()
