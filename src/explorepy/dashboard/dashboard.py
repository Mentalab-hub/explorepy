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

EEG_SRATE = 250  # Hz
WIN_LENGTH = 10  # Seconds
CHAN_LIST = ['Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5', 'Ch6', 'Ch7', 'Ch8']


class Dashboard:
    """Explorepy dashboard class"""

    def __init__(self, n_chan):
        self.n_chan = n_chan
        self.y_unit = 1.  # volt
        self.chan_key_list = ['Ch' + str(i + 1) for i in range(self.n_chan)]
        init_data = dict(zip(self.chan_key_list, self.y_unit * np.arange(self.n_chan)[::-1][:, np.newaxis]))
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
        plot = figure(y_range=(-1, 4), x_range=(0,10), y_axis_label='Voltage (v)', title="EEG signal", plot_height=700, plot_width=900)

        for i in range(self.n_chan):
            plot.line(x='t', y=CHAN_LIST[i], source=self.source, legend='Channel '+ CHAN_LIST[i][-1], line_width=2)

        plot.x_range.follow = "end"
        plot.x_range.follow_interval = WIN_LENGTH
        self.doc.add_root(column(plot))
        self.doc.theme = Theme(filename="theme.yaml")

    def update_periodically(self):
        pass

    @gen.coroutine
    def update(self, time_vector, ExG):
        ExG += (self.y_unit * np.arange(self.n_chan)[::-1][:, np.newaxis])
        new_data = dict(zip(self.chan_key_list, ExG))
        new_data['t'] = time_vector
        # new_data = {'t': time_vector,
        #             'Ch1': ExG[0, :] + 3,
        #             'Ch2': ExG[1, :] + 2,
        #             'Ch3': ExG[2, :] + 1,
        #             'Ch4': ExG[3, :],
        #             }
        self.source.stream(new_data, rollover=EEG_SRATE * WIN_LENGTH)


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
            EEG = np.random.rand(4, 50) * .5
            m_dashboard.doc.add_next_tick_callback(partial(m_dashboard.update, time_vector=time_vector, ExG=EEG))
            time.sleep(0.2)


    thread = Thread(target=my_loop)
    thread.start()
    m_dashboard.start_loop()
