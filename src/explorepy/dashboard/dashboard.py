import numpy as np

from bokeh.layouts import column
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.server.server import Server
from bokeh.themes import Theme

EEG_SRATE = 250
EEG = np.random.rand(EEG_SRATE*100)
T = np.linspace(0, 100, EEG_SRATE*100)


class Dashboard:
    """Explorepy dashboard class"""
    def __init__(self, n_chan):
        self.n_chan = n_chan
        self.win_length = 10
        data = {'t': T[0:EEG_SRATE * self.win_length],
                'ch1': EEG[:EEG_SRATE * self.win_length]}
        self.source = ColumnDataSource(data=data)

    def start_server(self,):
        server = Server({'/': self._init_doc}, num_procs=1)
        server.start()
        server.io_loop.add_callback(server.show, "/")
        server.io_loop.start()

    def _init_doc(self, doc):
        global T, EEG
        T = np.delete(T, slice(0, EEG_SRATE * self.win_length, 1))
        EEG = np.delete(EEG, slice(0, EEG_SRATE * self.win_length, 1))
        plot = figure(y_range=(-1, 1), y_axis_label='Voltage (v)', title="EEG signal", plot_height=600, plot_width=900)
        plot.line(x='t', y='ch1', source=self.source, legend='Channel 1', line_width=2)
        plot.x_range.follow = "end"
        plot.x_range.follow_interval = self.win_length
        doc.add_root(column(plot))
        doc.theme = Theme(filename="theme.yaml")
        doc.add_periodic_callback(self._update, 100)

    def _update(self):
        global T, EEG
        new_data = {'t': T[:int(EEG_SRATE / 10)],
                    'ch1': EEG[:int(EEG_SRATE / 10)]}
        EEG = np.delete(EEG, slice(0, int(EEG_SRATE / 10), 1))
        T = np.delete(T, slice(0, int(EEG_SRATE / 10), 1))
        self.source.stream(new_data, rollover=EEG_SRATE * self.win_length)


if __name__ == '__main__':
    print('Opening Bokeh application on http://localhost:5006/')
    m_dashboard = Dashboard(n_chan=4)
    m_dashboard.start_server()

