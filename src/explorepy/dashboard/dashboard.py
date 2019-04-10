import numpy as np
import time
from functools import partial
from threading import Thread
import os.path

from bokeh.layouts import column, layout, widgetbox, row
from bokeh.models import ColumnDataSource, ResetTool, PrintfTickFormatter
from bokeh.plotting import figure
from bokeh.server.server import Server
from bokeh.themes import Theme
from bokeh.models.widgets import Select, DataTable, TableColumn
from bokeh.models import SingleIntervalTicker

from tornado import gen

EEG_SRATE = 250  # Hz
WIN_LENGTH = 10  # Seconds
CHAN_LIST = ['Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5', 'Ch6', 'Ch7', 'Ch8']
DEFAULT_SCALE = 10 ** -3  # Volt
SCALE_MENU = {"1 uV": 6., "5 uV": 5.3333, "10 uV": 5., "100 uV": 4., "500 uV": 3.3333, "1 mV": 3., "5 mV": 2.3333,
              "10 mV": 2., "100 mV": 1.}
TIME_RANGE_MENU = {"10 s": 10., "5 s": 5., "20 s": 20.}


class Dashboard:
    """Explorepy dashboard class"""

    def __init__(self, n_chan):
        self.n_chan = n_chan
        self.y_unit = DEFAULT_SCALE
        self.offsets = np.arange(1, self.n_chan + 1)[::-1][:, np.newaxis].astype(float)
        self.chan_key_list = ['Ch' + str(i + 1) for i in range(self.n_chan)]

        # Init ExG data source
        exg_temp = self.offsets
        init_data = dict(zip(self.chan_key_list, exg_temp))
        init_data['t'] = np.array([0.])
        self.exg_source = ColumnDataSource(data=init_data)

        # Init Device info table sources
        self.firmware_source = ColumnDataSource(data={'firmware_version': ['NA']})
        self.battery_source = ColumnDataSource(data={'battery': ['NA']})
        self.temperature_source = ColumnDataSource(data={'temperature': ['NA']})
        self.light_source = ColumnDataSource(data={'light': ['NA']})

        self.server = None

    def start_server(self):
        self.server = Server({'/': self._init_doc}, num_procs=1)
        self.server.start()

    def start_loop(self):
        self.server.io_loop.add_callback(self.server.show, "/")
        self.server.io_loop.start()

    def _init_doc(self, doc):
        self.doc = doc
        self.doc.title = "Explore Dashboard"
        # Create plot
        self.plot = figure(y_range=(0.01, self.n_chan + 1 - 0.01), y_axis_label='Voltage', x_axis_label='Time (s)',
                           title="EEG signal",
                           plot_height=600, plot_width=1270,
                           y_minor_ticks=int(10),
                           tools=[ResetTool()], active_scroll=None, active_drag=None,
                           active_inspect=None, active_tap=None)

        # Set yaxis properties
        self.plot.yaxis.ticker = SingleIntervalTicker(interval=1, num_minor_ticks=10)

        # Initial plot line
        for i in range(self.n_chan):
            self.plot.line(x='t', y=CHAN_LIST[i], source=self.exg_source,
                           line_width=1.5)

        # Initial vertical line
        self.plot.line(x=[0, 0],
                       y=[self.offsets[-1]-2, self.offsets[0]+2],
                       name='vertical_line',
                       line_width=2,
                       color='red', alpha=.8)

        self.plot.x_range.follow = "end"
        self.plot.x_range.follow_interval = WIN_LENGTH

        # Create controls
        self.t_range = Select(title="Time Range", value="10 s", options=list(TIME_RANGE_MENU.keys()), width=210)
        self.t_range.on_change('value', self._change_t_range)
        self.y_scale = Select(title="Y-axis Scale", value="1 mV", options=list(SCALE_MENU.keys()), width=210)
        self.y_scale.on_change('value', self._change_scale)

        # Create device info tables
        columns = [TableColumn(field='firmware_version', title="Firmware Version")]
        self.firmware = DataTable(source=self.firmware_source, index_position=None, sortable=False, reorderable=False,
                                  columns=columns, width=200, height=50)

        columns = [TableColumn(field='battery', title="Battery")]
        self.battery = DataTable(source=self.battery_source, index_position=None, sortable=False, reorderable=False,
                                 columns=columns, width=200, height=50)

        columns = [TableColumn(field='temperature', title="temperature")]
        self.temperature = DataTable(source=self.temperature_source, index_position=None, sortable=False,
                                     reorderable=False, columns=columns, width=200, height=50)

        columns = [TableColumn(field='light', title="light")]
        self.light = DataTable(source=self.light_source, index_position=None, sortable=False, reorderable=False,
                               columns=columns, width=200, height=50)

        # Add widgets to the doc
        m_widgetbox = widgetbox([self.y_scale, self.t_range, self.firmware,
                                 self.battery, self.temperature, self.light],
                                width=220)

        self.doc.add_root(row([m_widgetbox, self.plot]))

        # Set the theme
        module_path = os.path.dirname(__file__)
        self.doc.theme = Theme(filename=os.path.join(module_path, "theme.yaml"))
        self.plot.ygrid.minor_grid_line_color = 'White'
        self.plot.ygrid.minor_grid_line_alpha = 0.05

        # Set the formatting of yaxis ticks' labels
        self.plot.yaxis[0].formatter = PrintfTickFormatter(format="Ch %i")

    @gen.coroutine
    def update_exg(self, time_vector, ExG):
        """Update ExG data in the visualization

        Args:
            time_vector (list): time vector
            ExG (np.ndarray): array of new data

        """
        # Delete old vertical line
        vertical_line = self.plot.select_one({'name': 'vertical_line'})
        self.plot.renderers.remove(vertical_line)

        # Update vertical line
        self.plot.line(x=[time_vector[-1], time_vector[-1]],
                       y=[self.offsets[-1] - 2, self.offsets[0] + 2],
                       name='vertical_line',
                       line_width=2,
                       color='red', alpha=.8)

        # Update ExG data
        ExG = self.offsets + ExG / self.y_unit
        new_data = dict(zip(self.chan_key_list, ExG))
        new_data['t'] = time_vector
        self.exg_source.stream(new_data, rollover=2 * EEG_SRATE * WIN_LENGTH)


    @gen.coroutine
    def update_info(self, new):
        """Update device information in the dashboard

        Args:
            new(dict): Dictionary of new values

        """
        for key in new.keys():
            data = {key: new[key]}
            if key == 'firmware_version':
                self.firmware_source.stream(data, rollover=1)
            elif key == 'battery':
                self.battery_source.stream(data, rollover=1)
            elif key == 'temperature':
                self.temperature_source.stream(data, rollover=1)
            elif key == 'light':
                self.light_source.stream(data, rollover=1)
            else:
                print("Warning: There is no field named: " + key)

    def _change_scale(self, attr, old, new):
        new, old = SCALE_MENU[new], SCALE_MENU[old]
        old_unit = 10 ** (-old)
        self.y_unit = 10 ** (-new)

        for ch, value in self.exg_source.data.items():
            if ch in CHAN_LIST:
                temp_offset = self.offsets[CHAN_LIST.index(ch)]
                self.exg_source.data[ch] = (value - temp_offset) * (old_unit / self.y_unit) + temp_offset

    def _change_t_range(self, attr, old, new):
        self.plot.x_range.follow = "end"
        self.plot.x_range.follow_interval = TIME_RANGE_MENU[new]


if __name__ == '__main__':
    print('Opening Bokeh application on http://localhost:5006/')
    m_dashboard = Dashboard(n_chan=8)
    m_dashboard.start_server()


    def my_loop():
        T = 0
        time.sleep(2)
        while True:
            time_vector = np.linspace(T, T + .2, 50)
            T += .2
            EEG = (np.random.randint(0, 2, (8, 50)) - .5) * .0002  # (np.random.rand(8, 50)-.5) * .0005
            m_dashboard.doc.add_next_tick_callback(partial(m_dashboard.update_exg, time_vector=time_vector, ExG=EEG))

            device_info_attr = ['firmware_version', 'battery', 'temperature', 'light']
            device_info_val = [['2.0.4'], ['95'], ['21'], ['13']]
            new_data = dict(zip(device_info_attr, device_info_val))
            m_dashboard.doc.add_next_tick_callback(partial(m_dashboard.update_info, new=new_data))

            time.sleep(0.2)


    thread = Thread(target=my_loop)
    thread.start()
    m_dashboard.start_loop()
