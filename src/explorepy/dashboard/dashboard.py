# -*- coding: utf-8 -*-
"""Dashboard module"""
import logging
import os
from datetime import datetime
from functools import partial

import numpy as np
from bokeh.core.property.validation import (
    validate,
    without_property_validation
)
from bokeh.layouts import (
    Spacer,
    column,
    row,
    widgetbox
)
from bokeh.models import (
    BoxZoomTool,
    Button,
    CheckboxGroup,
    ColumnDataSource,
    Div,
    Panel,
    RadioGroup,
    ResetTool,
    SingleIntervalTicker,
    Tabs,
    TextInput,
    Toggle,
    widgets
)
from bokeh.palettes import Category20
from bokeh.plotting import figure
from bokeh.server.server import Server
from bokeh.themes import Theme
from bokeh.transform import dodge
from jinja2 import Template
from scipy.ndimage import gaussian_filter1d
from tornado import gen

from explorepy.stream_processor import TOPICS
from explorepy.tools import (
    HeartRateEstimator,
    find_free_port
)


logger = logging.getLogger(__name__)
ORN_SRATE = 20  # Hz
EXG_VIS_SRATE = 125
WIN_LENGTH = 10  # Seconds
MODE_LIST = ['EEG', 'ECG']
CHAN_LIST = ['Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5', 'Ch6', 'Ch7', 'Ch8']
DEFAULT_SCALE = 10 ** 3  # Volt
BATTERY_N_MOVING_AVERAGE = 60
V_TH = [10, 5 * 10 ** 3]  # Noise threshold for ECG (microVolt)
ORN_LIST = ['accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ', 'magX', 'magY', 'magZ']

SCALE_MENU = {"1 uV": 0, "5 uV": -0.66667, "10 uV": -1, "100 uV": -2, "200 uV": -2.33333, "500 uV": -2.66667,
              "1 mV": -3, "5 mV": -3.66667, "10 mV": -4, "100 mV": -5}
TIME_RANGE_MENU = {"10 s": 10., "5 s": 5., "20 s": 20.}

LINE_COLORS = ['green', '#42C4F7', 'red']
FFT_COLORS = Category20[8]


class Dashboard:
    """Explorepy dashboard class"""

    def __init__(self, explore=None, mode='signal'):
        """
        Args:
            stream_processor (explorepy.stream_processor.StreamProcessor): Stream processor object
        """
        logger.debug(f"Initializing dashboard in {mode} mode")
        self.explore = explore
        self.stream_processor = self.explore.stream_processor
        self.n_chan = self.stream_processor.device_info['adc_mask'].count(1)
        self.y_unit = DEFAULT_SCALE
        self.offsets = np.arange(1, self.n_chan + 1)[:, np.newaxis].astype(float)
        self.chan_key_list = [CHAN_LIST[i]
                              for i, mask in enumerate(reversed(self.stream_processor.device_info['adc_mask'])) if
                              mask == 1]
        self.exg_mode = 'EEG'
        self.rr_estimator = None
        self.win_length = WIN_LENGTH
        self.mode = mode
        self.exg_fs = self.stream_processor.device_info['sampling_rate']
        self._vis_time_offset = None
        self._baseline_corrector = {"MA_length": 1.5 * EXG_VIS_SRATE,
                                    "baseline": 0}

        # Init ExG data source
        exg_temp = np.zeros((self.n_chan, 2))
        exg_temp[:, 0] = self.offsets[:, 0]
        exg_temp[:, 1] = np.nan
        init_data = dict(zip(self.chan_key_list, exg_temp))
        self._exg_source_orig = ColumnDataSource(data=init_data)
        init_data['t'] = np.array([0., 0.])
        self._exg_source_ds = ColumnDataSource(data=init_data)  # Downsampled ExG data for visualization purposes

        # Init ECG R-peak source
        init_data = dict(zip(['r_peak', 't'], [np.array([None], dtype=np.double), np.array([None], dtype=np.double)]))
        self._r_peak_source = ColumnDataSource(data=init_data)

        # Init marker source
        init_data = dict(zip(['marker', 't'], [np.array([None], dtype=np.double), np.array([None], dtype=np.double)]))
        self._marker_source = ColumnDataSource(data=init_data)

        # Init ORN data source
        init_data = dict(zip(ORN_LIST, np.zeros((9, 1))))
        init_data['t'] = [0.]
        self._orn_source = ColumnDataSource(data=init_data)

        # Init table sources
        self._heart_rate_source = ColumnDataSource(data={'heart_rate': ['NA']})
        self._firmware_source = ColumnDataSource(
            data={'firmware_version': [self.stream_processor.device_info['firmware_version']]}
        )
        self._battery_source = ColumnDataSource(data={'battery': ['NA']})
        self.temperature_source = ColumnDataSource(data={'temperature': ['NA']})
        self.light_source = ColumnDataSource(data={'light': ['NA']})
        self.battery_percent_list = []
        self.server = None

        # Init fft data source
        init_data = dict(zip(self.chan_key_list, np.zeros((self.n_chan, 1))))
        init_data['f'] = np.array([0.])
        self.fft_source = ColumnDataSource(data=init_data)

        # Init impedance measurement source
        init_data = {'channel': self.chan_key_list,
                     'impedance': ['NA' for i in range(self.n_chan)],
                     'row': ['1' for i in range(self.n_chan)],
                     'color': ['black' for i in range(self.n_chan)]}
        self.imp_source = ColumnDataSource(data=init_data)

        # Init timer source
        self._timer_source = ColumnDataSource(data={'timer': ['00:00:00']})

    def start_server(self):
        """Start bokeh server"""
        validate(False)
        logger.debug("Starting bokeh server...")
        port_number = find_free_port()
        logger.info("Opening the dashboard on port: %i", port_number)
        self.server = Server({'/': self._init_doc}, num_procs=1, port=port_number)
        self.server.start()

    def start_loop(self):
        """Start io loop and show the dashboard"""
        logger.debug("Starting bokeh io_loop...")
        self.server.io_loop.add_callback(self.server.show, "/")
        try:
            self.server.io_loop.start()
        except KeyboardInterrupt:
            if self.mode == 'signal':
                logger.info("Got Keyboard Interrupt. The program exits ...")
                self.explore.stop_lsl()
                self.explore.stop_recording()
                os._exit(0)
            else:
                logger.info("Got Keyboard Interrupt. The program exits after disabling the impedance mode ...")
                raise KeyboardInterrupt

    def exg_callback(self, packet):
        """
        Update ExG data in the visualization

        Args:
            packet (explorepy.packet.EEG): Received ExG packet

        """
        time_vector, exg = packet.get_data(self.exg_fs)
        if self._vis_time_offset is None:
            self._vis_time_offset = time_vector[0]
        time_vector -= self._vis_time_offset
        self._exg_source_orig.stream(dict(zip(self.chan_key_list, exg)), rollover=int(self.exg_fs * self.win_length))

        if self.mode == 'signal':
            # Downsampling
            exg = exg[:, ::int(self.exg_fs / EXG_VIS_SRATE)]
            time_vector = time_vector[::int(self.exg_fs / EXG_VIS_SRATE)]

            # Baseline correction
            if self.baseline_widget.active:
                samples_avg = exg.mean(axis=1)
                baseline = self._baseline_corrector["baseline"]
                ma_len = self._baseline_corrector["MA_length"]
                if baseline is None:
                    self._baseline_corrector["baseline"] = samples_avg
                else:
                    self._baseline_corrector["baseline"] -= ((baseline - samples_avg) / ma_len * exg.shape[1])
                exg -= self._baseline_corrector["baseline"][:, np.newaxis]
            else:
                self._baseline_corrector["baseline"] = None

            # Update ExG unit
            exg = self.offsets + exg / self.y_unit
            new_data = dict(zip(self.chan_key_list, exg))
            new_data['t'] = time_vector
            self.doc.add_next_tick_callback(partial(self._update_exg, new_data=new_data))

    def orn_callback(self, packet):
        """Update orientation data

        Args:
            packet (explorepy.packet.Orientation): Orientation packet
        """
        if self.tabs.active != 1:
            return
        timestamp, orn_data = packet.get_data()
        if self._vis_time_offset is None:
            self._vis_time_offset = timestamp[0]
        timestamp -= self._vis_time_offset
        new_data = dict(zip(ORN_LIST, np.array(orn_data)[:, np.newaxis]))
        new_data['t'] = timestamp
        self.doc.add_next_tick_callback(partial(self._update_orn, new_data=new_data))

    def info_callback(self, packet):
        """Update device information in the dashboard

        Args:
            packet (explorepy.packet.Environment): Environment/DeviceInfo packet

        """
        new_info = packet.get_data()
        for key in new_info.keys():
            data = {key: new_info[key]}
            if key == 'firmware_version':
                self.doc.add_next_tick_callback(partial(self._update_fw_version, new_data=data))
            elif key == 'battery':
                self.battery_percent_list.append(new_info[key][0])
                if len(self.battery_percent_list) > BATTERY_N_MOVING_AVERAGE:
                    del self.battery_percent_list[0]
                value = int(np.mean(self.battery_percent_list) / 5) * 5
                if value < 1:
                    value = 1
                self.doc.add_next_tick_callback(partial(self._update_battery, new_data={key: [value]}))
            elif key == 'temperature':
                self.doc.add_next_tick_callback(partial(self._update_temperature, new_data=data))
            elif key == 'light':
                data[key] = [int(data[key][0])]
                self.doc.add_next_tick_callback(partial(self._update_light, new_data=data))
            else:
                logger.warning("There is no field named: " + key)

    def marker_callback(self, packet):
        """Update markers
        Args:
            packet (explorepy.packet.EventMarker): Event marker packet
        """
        if self.mode == "impedance":
            return
        timestamp, _ = packet.get_data()
        if self._vis_time_offset is None:
            self._vis_time_offset = timestamp[0]
        timestamp -= self._vis_time_offset
        new_data = dict(zip(['marker', 't', 'code'], [np.array([0.01, self.n_chan + 0.99, None], dtype=np.double),
                                                      np.array([timestamp[0], timestamp[0], None], dtype=np.double)]))
        self.doc.add_next_tick_callback(partial(self._update_marker, new_data=new_data))

    def impedance_callback(self, packet):
        """Update impedances

        Args:
             packet (explorepy.packet.EEG): ExG packet
        """
        if self.mode == "impedance":
            imp = packet.get_impedances() / 2
            color = []
            imp_status = []
            for value in imp:
                if value > 250:
                    color.append("black")
                    imp_status.append("Open")
                elif value > 50:
                    color.append("black")
                    imp_status.append(str(round(value, 0)) + " K\u03A9")
                elif value > 30:
                    color.append("red")
                    imp_status.append(str(round(value, 0)) + " K\u03A9")
                elif value > 20:
                    color.append("orange")
                    imp_status.append(str(round(value, 0)) + " K\u03A9")
                elif value > 10:
                    color.append("yellow")
                    imp_status.append(str(round(value, 0)) + " K\u03A9")
                elif value > 5:
                    imp_status.append(str(round(value, 0)) + " K\u03A9")
                    color.append("green")
                else:
                    color.append("green")
                    imp_status.append("<5K\u03A9")  # As the ADS is not precise in low values.

            data = {"impedance": imp_status,
                    'channel': self.chan_key_list,
                    'row': ['1' for i in range(self.n_chan)],
                    'color': color
                    }
            self.doc.add_next_tick_callback(partial(self._update_imp, new_data=data))
        else:
            raise RuntimeError("Trying to compute impedances while the dashboard is not in Impedance mode!")

    @gen.coroutine
    @without_property_validation
    def _update_exg(self, new_data):
        self._exg_source_ds.stream(new_data, rollover=int(2 * EXG_VIS_SRATE * WIN_LENGTH))

    @gen.coroutine
    @without_property_validation
    def _update_orn(self, new_data):
        self._orn_source.stream(new_data, rollover=int(2 * WIN_LENGTH * ORN_SRATE))

    @gen.coroutine
    @without_property_validation
    def _update_fw_version(self, new_data):
        self._firmware_source.stream(new_data, rollover=1)

    @gen.coroutine
    @without_property_validation
    def _update_battery(self, new_data):
        self._battery_source.stream(new_data, rollover=1)

    @gen.coroutine
    @without_property_validation
    def _update_temperature(self, new_data):
        self.temperature_source.stream(new_data, rollover=1)

    @gen.coroutine
    @without_property_validation
    def _update_light(self, new_data):
        self.light_source.stream(new_data, rollover=1)

    @gen.coroutine
    @without_property_validation
    def _update_marker(self, new_data):
        self._marker_source.stream(new_data=new_data, rollover=100)

    @gen.coroutine
    @without_property_validation
    def _update_imp(self, new_data):
        self.imp_source.stream(new_data, rollover=self.n_chan)

    @gen.coroutine
    @without_property_validation
    def _update_fft(self):
        """ Update spectral frequency analysis plot"""
        # Check if the tab is active and if EEG mode is active
        if (self.tabs.active != 2) or (self.exg_mode != 'EEG'):
            return

        exg_data = np.array([self._exg_source_orig.data[key] for key in self.chan_key_list])

        if exg_data.shape[1] < self.exg_fs * 5:
            return
        fft_content, freq = get_fft(exg_data, self.exg_fs)
        data = dict(zip(self.chan_key_list, fft_content))
        data['f'] = freq
        self.fft_source.data = data

    @gen.coroutine
    @without_property_validation
    def _update_heart_rate(self):
        """Detect R-peaks and update the plot and heart rate"""
        if self.exg_mode == 'EEG':
            self._heart_rate_source.stream({'heart_rate': ['NA']}, rollover=1)
            return
        if CHAN_LIST[0] not in self.chan_key_list:
            logger.warning('Heart rate estimation works only when channel 1 is enabled.')
            return
        if self.rr_estimator is None:
            self.rr_estimator = HeartRateEstimator(fs=self.exg_fs)
            # Init R-peaks plot
            self.exg_plot.circle(x='t', y='r_peak', source=self._r_peak_source,
                                 fill_color="red", size=8)

        ecg_data = (np.array(self._exg_source_ds.data['Ch1'])[-2 * EXG_VIS_SRATE:] - self.offsets[0]) * self.y_unit
        time_vector = np.array(self._exg_source_ds.data['t'])[-2 * EXG_VIS_SRATE:]

        # Check if the peak2peak value is bigger than threshold
        if (np.ptp(ecg_data) < V_TH[0]) or (np.ptp(ecg_data) > V_TH[1]):
            logger.warning("P2P value larger or less than threshold. Cannot compute heart rate!")
            return

        peaks_time, peaks_val = self.rr_estimator.estimate(ecg_data, time_vector)
        peaks_val = (np.array(peaks_val) / self.y_unit) + self.offsets[0]
        if peaks_time:
            data = dict(zip(['r_peak', 't'], [peaks_val, peaks_time]))
            self._r_peak_source.stream(data, rollover=50)

        # Update heart rate cell
        estimated_heart_rate = self.rr_estimator.heart_rate
        data = {'heart_rate': [estimated_heart_rate]}
        self._heart_rate_source.stream(data, rollover=1)

    @gen.coroutine
    @without_property_validation
    def _change_scale(self, attr, old, new):
        """Change y-scale of ExG plot"""
        logger.debug(f"ExG scale has been changed from {old} to {new}")
        new, old = SCALE_MENU[new], SCALE_MENU[old]
        old_unit = 10 ** (-old)
        self.y_unit = 10 ** (-new)

        for chan, value in self._exg_source_ds.data.items():
            if chan in self.chan_key_list:
                temp_offset = self.offsets[self.chan_key_list.index(chan)]
                self._exg_source_ds.data[chan] = (value - temp_offset) * (old_unit / self.y_unit) + temp_offset
        self._r_peak_source.data['r_peak'] = (np.array(self._r_peak_source.data['r_peak']) - self.offsets[0]) * \
                                             (old_unit / self.y_unit) + self.offsets[0]

    @gen.coroutine
    @without_property_validation
    def _change_t_range(self, attr, old, new):
        """Change time range"""
        logger.debug(f"Time scale has been changed from {old} to {new}")
        self._set_t_range(TIME_RANGE_MENU[new])

    @gen.coroutine
    def _change_mode(self, attr, old, new):
        """Set EEG or ECG mode"""
        logger.debug(f"ExG mode has been changed to {new}")
        self.exg_mode = new

    def _init_doc(self, doc):
        self.doc = doc
        self.doc.title = "Explore Dashboard"
        with open(os.path.join(os.path.dirname(__file__), 'templates', 'index.html')) as f:
            index_template = Template(f.read())
        doc.template = index_template
        self.doc.theme = Theme(os.path.join(os.path.dirname(__file__), 'theme.yaml'))
        self._init_plots()
        m_widgetbox = self._init_controls()

        # Create tabs
        if self.mode == "signal":
            exg_tab = Panel(child=self.exg_plot, title="ExG Signal")
            orn_tab = Panel(child=column([self.acc_plot, self.gyro_plot, self.mag_plot], sizing_mode='scale_width'),
                            title="Orientation")
            fft_tab = Panel(child=self.fft_plot, title="Spectral analysis")
            self.tabs = Tabs(tabs=[exg_tab, orn_tab, fft_tab], width=400, sizing_mode='scale_width')
            self.recorder_widget = self._init_recorder()
            self.push2lsl_widget = self._init_push2lsl()
            self.set_marker_widget = self._init_set_marker()
            self.baseline_widget = CheckboxGroup(labels=['Baseline correction'], active=[0])

        elif self.mode == "impedance":
            imp_tab = Panel(child=self.imp_plot, title="Impedance")
            self.tabs = Tabs(tabs=[imp_tab], width=500, sizing_mode='scale_width')
        banner = Div(text=""" <a href="https://www.mentalab.com"><img src=
        "https://images.squarespace-cdn.com/content/5428308ae4b0701411ea8aaf/1505653866447-R24N86G5X1HFZCD7KBWS/
        Mentalab%2C+Name+copy.png?format=1500w&content-type=image%2Fpng" alt="Mentalab"  width="225" height="39">""",
                     width=1500, height=50, css_classes=["banner"], align='center', sizing_mode="stretch_width")
        heading = Div(text=""" """, height=2, sizing_mode="stretch_width")
        if self.mode == 'signal':
            layout = column([heading,
                             banner,
                             row(m_widgetbox,
                                 Spacer(width=10, height=300),
                                 self.tabs,
                                 Spacer(width=10, height=300),
                                 column(Spacer(width=170, height=50), self.baseline_widget, self.recorder_widget,
                                        self.set_marker_widget, self.push2lsl_widget),
                                 Spacer(width=50, height=300)),
                             ],
                            sizing_mode="stretch_both")

        elif self.mode == 'impedance':
            layout = column(banner,
                            Spacer(width=600, height=20),
                            row([m_widgetbox, Spacer(width=25, height=500), self.tabs])
                            )
        self.doc.add_root(layout)
        self.doc.add_periodic_callback(self._update_fft, 2000)
        self.doc.add_periodic_callback(self._update_heart_rate, 2000)
        if self.stream_processor:
            self.stream_processor.subscribe(topic=TOPICS.filtered_ExG, callback=self.exg_callback)
            self.stream_processor.subscribe(topic=TOPICS.raw_orn, callback=self.orn_callback)
            self.stream_processor.subscribe(topic=TOPICS.device_info, callback=self.info_callback)
            self.stream_processor.subscribe(topic=TOPICS.marker, callback=self.marker_callback)
            self.stream_processor.subscribe(topic=TOPICS.env, callback=self.info_callback)
            self.stream_processor.subscribe(topic=TOPICS.imp, callback=self.impedance_callback)

    def _init_plots(self):
        """Initialize all plots in the dashboard"""
        self.exg_plot = figure(y_range=(0.01, self.n_chan + 1 - 0.01), y_axis_label='Voltage', x_axis_label='Time (s)',
                               title="ExG signal",
                               plot_height=250, plot_width=500,
                               y_minor_ticks=int(10),
                               tools=[ResetTool()], active_scroll=None, active_drag=None,
                               active_inspect=None, active_tap=None, sizing_mode="scale_width")

        self.mag_plot = figure(y_axis_label='Mag [mgauss/LSB]', x_axis_label='Time (s)',
                               plot_height=100, plot_width=500,
                               tools=[ResetTool()], active_scroll=None, active_drag=None,
                               active_inspect=None, active_tap=None, sizing_mode="scale_width")
        self.acc_plot = figure(y_axis_label='Acc [mg/LSB]',
                               plot_height=75, plot_width=500,
                               tools=[ResetTool()], active_scroll=None, active_drag=None,
                               active_inspect=None, active_tap=None, sizing_mode="scale_width")
        self.acc_plot.xaxis.visible = False
        self.gyro_plot = figure(y_axis_label='Gyro [mdps/LSB]',
                                plot_height=75, plot_width=500,
                                tools=[ResetTool()], active_scroll=None, active_drag=None,
                                active_inspect=None, active_tap=None, sizing_mode="scale_width")
        self.gyro_plot.xaxis.visible = False

        self.fft_plot = figure(y_axis_label='Amplitude (uV)', x_axis_label='Frequency (Hz)', title="FFT",
                               x_range=(0, 70), plot_height=250, plot_width=500, y_axis_type="log",
                               tools=[BoxZoomTool(), ResetTool()], active_scroll=None, active_drag=None,
                               active_tap=None,
                               sizing_mode="scale_width")

        self.imp_plot = self._init_imp_plot()

        # Set yaxis properties
        self.exg_plot.yaxis.ticker = SingleIntervalTicker(interval=1, num_minor_ticks=0)

        # Initial plot line
        for i in range(self.n_chan):
            self.exg_plot.line(x='t', y=self.chan_key_list[i], source=self._exg_source_ds,
                               line_width=1.0, alpha=.9, line_color="#42C4F7")
            self.fft_plot.line(x='f', y=self.chan_key_list[i], source=self.fft_source,
                               legend_label=self.chan_key_list[i] + " ",
                               line_width=1.5, alpha=.9, line_color=FFT_COLORS[i])
        self.fft_plot.yaxis.axis_label_text_font_style = 'normal'
        self.exg_plot.line(x='t', y='marker', source=self._marker_source,
                           line_width=1, alpha=.8, line_color='#7AB904', line_dash="4 4")

        for i in range(3):
            self.acc_plot.line(x='t', y=ORN_LIST[i], source=self._orn_source, legend_label=ORN_LIST[i] + " ",
                               line_width=1.5, line_color=LINE_COLORS[i], alpha=.9)
            self.gyro_plot.line(x='t', y=ORN_LIST[i + 3], source=self._orn_source, legend_label=ORN_LIST[i + 3] + " ",
                                line_width=1.5, line_color=LINE_COLORS[i], alpha=.9)
            self.mag_plot.line(x='t', y=ORN_LIST[i + 6], source=self._orn_source, legend_label=ORN_LIST[i + 6] + " ",
                               line_width=1.5, line_color=LINE_COLORS[i], alpha=.9)

        # Set x_range
        self.plot_list = [self.exg_plot, self.acc_plot, self.gyro_plot, self.mag_plot]
        self._set_t_range(WIN_LENGTH)

        # Set the formatting of yaxis ticks' labels
        self.exg_plot.yaxis.major_label_overrides = dict(zip(range(1, self.n_chan + 1), self.chan_key_list))
        for plot in self.plot_list:
            plot.toolbar.autohide = True
            plot.yaxis.axis_label_text_font_style = 'normal'
            if len(plot.legend) != 0:
                plot.legend.location = "bottom_left"
                plot.legend.orientation = "horizontal"
                plot.legend.padding = 2

    def _init_imp_plot(self):
        plot = figure(plot_width=600, plot_height=200, x_range=self.chan_key_list[0:self.n_chan],
                      y_range=[str(1)], toolbar_location=None, sizing_mode="scale_width")

        plot.circle(x='channel', y="row", size=50, source=self.imp_source, fill_alpha=0.6, color="color",
                    line_color='color', line_width=2)

        text_props = {"source": self.imp_source, "text_align": "center",
                      "text_color": "white", "text_baseline": "middle", "text_font": "helvetica",
                      "text_font_style": "bold"}

        x = dodge("channel", -0.1, range=plot.x_range)

        plot.text(x=x, y=dodge('row', -.35, range=plot.y_range),
                  text="impedance", **text_props).glyph.text_font_size = "10pt"
        plot.text(x=x, y=dodge('row', -.25, range=plot.y_range), text="channel",
                  **text_props).glyph.text_font_size = "12pt"

        plot.outline_line_color = None
        plot.grid.grid_line_color = None
        plot.axis.axis_line_color = None
        plot.axis.major_tick_line_color = None
        plot.axis.major_label_standoff = 0
        plot.axis.visible = False
        return plot

    def _init_controls(self):
        """Initialize all controls in the dashboard"""
        # EEG/ECG Radio button
        self.mode_control = widgets.Select(title="Signal", value='EEG', options=MODE_LIST, width=170, height=50)
        self.mode_control.on_change('value', self._change_mode)

        self.t_range = widgets.Select(title="Time window", value="10 s", options=list(TIME_RANGE_MENU.keys()),
                                      width=170, height=50)
        self.t_range.on_change('value', self._change_t_range)
        self.y_scale = widgets.Select(title="Y-axis Scale", value="1 mV", options=list(SCALE_MENU.keys()),
                                      width=170, height=50)
        self.y_scale.on_change('value', self._change_scale)

        # Create device info tables
        columns = [widgets.TableColumn(field='heart_rate', title="Heart Rate (bpm)")]
        self.heart_rate = widgets.DataTable(source=self._heart_rate_source, index_position=None, sortable=False,
                                            reorderable=False,
                                            columns=columns, width=170, height=50)

        columns = [widgets.TableColumn(field='firmware_version', title="Firmware Version")]
        self.firmware = widgets.DataTable(source=self._firmware_source, index_position=None, sortable=False,
                                          reorderable=False,
                                          columns=columns, width=170, height=50)

        columns = [widgets.TableColumn(field='battery', title="Battery (%)")]
        self.battery = widgets.DataTable(source=self._battery_source, index_position=None, sortable=False,
                                         reorderable=False,
                                         columns=columns, width=170, height=50)

        columns = [widgets.TableColumn(field='temperature', title="Device temperature (C)")]
        self.temperature = widgets.DataTable(source=self.temperature_source, index_position=None, sortable=False,
                                             reorderable=False, columns=columns, width=170, height=50)

        columns = [widgets.TableColumn(field='light', title="Light (Lux)")]
        self.light = widgets.DataTable(source=self.light_source, index_position=None, sortable=False, reorderable=False,
                                       columns=columns, width=170, height=50)
        if self.mode == 'signal':
            widget_list = [Spacer(width=170, height=30), self.mode_control, self.y_scale, self.t_range, self.heart_rate,
                           self.battery, self.temperature, self.firmware]
        elif self.mode == 'impedance':
            widget_list = [Spacer(width=170, height=40), self.battery, self.temperature, self.firmware]

        widget_box = widgetbox(widget_list, width=175, height=450, sizing_mode='fixed')
        return widget_box

    def _init_recorder(self):
        self.rec_button = Toggle(label=u"\u25CF  Record", button_type="default", active=False,
                                 width=170, height=35)
        self.file_name_widget = TextInput(value="test_file", title="File name:", width=170, height=50)
        self.file_type_widget = RadioGroup(labels=["EDF (BDF+)", "CSV"], active=0, width=170, height=50)

        columns = [widgets.TableColumn(field='timer', title="Record time",
                                       formatter=widgets.StringFormatter(text_align='center'))]
        self.timer = widgets.DataTable(source=self._timer_source, index_position=None, sortable=False,
                                       reorderable=False,
                                       header_row=False, columns=columns,
                                       width=170, height=50, css_classes=["timer_widget"])

        self.rec_button.on_click(self._toggle_rec)
        return column([Spacer(width=170, height=5), self.file_name_widget, self.file_type_widget, self.rec_button,
                      self.timer], width=170, height=200, sizing_mode='fixed')

    def _toggle_rec(self, active):
        logger.debug(f"Pressed record button -> {active}")
        if active:
            self.event_code_input.disabled = False
            self.marker_button.disabled = False
            if self.explore.is_connected:
                self.explore.record_data(file_name=self.file_name_widget.value,
                                         file_type=['edf', 'csv'][self.file_type_widget.active],
                                         do_overwrite=True)
                self.rec_button.label = u"\u25A0  Stop"
                self.rec_start_time = datetime.now()
                self.rec_timer_id = self.doc.add_periodic_callback(self._timer_callback, 1000)
            else:
                self.rec_button.active = False
                self.doc.remove_periodic_callback(self.rec_timer_id)
                self.doc.add_next_tick_callback(partial(self._update_rec_timer, new_data={'timer': '00:00:00'}))
        else:
            self.explore.stop_recording()
            self.rec_button.label = u"\u25CF  Record"
            self.doc.add_next_tick_callback(partial(self._update_rec_timer, new_data={'timer': '00:00:00'}))
            self.doc.remove_periodic_callback(self.rec_timer_id)
            if not self.push2lsl_button.active:
                self.event_code_input.disabled = True
                self.marker_button.disabled = True

    def _timer_callback(self):
        t_delta = (datetime.now() - self.rec_start_time).seconds
        timer_text = ':'.join([str(int(t_delta / 3600)).zfill(2), str(int(t_delta / 60) % 60).zfill(2),
                               str(int(t_delta % 60)).zfill(2)])
        data = {'timer': timer_text}
        self.doc.add_next_tick_callback(partial(self._update_rec_timer, new_data=data))

    def _init_push2lsl(self):
        push2lsl_title = Div(text="""Push to LSL""", width=170, height=10)
        self.push2lsl_button = Toggle(label=u"\u25CF  Start", button_type="default", active=False,
                                      width=170, height=35)
        self.push2lsl_button.on_click(self._toggle_push2lsl)
        return column([Spacer(width=170, height=30), push2lsl_title, self.push2lsl_button],
                      width=170, height=200, sizing_mode='fixed')

    def _toggle_push2lsl(self, active):
        logger.debug(f"Pressed push2lsl button -> {active}")
        if active:
            self.event_code_input.disabled = False
            self.marker_button.disabled = False
            if self.explore.is_connected:
                self.explore.push2lsl()
                self.push2lsl_button.label = u"\u25A0  Stop"
            else:
                self.push2lsl_button.active = False
        else:
            self.explore.stop_lsl()
            self.push2lsl_button.label = u"\u25CF  Start"
            if not self.rec_button.active:
                self.event_code_input.disabled = True
                self.marker_button.disabled = True

    def _init_set_marker(self):
        self.marker_button = Button(label=u"Set", button_type="default", width=80, height=31, disabled=True)
        self.event_code_input = TextInput(value="0", title="Event code:", width=80, disabled=True)
        self.event_code_input.on_change('value', self._check_marker_value)
        self.marker_button.on_click(self._set_marker)
        return column([Spacer(width=170, height=5),
                      row([self.event_code_input,
                          column(Spacer(width=50, height=19), self.marker_button)], height=50, width=170)],
                      width=170, height=50, sizing_mode='fixed'
                      )

    def _set_marker(self):
        code = self.event_code_input.value
        self.stream_processor.set_marker(int(code))

    def _check_marker_value(self, attr, old, new):
        try:
            code = int(self.event_code_input.value)
            if code < 0 or code > 65535:
                raise ValueError('Value must be an integer between 0 and 65535')
        except ValueError:
            self.event_code_input.value = "0<=val<=65535"

    @gen.coroutine
    @without_property_validation
    def _update_rec_timer(self, new_data):
        self._timer_source.stream(new_data, rollover=1)

    def _set_t_range(self, t_length):
        """Change time range of ExG and orientation plots"""
        for plot in self.plot_list:
            self.win_length = int(t_length)
            plot.x_range.follow = "end"
            plot.x_range.follow_interval = t_length
            plot.x_range.range_padding = 0.
            plot.x_range.min_interval = t_length


def get_fft(exg, s_rate):
    """Compute FFT"""
    n_point = 1024
    exg -= exg.mean(axis=1)[:, np.newaxis]
    freq = s_rate * np.arange(int(n_point / 2)) / n_point
    fft_content = np.fft.fft(exg, n=n_point) / n_point
    fft_content = np.abs(fft_content[:, range(int(n_point / 2))])
    fft_content = gaussian_filter1d(fft_content, 1)
    return fft_content[:, 1:], freq[1:]


if __name__ == '__main__':
    from explorepy import Explore
    from explorepy.stream_processor import StreamProcessor

    explore = Explore()
    explore.stream_processor = StreamProcessor()
    explore.stream_processor.device_info = {'firmware_version': '0.0.0',
                                            'adc_mask': [1 for i in range(8)],
                                            'sampling_rate': 250}

    dashboard = Dashboard(explore=explore)
    dashboard.start_server()
    dashboard.start_loop()
