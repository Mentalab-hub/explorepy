import numpy as np

from bokeh.layouts import column
from bokeh.models import ColumnDataSource, Slider
from bokeh.plotting import figure
from bokeh.server.server import Server
from bokeh.themes import Theme

from bokeh.sampledata.sea_surface_temperature import sea_surface_temperature


def modify_doc(doc):
    srate = 250
    win_length = 10
    t = np.linspace(0, win_length, srate*win_length)
    y = np.random.rand(srate*win_length)
    plot = figure(y_range=(-1, 1), y_axis_label='Voltage (v)', title="EEG signal", plot_height=600, plot_width=900)
    plot.line(t, y, legend='Channel 1', line_width=4)

    def callback(attr, old, new):
        pass

    # slider = Slider(start=0, end=30, value=0, step=1, title="Smoothing by N Days")
    # slider.on_change('value', callback)

    # doc.add_root(column(slider, plot))
    doc.add_root(column(plot))
    doc.theme = Theme(filename="theme.yaml")


# Setting num_procs here means we can't touch the IOLoop before now, we must
# let Server handle that. If you need to explicitly handle IOLoops then you
# will need to use the lower level BaseServer class.
server = Server({'/': modify_doc}, num_procs=1)
server.start()

if __name__ == '__main__':
    print('Opening Bokeh application on http://localhost:5006/')

    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()
