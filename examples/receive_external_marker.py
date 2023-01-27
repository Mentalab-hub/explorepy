import time
from threading import Thread

import explorepy


def modify_variable(explore_instance):
    from pylsl import StreamInlet, resolve_stream
    print("looking for a marker stream...")
    streams = resolve_stream('type', 'Markers')

    # create a new inlet to read from the stream
    inlet = StreamInlet(streams[0])

    while True:
        sample, timestamp = inlet.pull_sample()
        print("got %s at time %s" % (sample[0], timestamp))
        explore_instance.set_external_marker(timestamp, str(sample[0]))
        time.sleep(1)


def main():

    # Create an Explore object
    explore = explorepy.Explore()
    explore.connect(device_name='Explore_XXXX')
    explore.record_data(file_name='test_event_gen', file_type='csv', do_overwrite=True, block=False)
    t = Thread(target=modify_variable, args=(explore,))
    t.start()

    n_iteration = 100
    interval = 2

    for i in range(n_iteration):
        time.sleep(interval)

    explore.stop_recording()


if __name__ == "__main__":
    main()
