import time
from threading import Thread

from pylsl import proc_dejitter, proc_threadsafe, proc_clocksync

import explorepy


def modify_variable(explore_instance):
    from pylsl import StreamInlet, resolve_stream
    print("looking for a marker stream...")
    streams = resolve_stream('type', 'Markers')

    # create a new inlet to read from the stream
    # adds processing flags for clocksync and thread safety
    # Reference: https://labstreaminglayer.readthedocs.io/info/faqs.html#lsl-local-clock
    inlet = StreamInlet(streams[0], processing_flags=proc_dejitter | proc_threadsafe | proc_clocksync)

    while True:
        sample, timestamp = inlet.pull_sample()
        print("got %s at time %s" % (sample[0], timestamp))
        explore_instance.set_external_marker(timestamp, str(sample[0]), inlet.time_correction())
        time.sleep(1)


def main():
    # Create an Explore object
    explore = explorepy.Explore()
    explore.connect(device_name='Explore_8444')
    explore.record_data(file_name='test_event_gen_8.3_1', file_type='csv', do_overwrite=True, block=False)
    t = Thread(target=modify_variable, args=(explore,))
    t.start()



if __name__ == "__main__":
    main()
