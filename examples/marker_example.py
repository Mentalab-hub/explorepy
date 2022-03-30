import time
import explorepy
import argparse


def main():
    parser = argparse.ArgumentParser(description="Example code for marker generation")
    parser.add_argument("-n", "--name", dest="name", type=str, help="Name of the device.")
    args = parser.parse_args()

    # Create an Explore object
    explore = explorepy.Explore()
    explore.connect(device_name=args.name)
    explore.record_data(file_name='test_event_gen', file_type='csv', do_overwrite=True, block=False)

    n_markers = 20
    interval = 2

    for i in range(n_markers):
        explore.set_marker(code=i)
        time.sleep(interval)

    explore.stop_recording()


if __name__ == "__main__":
    main()
