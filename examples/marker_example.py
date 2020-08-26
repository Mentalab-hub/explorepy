import explorepy
import argparse
import threading


def main():
    parser = argparse.ArgumentParser(description="Example code for marker generation")
    parser.add_argument("-n", "--name", dest="name", type=str, help="Name of the device.")
    args = parser.parse_args()

    # Create an Explore object
    exp_device = explorepy.Explore()

    def marker_gen():
        exp_device.set_marker(code=12)
        threading.Timer(5, marker_gen).start()

    exp_device.connect(device_name=args.name)
    marker_gen()
    exp_device.record_data(file_name='test_event_gen', duration=25, file_type='edf', do_overwrite=True, block=True)
    print("Press Ctrl+c to exit.")


if __name__ == "__main__":
    main()
