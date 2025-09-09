"""
Integration test script for explorepy

Examples:
    $ python integration_test_ble.py -n Explore_1438
"""
import argparse
import time
import traceback

import explorepy
from explorepy.settings_manager import SettingsManager


SPS_BY_CHANNELS = {8: 1000, 16: 500, 32: 250}  # starting SPS per device channel count


def resolve_start_sps(dev_name: str) -> int:
    """Infer starting SPS from device channel count."""
    try:
        dev_ch = SettingsManager(dev_name).get_channel_count()
    except Exception:
        raise RuntimeError("Could not read channel count via SettingsManager.")

    if dev_ch not in SPS_BY_CHANNELS:
        raise ValueError(
            f"Unexpected channel count {dev_ch!r}. "
            f"Known: {sorted(SPS_BY_CHANNELS)}. "
            "If this is a new model, update SPS_BY_CHANNELS."
        )
    return SPS_BY_CHANNELS[dev_ch]


def run_once(exp_device: explorepy.Explore, dev_name: str, sps: int, duration: int):
    """Run a single recording at the given sampling rate, with full cleanup."""
    filename = f"{dev_name}_{sps}.csv"
    try:
        exp_device.connect(device_name=dev_name)
        exp_device.set_sampling_rate(sampling_rate=sps)
        # Creates LSL streams (ExG, ORN, marker)
        exp_device.push2lsl()
        exp_device.record_data(
            file_name=filename,
            do_overwrite=True,
            duration=duration,
            block=True,
        )
    finally:
        try:
            exp_device.disconnect()
        except Exception:
            # might throw threading error
            pass


def main():
    parser = argparse.ArgumentParser(
        description="BLE Integration test at different sampling rates"
    )
    parser.add_argument(
        "-n", "--name", dest="name", type=str, required=True,
        help="Bluetooth name or MAC of the device (required)."
    )
    parser.add_argument(
        "-d", "--duration", dest="duration", type=int, default=120,
        help="Duration of each recording in seconds (default: 120)."
    )
    parser.add_argument(
        "--sleep", dest="sleep_s", type=int, default=5,
        help="Pause between runs in seconds (default: 5)."
    )
    args = parser.parse_args()

    exp_device = explorepy.Explore()

    # Determine starting SPS from channel count
    try:
        current_sps = resolve_start_sps(args.name)
    except Exception as e:
        print(f"[setup] Failed to resolve starting SPS: {e}")
        traceback.print_exc()
        return

    # Run at 1000 → 500 → 250 as applicable, halving each time
    while 250 <= current_sps <= 1000:
        try:
            print(f"[run] {args.name}: sampling_rate={current_sps}, duration={args.duration}s")
            run_once(exp_device, args.name, current_sps, args.duration)
            print(f"[ok]  Saved {args.name}_{current_sps}.csv")
        except Exception as e:
            print(f"[error] Run at {current_sps} SPS failed: {e}")
            traceback.print_exc()
        finally:
            # Next SPS
            current_sps //= 2
            if current_sps >= 250:
                time.sleep(args.sleep_s)


if __name__ == "__main__":
    main()
