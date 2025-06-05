import argparse
import csv
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

import explorepy
from explorepy.stream_processor import TOPICS


# ----------------------------- Argument Parsing ----------------------------- #
parser = argparse.ArgumentParser(description="Acquire and filter impedance-mode ExG data from an Explore device.")
parser.add_argument("--device-name", required=True, help="Name of the Explore device (e.g., Explore_AAXX)")
args = parser.parse_args()


# ----------------------------- Configuration ----------------------------- #
FS = 250  # Sampling rate in Hz
CHANNEL_LABELS = [f"ch{i}" for i in range(1, 33)]
OUTPUT_FILENAME = "exg_data_imp_mode.csv"
RECORD_SECONDS = 40


# ----------------------------- CSV Setup ----------------------------- #
csv_file = open(OUTPUT_FILENAME, 'w', newline='\n')
csv_writer = csv.writer(csv_file, delimiter=",")
csv_writer.writerow(['Timestamp'] + CHANNEL_LABELS[:8])  # Log only first 8 channels


# ----------------------------- Packet Handlers ----------------------------- #
def handle_exg_packet(packet):
    """Callback to handle incoming ExG data packets."""
    timestamps, signals = packet.get_data(FS)
    data = np.concatenate((np.array(timestamps)[:, np.newaxis].T, np.array(signals)), axis=0)
    np.savetxt(csv_file, np.round(data.T, 4), fmt='%4f', delimiter=',')


def handle_impedance_packet(packet):
    """Callback to handle incoming impedance packets."""
    impedance_values = packet.get_impedances()
    print("Impedance:", impedance_values)


# ----------------------------- Device Initialization ----------------------------- #
device = explorepy.Explore()
device.connect(args.device_name)
device.stream_processor.subscribe(callback=handle_impedance_packet, topic=TOPICS.imp)
device.stream_processor.subscribe(callback=handle_exg_packet, topic=TOPICS.raw_ExG)
device.stream_processor.imp_initialize(notch_freq=50)


# ----------------------------- Data Acquisition Loop ----------------------------- #
for _ in range(RECORD_SECONDS):
    time.sleep(1)


# ----------------------------- Cleanup ----------------------------- #
device.stream_processor.disable_imp()
device.stream_processor.unsubscribe(callback=handle_impedance_packet, topic=TOPICS.imp)
device.stream_processor.unsubscribe(callback=handle_exg_packet, topic=TOPICS.raw_ExG)
csv_file.close()


# ----------------------------- Signal Processing ----------------------------- #
def apply_bandpass_filter(signal_data, low_freq, high_freq, fs, order=3):
    """Apply a Butterworth bandpass filter to the signal."""
    b, a = signal.butter(order, [low_freq / fs, high_freq / fs], btype='bandpass')
    return signal.filtfilt(b, a, signal_data)


def apply_notch_filter(signal_data, fs, freq=50, quality_factor=30.0):
    """Apply a notch filter at the specified frequency."""
    b, a = signal.iirnotch(freq, quality_factor, fs)
    return signal.filtfilt(b, a, signal_data)


# ----------------------------- Load and Filter Data ----------------------------- #
df = pd.read_csv(OUTPUT_FILENAME, delimiter=',', dtype=np.float64)
raw_ch1 = df['ch1']
filtered = apply_notch_filter(raw_ch1, FS, freq=62.5)
filtered = apply_notch_filter(filtered, FS, freq=50)
filtered = apply_bandpass_filter(filtered, low_freq=0.5, high_freq=30, fs=FS)


# ----------------------------- Plot Filtered Signal ----------------------------- #
plt.figure(figsize=(10, 4))
plt.plot(df['Timestamp'], filtered, label='Filtered ch1')
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (µV)")
plt.title("Filtered EEG Signal - Channel 1")
plt.grid(True)
plt.tight_layout()
plt.show()
