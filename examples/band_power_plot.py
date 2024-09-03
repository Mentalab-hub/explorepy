import time
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

import explorepy
from explorepy.packet import EEG
from explorepy.stream_processor import TOPICS

rows, cols = 8, 1024
data_buf = deque([deque(maxlen=cols) for _ in range(rows)])
for row in data_buf:
    row.extend(range(1024))


def get_data_buf():
    output = np.array([[0] * 1024 for _ in range(rows)])
    for i in range(len(output)):
        output[i] = np.array(data_buf[i])
    return output


def on_exg_received(packet: EEG):
    _, data = packet.get_data()
    for r in range(len(data)):
        for column in range(len(data[r])):
            data_buf[r].append(data[r][column])


exp_device = explorepy.Explore()
# Subscribe your function to the stream publisher

exp_device.connect(device_name="Explore_AAAK")
exp_device.stream_processor.subscribe(callback=on_exg_received, topic=TOPICS.raw_ExG)

# Define the frequency bands
bands = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30),
    'Gamma': (30, 50)
}

# FFT and signal parameters
sampling_rate = 250  # in Hz
num_channels = 8
num_samples = 1024

# Set up the figure and axes for subplots
fig, axs = plt.subplots(4, 2, figsize=(10, 15), sharex=True)
fig.tight_layout(pad=4.0)  # Adjust layout padding

# Flatten the 2D array of axes for easier iteration
axs = axs.flatten()

# Initialize bar plots for each subplot
bars = []
for ax in axs:
    bars.append(ax.bar(bands.keys(), [0] * len(bands), color='skyblue'))
    ax.set_ylim(0, 1)  # Set an initial y-limit, can adjust based on your data
    ax.set_xlabel('Frequency Band')
    ax.set_ylabel('Power')


def update(frame):
    eeg_signals = get_data_buf()  # Generate data for all channels

    for idx, (axis, bar) in enumerate(zip(axs, bars)):
        eeg_signal = eeg_signals[idx]

        # Perform FFT
        fft_values = np.fft.fft(eeg_signal)
        fft_frequencies = np.fft.fftfreq(num_samples, 1 / sampling_rate)
        fft_magnitude = np.abs(fft_values) ** 2  # Power spectrum

        # Only consider positive frequencies
        positive_frequencies = fft_frequencies[:num_samples // 2]
        positive_magnitude = fft_magnitude[:num_samples // 2]

        # Calculate band powers
        band_powers = []
        for band, (low, high) in bands.items():
            indices = np.where((positive_frequencies >= low) & (positive_frequencies <= high))
            band_power = np.sum(positive_magnitude[indices])
            band_powers.append(band_power)

        # Update the bar heights
        for b, power in zip(bar, band_powers):
            b.set_height(power)

        # Optionally adjust y-axis limits based on data
        axis.set_ylim(0, max(band_powers) * 1.1)
        axis.set_title(f'Channel {idx + 1}')

    return [b for bar in bars for b in bar]  # Flatten the list of bars


# Create the animation object
ani = FuncAnimation(fig, update, interval=500, blit=True)  # Update every 500 ms (0.5 seconds)

# Display the plot
plt.show()

while True:
    time.sleep(1)
