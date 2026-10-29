import os

import mne
import numpy as np
import pandas as pd
import pyedflib
import pytest

from explorepy.packet import EEG
from explorepy.tools import (
    TIMESTAMP_SPAN,
    create_exg_recorder,
    generate_eeglab_dataset
)


FS = 250
N_CHANNELS = 8
ADC_MASK = [1] * N_CHANNELS


class FakeEEG(EEG):
    def __init__(self, timestamp, data):
        self.timestamp = timestamp
        self.data = data

    def _convert(self, bin_data):
        pass


def make_packets(timestamps, samples_per_packet=1):
    rng = np.random.default_rng(0)
    return [FakeEEG(ts, rng.normal(0, 20, (N_CHANNELS, samples_per_packet))) for ts in timestamps]


def write_bdf(folder, timestamps):
    recorder = create_exg_recorder(os.path.join(folder, "rec_ExG"), "edf", ADC_MASK, FS, do_overwrite=True)
    for packet in make_packets(timestamps):
        recorder.write_data(packet)
    recorder.stop()
    return os.path.join(folder, "rec_ExG.bdf")


def device_timestamps(start, seconds=3):
    jitter = np.random.default_rng(1).normal(0, 2e-5, seconds * FS)
    return np.round(start + np.arange(seconds * FS) / FS + jitter, 5)


def test_batch_csv_keeps_packet_timestamps(tmp_path):
    samples_per_packet = 4
    starts = 100.0 + np.arange(20) * samples_per_packet / FS
    starts = np.delete(starts, 5)
    recorder = create_exg_recorder(str(tmp_path / "rec_ExG"), "csv", ADC_MASK, FS, do_overwrite=True,
                                   batch_mode=True)
    recorder.write_data(make_packets(starts, samples_per_packet))
    recorder.stop()
    written = pd.read_csv(tmp_path / "rec_ExG.csv")["TimeStamp"].to_numpy()
    expected = np.concatenate([s + np.arange(samples_per_packet) / FS for s in starts])
    np.testing.assert_allclose(written, expected, atol=1e-9)


@pytest.mark.parametrize("start", [12.5, 13885.46803, 30000.12345])
def test_bdf_timestamp_channels_add_up(tmp_path, start):
    ts = device_timestamps(start)
    reader = pyedflib.EdfReader(write_bdf(str(tmp_path), ts))
    assert reader.getSignalLabels()[:3] == ["TimeStamp", "TimeStampFine", "ch1"]
    combined = reader.readSignal(0) + reader.readSignal(1)
    reader.close()
    np.testing.assert_allclose(combined, ts, atol=1e-6)


def test_bdf_timestamp_range_starts_at_recording(tmp_path):
    start = 30000.12345
    reader = pyedflib.EdfReader(write_bdf(str(tmp_path), device_timestamps(start)))
    assert reader.getPhysicalMinimum(0) == np.floor(start)
    assert reader.getPhysicalMaximum(0) == np.floor(start) + TIMESTAMP_SPAN
    reader.close()


def test_eeglab_export_drops_timestamp_channels(tmp_path):
    bdf = write_bdf(str(tmp_path), device_timestamps(13885.46803))
    out = str(tmp_path / "rec.set")
    generate_eeglab_dataset(bdf, out)
    raw = mne.io.read_raw_eeglab(out, verbose=False)
    assert raw.ch_names == [f"ch{i}" for i in range(1, N_CHANNELS + 1)]
