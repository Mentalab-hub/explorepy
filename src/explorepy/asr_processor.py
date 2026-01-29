import math
import time
from copy import deepcopy

from eegprep import clean_asr, clean_artifacts
import numpy as np
import logging

from eegprep.utils import round_mat
from eegprep.utils.asr import asr_calibrate, asr_process

logger = logging.getLogger(__name__)

def get_asr_state(clean_data, sampling_rate, cutoff=0.5):
    return asr_calibrate(clean_data, sampling_rate, cutoff=cutoff)

def asr_pipeline(data_array, sampling_rate, n_chan, state, step_size=None, window_len=None, max_dims=0.66):
    """This code is mostly taken from the eegprep implementation of clean_asr and adapted to work with a previously
    calculated state."""
    data = np.asarray(data_array, dtype=np.float64)
    srate = float(sampling_rate)
    nbchan = int(n_chan)
    C, S = data.shape

    if window_len is None:
        window_len = max(0.5, 1.5 * nbchan / srate)

    if step_size is None:
        step_size = int(math.floor(srate * window_len / 2))  # Samples

    N_extrap = int(round_mat(window_len / 2 * srate))
    if N_extrap > 0:
        extrap_len = min(N_extrap, S - 1 if S > 1 else 0)
        if extrap_len > 0:
            extrap_indices = np.arange(S - 2, S - extrap_len - 2, -1)
            extrap_part = 2 * data[:, [-1]] - data[:, extrap_indices]
            sig = np.concatenate((data, extrap_part), axis=1)
        else:
            sig = data
    else:
        sig = data

    lookahead_sec = window_len / 2.0
    outdata, _ = asr_process(
        sig,
        srate,
        state,
        window_len=window_len,
        lookahead=lookahead_sec,
        step_size=step_size,
        max_dims=max_dims
    )

    outdata = outdata[:, :S]

    return outdata

class AsrProcessor:
    _min_calibration_length: float = 10.  # in s
    _default_calibration_length: float = 30.  # in s
    _max_calibration_length: float = 120.  # in s

    _min_cutoff: float = 3.0
    _default_cutoff: float = 5.0
    _max_cutoff: float = 20.0

    _min_clean_timer: float = 0.5  # in s
    _default_clean_timer: float = 1.0  # in s
    _max_clean_timer: float = 3.0  # in s

    def __init__(self, stream_proc, in_topic):
        self.stream_processor = stream_proc
        self.in_topic = in_topic
        self.is_initialized = False
        self.cleaned_data_available = False
        self.cleaned_data = None
        self.cleaned_data_ts = None
        info_keys = self.stream_processor.device_info.keys()
        if "sampling_rate" not in info_keys or "firmware_version" not in info_keys:
            logger.error("Sampling rate or firmware version not available from stream processor, "
                         "cannot instantiate AsrProcessor!")
            return
        self.sr = self.stream_processor.device_info["sampling_rate"]
        fw = self.stream_processor.device_info["firmware_version"]
        self.ch_count = 8 if fw[0] == '7' else 16 if fw[0] == '8' else 32

        self.calibration_data_input = np.empty(shape=(self.ch_count, 0))
        self.is_calibrating = False
        self.calibration_data_available = False
        self.is_cleaning = False

        self.asr_packet_count = 0
        self.calib_started_at: float = -1.0
        self.calibration_length: float = self._default_calibration_length  # in s

        self.last_clean_at: float = -1.0  # the last time the to_clean buffer was cleaned
        self.last_cleaned_timestamp = 0.0
        self._refresh_window: float = self._default_clean_timer  # how long to wait between running asr, in s

        self._cutoff = self._default_cutoff
        self._state = None

        self.to_clean_buffer_length = 5.  # in s
        self.instantiate_buffers()
        self.is_initialized = True

    @property
    def cutoff(self):
        return self._cutoff

    @cutoff.setter
    def cutoff(self, new_cutoff):
        self._cutoff = new_cutoff
        self._state = get_asr_state(self.calibration_data_input, self.sr, self._cutoff)

    @property
    def refresh_window(self):
        return self._refresh_window

    @refresh_window.setter
    def refresh_window(self, new_window):
        self._refresh_window = new_window

    def instantiate_buffers(self):
        self.to_clean = np.zeros(shape=(self.ch_count, int(self.sr * self.to_clean_buffer_length)))
        self.to_clean_ts = np.zeros(shape=(1, int(self.sr * self.to_clean_buffer_length)))

    def update_device_data(self, ch_count: int, sr: float):
        self.ch_count = ch_count
        self.sr = sr

        self.calibration_data_input = np.empty(shape=(self.ch_count, 0))
        self.instantiate_buffers()

    def clear_calibration_data(self):
        self.calibration_data_input = np.empty(shape=(self.ch_count, 0))

    def clear_data_buffer(self):
        self.instantiate_buffers()

    def on_calib_data_received(self, packet):
        if (self.calib_started_at <= 0.0 or
            not self._min_calibration_length <= self.calibration_length <= self._max_calibration_length):
            raise ValueError(
                "Error writing calibration packet, timer has not been set correctly or calibration length is invalid!")
        if (time.time() - self.calib_started_at) > self.calibration_length:
            self.calibration_data_available = True
            self.stop_calibration()
            return
        self.calibration_data_input = np.append(self.calibration_data_input, packet.get_data()[1], axis=1)

    def on_unclean_data_received(self, packet):
        # TODO investigate potentially swapped channels (??)
        # TODO investigate odd spikes in cleaned data
        if self.last_clean_at <= 0.0:
            self.last_clean_at = time.time()
        if not self.calibration_data_available:
            logger.warning("Attempting to clean data with no calibration available - returning...")
        new_data = np.array(packet.get_data()[1])
        new_ts = np.array(packet.get_data()[0])
        # arr[:, :new_data.shape[1]] = new_data
        self.to_clean[:, :new_data.shape[1]] = new_data
        self.to_clean = np.roll(self.to_clean, -new_data.shape[1])
        self.to_clean_ts[0, :new_ts.shape[0]] = new_ts
        self.to_clean_ts = np.roll(self.to_clean_ts, -new_ts.shape[0])

        if time.time() - self.last_clean_at >= self._refresh_window:
            self.clean_data()
            self.last_clean_at = -1.0

    def clear_cleaned_data(self):
        self.cleaned_data_available = False
        self.cleaned_data = {}

    def clean_data(self):
        if self._state is None:
            logger.warning("Requested cleaning data with ASR but internal calibration state is None!")
            return
        if self.to_clean_ts[0][0] <= 1.0:
            return
        try:
            ret = asr_pipeline(self.to_clean, self.sr, self.ch_count, self._state)
            self.cleaned_data_available = True
            idx = np.searchsorted(self.to_clean_ts[0], self.last_cleaned_timestamp)
            self.cleaned_data = ret[:, idx:]
            self.cleaned_data_ts = self.to_clean_ts.copy()[0, idx:]

            self.last_cleaned_timestamp = self.to_clean_ts[0][-1]
        except ValueError:
            logger.error("Could not get ASR from input window!")

    def start_cleaning(self, clean_timer: float = None):
        if clean_timer is None:
            self._refresh_window = self._default_clean_timer
        elif self._min_clean_timer <= clean_timer <= self._max_clean_timer:
            self._refresh_window = clean_timer
        else:
            logger.error(f"Passed refresh timer for ASR of {clean_timer} is not within accepted range of "
                         f"[{self._min_clean_timer},{self._max_clean_timer}]")
        logger.info(f"Starting cleaning with ASR (refresh every {self._refresh_window}s)...")
        self.is_cleaning = True
        self.stream_processor.subscribe(self.on_unclean_data_received, topic=self.in_topic)

    def stop_cleaning(self):
        logger.info("Stopping cleaning with ASR.")
        self.is_cleaning = False
        self.stream_processor.unsubscribe(self.on_unclean_data_received, topic=self.in_topic)

    def start_calibration(self, calib_length: float=-1.0):
        self.is_calibrating = True
        if self._min_calibration_length <= calib_length <= self._max_calibration_length:
            self.calibration_length = calib_length
        else:
            logger.error(f"Passed refresh timer for ASR of {calib_length} is not within accepted range of "
                         f"[{self._min_calibration_length},{self._max_calibration_length}]")
        logger.info(f"Starting ASR calibration for {self.calibration_length}s...")
        self.calib_started_at = time.time()
        self.stream_processor.subscribe(self.on_calib_data_received, topic=self.in_topic)

    def stop_calibration(self):
        logger.info(f"Stopping ASR calibration.")
        # TODO potentially run clean_windows on calibration data as that may still contain artifacts...
        self.stream_processor.unsubscribe(self.on_calib_data_received, topic=self.in_topic)
        self.is_calibrating = False
        self.calib_started_at = -1.0
        self.calibration_length = self._default_calibration_length
        self._state = get_asr_state(self.calibration_data_input, self.sr, self.cutoff)

    def set_cutoff(self, new_cutoff: float):
        if self._min_cutoff <= new_cutoff <= self._max_cutoff:
            self._cutoff = new_cutoff
            self._state = get_asr_state(self.calibration_data_input, self.sr, self.cutoff)
        else:
            logger.error(f"Passed cutoff for ASR of {new_cutoff} is not within accepted range of "
                         f"[{self._min_cutoff},{self._max_cutoff}]")
