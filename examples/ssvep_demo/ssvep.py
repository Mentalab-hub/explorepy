# -*- coding: utf-8 -*-
"""
SSVEP experiment module
"""
import time
from threading import Lock
from psychopy import visual, event
import numpy as np
from analysis import CCAAnalysis

SCORE_TH = .1


class CheckerBoard:
    """Flickering radial checkerboard stimulation"""

    def __init__(self, window, size, position, n_frame, log_time=False):
        """
        Args:
            window (psychopy.visual.window): Psychopy window
            size (int): Size of the checkerboard stimulation
            position (tuple): Position of stimulation on the screen
            n_frame (int): Number of frames for the stim to flicker (frequency = monitor_refresh_rate/n_frame)
            log_time (bool): Whether to log toggle times
        """
        self._window = window
        self._fr_rate = n_frame
        self._fr_counter = n_frame
        pattern = np.ones((4, 4))
        pattern[::2, ::2] *= -1
        pattern[1::2, 1::2] *= -1
        self._stim1 = visual.RadialStim(win=self._window, tex=pattern, pos=position,
                                        size=size, radialCycles=1, texRes=256, opacity=1)
        self._stim2 = visual.RadialStim(win=self._window, tex=pattern*-1, pos=position,
                                        size=size, radialCycles=1, texRes=256, opacity=1)
        self._toggle_flag = False
        self.log_time = log_time
        self.toggle_times = []

    def draw(self):
        """Draw stimulation"""
        if self._fr_counter == 0:
            if self.log_time:
                self.toggle_times.append(time.time())
            self._fr_counter = self._fr_rate
            if self._toggle_flag:
                self._stim1.draw()
            else:
                self._stim2.draw()
            self._toggle_flag = not self._toggle_flag
        self._fr_counter -= 1

    def get_statistics(self):
        """Get stimulation toggle statistics
        Returns:
            mean and standard deviation of the stimulation time length (in case log_time is False returns None)
        """
        assert self.log_time, "Time logging has not been activated for this checkerboard."
        if self.log_time:
            diff_t = np.diff(np.array(self.toggle_times))
            return diff_t.mean(), diff_t.std()

class SSVEPRealTime:
    """Steady State Visual Evoked Potential (SSVEP) Experiment

    This class implements a simple SSVEP experiment in which flickering targets will be shown on the screen. During
    experiment, EEG data is received continuously in the buffer of this class and data will be analysed by CCA
    algorithm to predict the target which the subject has focused on. EEG data is analysed based on sliding windows with
    the length of `signal_len` and with overlap times of `overlap`"""
    def __init__(self, frame_rates, positions, labels, signal_len, eeg_s_rate, overlap=.25, screen_refresh_rate=60):
        """
        Args:
            frame_rates (list): List of number of frames in which each target is flickering (one number for each target)
            positions (list): List of target positions in the screen
            signal_len (float): EEG signal length (in seconds) to be analysed
            eeg_s_rate (int): Sampling rate of EEG signal
            overlap (float): Time overlap between two consecutive data chunk
            screen_refresh_rate (int): Refresh rate of your screen
        """
        self._fr_rates = frame_rates
        self._freqs = [screen_refresh_rate / fr_no for fr_no in self._fr_rates]
        print("Target frequencies: ", self._freqs)
        self.targets = []
        self._positions = positions
        self._labels = labels
        self.win = None
        self._data_buff = np.array([])
        self.chunk_len = signal_len
        self.eeg_s_rate = eeg_s_rate
        self.overlap = overlap
        self.cca = CCAAnalysis(freqs=self._freqs, win_len=self.chunk_len, s_rate=self.eeg_s_rate, n_harmonics=2)
        self.lock = Lock()
        self._prediction_text = []
        self._predicted_ind = None

    def _init_vis(self):
        self._data_buff = np.array([])
        self.win = visual.Window([800, 600], monitor="testMonitor",
                                 fullscr=True, screen=1, units="norm", color=[0.1, 0.1, 0.1])
        self.win.recordFrameIntervals = True
        stim_size = (.6 * self.win.size[1]/self.win.size[0], .6)
        for fr_no, pos, freq, label in zip(self._fr_rates, self._positions, self._freqs, self._labels):
            self.targets.append(CheckerBoard(window=self.win,
                                             size=stim_size,
                                             n_frame=fr_no,
                                             position=pos,
                                             log_time=True))
            self._prediction_text.append(visual.TextStim(win=self.win, pos=[0, 0], text=label,
                                                         color=(-1, -1, -1), height=.15,
                                                         colorSpace='rgb', bold=True))

    def run(self, duration):
        """Run the experiment
        Args:
            duration (float): Duration of the experiment
        """
        self._init_vis()

        start_time = time.time()
        while time.time() - start_time < duration:
            self.win.flip()
            for stim in self.targets:
                stim.draw()
            if self._predicted_ind is not None:
                self._prediction_text[self._predicted_ind].draw()
            self._analyze_data()
        self.win.close()

    def update_buffer(self, packet):
        """Update EEG buffer of the experiment

        Args:
            packet (explorepy.packet.EEG): EEG packet

        """
        timestamp, eeg = packet.get_data()
        if not len(self._data_buff):
            self._data_buff = eeg.T
        else:
            self._data_buff = np.concatenate((self._data_buff, eeg.T), axis=0)

    def _analyze_data(self):
        """Analyse data

        This function checks if there is enough data in the buffer and applies CCA on the data. If all scores are less
        than the threshold `SCORE_TH`, no target is chosen, otherwise the target with maximum score will be chosen as
        the prediction.
        """
        if len(self._data_buff) > 0:
            if self._data_buff.shape[0] > self.chunk_len * self.eeg_s_rate:
                with self.lock:
                    scores = self.cca.apply_cca(self._data_buff[:self.chunk_len * self.eeg_s_rate, :])
                    self._data_buff = self._data_buff[:int(self.overlap * self.eeg_s_rate), :]
                print(scores)
                if not all(val < SCORE_TH for val in scores):
                    self._predicted_ind = np.argmax(scores)
                    print(self._predicted_ind)
                else:
                    self._predicted_ind = None

    def show_statistics(self):
        """Show statistics of frame length and frequencies of the targets"""
        for stim in self.targets:
            avg, std = stim.get_statistics()
            print('frequency: {},  std: {}'.format(1 / avg, std))
        frame_intervals = self.win.frameIntervals
        import matplotlib.pyplot as plt
        plt.plot(frame_intervals)
        plt.xlabel("Frame number")
        plt.ylabel("Frame interval (s)")
        plt.show()


class SSVEPExperiment:
    """Steady State Visual Evoked Potential (SSVEP) Experiment

    This class implements a simple SSVEP experiment. Some flickering stimuli will be shown on the screen and the subject
    will be asked to focus on the specified target. The EEG signal along with the event markers are recorded during the
    experiment.
    """

    def __init__(self, frame_rates, positions, hints, marker_callback, trial_len=4, trials_per_block=5, n_blocks=10,
                 screen_refresh_rate=60):
        """
        Args:
            frame_rates (list): List of number of frames in which each target is flickering (one number for each target)
            positions (list): List of target positions in the screen
            hints (list): List of hints for each target.
            trial_len (float): Trial length in seconds
            trials_per_block (int): Number of trials per block
            n_blocks (int): Number of blocks for the whole experiment
            screen_refresh_rate (int): Refresh rate of your screen
        """
        self._fr_rates = frame_rates
        self._freqs = [screen_refresh_rate / fr_no for fr_no in self._fr_rates]
        print("Target frequencies: ", self._freqs)
        self.targets = []
        self._positions = positions
        self._hints = hints
        self._trial_len = trial_len
        self._n_blocks = n_blocks
        self._trials_per_block = trials_per_block
        self._marker_callback = marker_callback
        self.win = None
        self._hint_stim = []
        self.lock = Lock()

    def _init_vis(self):
        self.win = visual.Window([800, 600], monitor="testMonitor",
                                 fullscr=True, screen=1, units="norm", color=[0.1, 0.1, 0.1])
        self.win.recordFrameIntervals = True
        stim_size = (.6 * self.win.size[1] / self.win.size[0], .6)
        for i in range(len(self._fr_rates)):
            self.targets.append(CheckerBoard(window=self.win,
                                             size=stim_size,
                                             n_frame=self._fr_rates[i],
                                             position=self._positions[i],
                                             log_time=True))
            self._hint_stim.append(visual.TextStim(win=self.win, pos=[0, 0], text=self._hints[i],
                                                   color=(-1, -1, -1), height=.15,
                                                   colorSpace='rgb', bold=True))
        self._block_start_stim = visual.TextStim(win=self.win, pos=[0, 0], text="Press space to continue",
                                                 color=(-1, -1, -1), height=.2,
                                                 colorSpace='rgb', bold=True)

    def run(self):
        """Run the experiment
        """
        self._init_vis()

        for block_idx in range(self._n_blocks):

            self._block_start_stim.draw()
            self.win.flip()
            event.waitKeys(keyList=['space'])
            for trial_idx in range(self._trials_per_block):
                stim_idx = np.random.randint(0, len(self._fr_rates))

                self._hint_stim[stim_idx].draw()
                self.win.flip()
                time.sleep(2)
                self._marker_callback(stim_idx)
                start_time = time.time()
                while time.time() - start_time < self._trial_len:
                    self.win.flip()
                    for stim in self.targets:
                        stim.draw()
                self._marker_callback(10)
                self.win.flip()
                self.win.flip()
                time.sleep(2)
        self.win.close()

    def show_statistics(self):
        """Show statistics of frame length and frequencies of the targets"""
        for stim in self.targets:
            avg, std = stim.get_statistics()
            print('frequency: {},  std: {}'.format(1 / avg, std))
        frame_intervals = self.win.frameIntervals
        import matplotlib.pyplot as plt
        plt.plot(frame_intervals)
        plt.xlabel("Frame number")
        plt.ylabel("Frame interval (s)")
        plt.show()
