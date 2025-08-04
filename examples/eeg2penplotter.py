import argparse
import os.path
import time

import numpy as np
import serial
from mne.time_frequency import psd_array_multitaper
from numpy._typing import NDArray
from scipy.integrate import simpson
from scipy.signal import (
    periodogram,
    welch
)

import explorepy
from explorepy.stream_processor import TOPICS

_VISPY_AVAILABLE = False

try:
    import vispy
    from vispy import gloo, app
    _VISPY_AVAILABLE = True
except ModuleNotFoundError:
    print("Vispy is not installed, drawing on screen will not be available.")


class Coordinate:
    """Helper class for 2D coordinate manipulation"""
    def __init__(self, x: float, y: float):
        self._coord = np.array([x, y, 1.], dtype=np.float64)

    def as_tuple(self) -> tuple[float, float]:
        return float(self._coord[0]), float(self._coord[1])

    def translate(self, x: float, y: float, in_place: bool = False):
        """Translates this coordinate by given x and y, updates itself if in_place is True and returns the resulting
        coordinate."""
        t_mat = np.array([[1., 0., x],
                          [0., 1., y],
                          [0., 0., 1.]])
        ret = np.matmul(t_mat, self._coord)
        if in_place:
            self._coord = ret
        return Coordinate(ret[0], ret[1])

    def rotate(self, angle: float, in_place: bool = False):
        """Rotates this coordinate by a given angle, updates itself if in_place is True and returns the resulting
        coordinate."""
        angle = np.deg2rad(angle)
        r_mat = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                          [np.sin(angle), np.cos(angle), 0.0],
                          [0.0, 0.0, 1.0]])
        ret = np.matmul(r_mat, self._coord)
        if in_place:
            self._coord = ret
        return Coordinate(ret[0], ret[1])

    def scale(self, x: float, y: float, in_place: bool = False):
        """Scales this coordinate by a given x and y factor, updates itself if in_place is True and returns the
        resulting coordinate."""
        s_mat = np.array([[x, 0.0, 0.0],
                          [0.0, y, 0.0],
                          [0.0, 0.0, 1.0]])
        ret = np.matmul(s_mat, self._coord)
        if in_place:
            self._coord = ret
        return Coordinate(ret[0], ret[1])

    def to_polar(self, in_place=False):
        x = self._coord[0]
        y = self._coord[1]
        r = np.sqrt(x**2 + y**2)
        angle = np.arctan2(y, x)
        angle = np.rad2deg(angle)
        if in_place:
            self._coord[0] = angle
            self._coord[1] = -r
        return angle, -r  # plotter takes X=angle, Y=radius

    def from_polar(self):
        phi = self._coord[0]
        phi = np.deg2rad(phi)
        r = -self._coord[1]
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        return x, y

    def length(self):
        """Returns the length of this coordinate if interpreted as a vector"""
        return np.sqrt(self._coord[0]**2 + self._coord[1]**2)  # drop z == 1 since we're in the xy-plane

    def copy(self):
        """Returns a new coordinate with the same values as the given one"""
        return Coordinate(self._coord[0], self._coord[1])

    def __getitem__(self, item):
        if item < 0 or item > 2:
            raise ValueError(f"Can't access index {item}, valid indices for Coordinates are 0 (x) and 1 (y)")
        return float(self._coord[item])

    def __add__(self, other):
        if type(other) is not Coordinate:
            raise ValueError(f"Addition to Coordinate not defined for type {type(other)}")
        x = self.as_tuple()
        y = other.as_tuple()
        return Coordinate(x[0] + y[0], x[1] + y[1])

    def __sub__(self, other):
        if type(other) is not Coordinate:
            raise ValueError(f"Subtraction from Coordinate not defined for type {type(other)}")
        x = self.as_tuple()
        y = other.as_tuple()
        return Coordinate(x[0] - y[0], x[1] - y[1])

    def __mul__(self, other):
        if type(other) not in [int, float]:
            raise ValueError("Coordinate multiplication is currently only implemented for uniform scaling with int or "
                             "float")
        x = self.as_tuple()
        return Coordinate(x[0] * other, x[1] * other)

    def __rmul__(self, other):
        return self * other

    def __str__(self):
        return f"({self._coord[0]}, {self._coord[1]})"


class BandpowerCalculator:
    """Helper class that holds necessary data to calculate the bandpower(s) on a given signal"""
    def __init__(self, fs: float = 250.0):
        self.fs = fs
        self.bands = {
            'Delta': (0.5, 4),
            'Theta': (4, 8),
            'Alpha': (8, 13),
            'Beta': (13, 30),
            'Gamma': (30, 50)
        }

    def compute(self, data: NDArray[np.float64], method: str = "welch", relative: bool = True) -> dict:
        """
        Computes the bandpowers of a given signal with the provided method

        Args:
            data (NDArray[np.float64]): The input signal
            method (str): The method to use for calculating the bandpower, needs to one of [periodogram, welch,
            multitaper]
            relative (bool): Whether to calculate the bandpower relative to the other bandpowers
        """
        bandpowers = {}
        for name, band in self.bands.items():
            bp = self._bandpower(data, self.fs, method, band, relative)  # array of channels
            bandpowers[name] = np.mean(bp)
        return bandpowers

    def _bandpower(self, data: NDArray[np.float64], fs: float, method: str, band: str, relative: bool):
        """
        Computes a specific bandpower of a given signal with the provided method

        Args:
            data (NDArray[np.float64]): The input signal
            fs (float): The frequency of the input signal
            method (str): The method to use for calculating the bandpower, needs to one of [periodogram, welch,
            multitaper]
            band (str): The band to calculate the power on (Alpha, Gamma, Beta etc.)
            relative (bool): Whether to calculate the bandpower relative to the other bandpowers
        """
        if method == "periodogram":
            freqs, psd = periodogram(data, fs, window='hann')
        elif method == "welch":
            freqs, psd = welch(data, fs, window='hann', nperseg=500, noverlap=250)
        elif method == "multitaper":
            psd, freqs = psd_array_multitaper(data, fs, verbose="ERROR")
        else:
            raise ValueError(f"Unknown method: {method}")

        freq_res = freqs[1] - freqs[0]
        idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
        if psd.ndim == 1:
            bp = simpson(psd[idx_band], dx=freq_res)
        else:
            bp = simpson(psd[:, idx_band], dx=freq_res)

        if relative:
            idx_total = np.logical_and(freqs >= 0.5, freqs <= 50.0)
            if psd.ndim == 1:
                total = simpson(psd[idx_total], dx=freq_res)
            else:
                total = simpson(psd[:, idx_total], dx=freq_res)
            with np.errstate(divide='ignore', invalid='ignore'):
                bp = np.divide(bp, total, out=np.zeros_like(bp), where=total != 0)

        return bp


def multiply_line_with_rect_wave(start: Coordinate, stop: Coordinate, amplitude: float):
    diff = stop - start
    mid = start + 0.5 * diff

    # get an orthogonal vector to our start -> stop vector
    orth = Coordinate(-diff[1], diff[0])
    orth_length = orth.length()
    orth.scale(1. / orth_length, 1. / orth_length, in_place=True)  # normalize orthogonal vector

    c1 = orth.scale(amplitude, amplitude)  # scale orthogonal vector according to amplitude
    c1.translate(start[0], start[1], in_place=True)  # add orthogonal vector to start coordinate

    c2 = orth.scale(amplitude, amplitude)
    c2.translate(mid[0], mid[1], in_place=True)

    c3 = orth.scale(-amplitude, -amplitude)
    c3.translate(mid[0], mid[1], in_place=True)

    c4 = orth.scale(-amplitude, -amplitude)
    c4.translate(stop[0], stop[1], in_place=True)

    return [start, c1, c2, c3, c4, stop]


def get_heart_coordinates_cartesian(t: float, max_amplitude:float = 0.1):
    # original equation (use i.e. t in range [0., 6.5] for a full rotation)
    x = -np.sqrt(2) * np.sin(t) ** 3
    cos_t = np.cos(t)
    y = 2 * cos_t - cos_t ** 2 - cos_t ** 3

    # scale and center (range and offset for this equation are pre-calculated)
    y_range = 2.6311290113857355
    y_offset = -0.23987051829473616
    x /= y_range
    y /= y_range
    y += y_offset
    y += 0.5
    range_after_amplitude = 1 - 2 * max_amplitude
    y *= range_after_amplitude
    x *= range_after_amplitude
    return x, y


class CommandGenerator:
    """Class that generates pattern commands as GCode that can be interpreted by CNC machines, i.e. pen plotters"""
    _valid_modes = ["rect_line", "rect_circle", "rect_spiral", "rect_heart"]

    def __init__(self,
                 mode: str = "rect_line",
                 num_segments: int = 250,
                 max_amp=1.0,
                 spiral_b=0.5,
                 rotations=5,
                 width: float = 300.,
                 height: float = 350.,
                 coordinate_system="cartesian",
                 canvas=None):
        self.canvas: PatternCanvas = canvas
        self.mode = mode if mode in self._valid_modes else "rect_line"
        self.coord_mode = coordinate_system
        if self.canvas:
            self.canvas.set_mode(coordinate_system)
            if self.coord_mode == "cartesian":
                self.canvas.set_uniforms(0.0, 0.0, width, height)
            else:
                self.canvas.set_uniforms(-width/2., -height/2., width/2., height/2.)
            self.canvas.show()
        self.rotations = 0
        self.max_size = None
        self.spiral_b = spiral_b
        self.amp_factor = 0.4 * max_amp if self.mode == "rect_circle" else max_amp
        # Note: max amp should be less than half of circle pattern size to make sure we're inside the canvas boundaries
        self.max_t = 6.5  # maximum parameter t for drawing a heart shape (assuming start is 0.0)

        if self.mode == "rect_circle" or self.mode == "rect_heart":
            self.rotations = 1
        elif self.mode == "rect_spiral":
            self.rotations = rotations
            theta = np.deg2rad(self.rotations * 360.)
            # (x, y) = (b * theta * cos(theta), b * theta * sin(theta)) for an archimedean spiral
            max_coordinate = Coordinate(self.spiral_b * theta * np.cos(theta), self.spiral_b * theta * np.sin(theta))
            self.max_size = 2. * max_coordinate.length() + 2. * self.amp_factor

        self.canvas_width: float = width
        self.canvas_height: float = height
        self.canvas_middle: Coordinate = Coordinate(np.round(self.canvas_width / 2., 1),
                       np.round(self.canvas_height / 2., 1))

        # Assume GL coordinate system, i.e. x, y in [-1.0; 1.0], P = (-1.0, -1.0) being bottom left
        # -> center of rotation is (0.0, 0.0)
        self.current_coord: Coordinate = Coordinate(0.0, 0.0)
        self.prev: Coordinate = Coordinate(0.0, 0.0)

        self.num_segments: int = num_segments
        self.current_segment: int = -1
        self.current_width = np.pi * 0.5 / self.num_segments
        if self.mode == "rect_line":
            self.current_width = 1 / self.num_segments

    def create_calibration_commands(self) -> list[bytes]:
        """Creates a set of calibration commands for the start of the GCode stream / file"""
        if self.coord_mode == "cartesian":
            cmds = [b"G21\n",  # programming in mm
                    b"G90\n",  # programming in absolute positioning
                    b"F800\n",  # set speed/feedrate
                    self.create_line_command(self.canvas_middle),  # move to middle
                    ]
        else:
            cmds = [b"$HY\n",
                    b"G92X0Y0\n",
                    f"G1Y{int(self.canvas_width)}F2000\n".encode(),
                    b"G92X0Y0\n",
                    ]
        return cmds

    def create_raise_pen_command(self):
        raise NotImplementedError

    def create_lower_pen_command(self):
        raise NotImplementedError

    def create_line_command(self, stop: Coordinate) -> str:
        """Create a GCode command to move to a given coordinate"""
        stop_tuple = stop.as_tuple()
        cmd = f"G1 X{np.round(stop_tuple[0], 1)} Y{np.round(stop_tuple[1], 1)}\n"
        return cmd.encode()

    def segment_line(self, start: Coordinate, stop: Coordinate, step_distance: float = 0.1):
        """Takes a start and stop Coordinate and outputs a list of coordinates on the line between start and stop
        (according to distance requested between coordinates)"""
        subdivison_line = [start]
        line = (stop - start)
        line_length = line.length()
        line.scale(1. / line_length, 1. / line_length, in_place=True)
        current_distance = step_distance
        while current_distance < line_length:
            step = line.scale(current_distance, current_distance)
            step = step.translate(start[0], start[1])
            subdivison_line.append(step)
            current_distance += step_distance
        subdivison_line.append(stop)
        return subdivison_line

    def generate_segment_coordinates_line(self,
                                          width: float,
                                          offset: Coordinate,
                                          amplitude: float = 0.1) -> list[Coordinate]:
        """
        Generate a set of coordinates that represent one segment of a line. The segment is multiplied with a
        rectangular wave.

        Args:
            width (float): The width of the segment
            offset (Coordinate): The offset to translate the segment to
            amplitude (float): The amplitude of the overlaid rectangular wave
        """
        seg_coords = []

        coord = Coordinate(0.0, 0.0)

        seg_coords.append(coord.translate(0.0, amplitude, in_place=True))
        seg_coords.append(coord.translate(width / 2., 0.0, in_place=True))
        seg_coords.append(coord.translate(0.0, -2 * amplitude, in_place=True))
        seg_coords.append(coord.translate(width / 2., 0.0, in_place=True))
        seg_coords.append(coord.translate(0.0, amplitude, in_place=True))

        for coordinate in seg_coords:
            coordinate.translate(offset[0], offset[1], in_place=True)
            self.current_coord = coordinate.copy()
            coordinate.translate(-1.0, 0.0, in_place=True)
            # Scale all coordinates to fit the canvas and move to its middle
            coordinate.scale(self.canvas_width / 2., self.canvas_height / 2., in_place=True)
            if self.coord_mode == "cartesian":
                coordinate.translate(self.canvas_middle[0], self.canvas_middle[1], in_place=True)
            else:
                coordinate.to_polar()

        return seg_coords

    def check_oob(self, coord: Coordinate):
        if self.coord_mode == "cartesian":
            if coord[0] < 0 or coord[0] > self.canvas_width or coord[1] < 0 or coord[1] > self.canvas_height:
                raise ValueError(f"Coordinate is out of bounds, got: {coord} ({self.coord_mode})")
        else:
            if coord[1] >= self.canvas_height/2. or coord[1] >= self.canvas_width/2.:
                raise ValueError(f"Coordinate is out of bounds, got: {coord} ({self.coord_mode})")

    def generate_segment_coordinates_circle(self,
                                            width: float,
                                            offset: Coordinate,
                                            rotation: float,
                                            amplitude: float = 0.1) -> list[Coordinate]:
        """
        Generate a set of coordinates that represent one segment of a circle. The segment is a line multiplied by a
        rectangular wave.

        Args:
            width (float): The width of the segment
            offset (Coordinate): The offset to translate the segment to
            rotation (float): The target rotation of the segment
            amplitude (float): The amplitude of the overlaid rectangular wave
        """
        seg_coords = []

        coord = Coordinate(0.0, 0.0)

        seg_coords.append(coord.translate(0.0, amplitude, in_place=True))
        seg_coords.append(coord.translate(width / 2., 0.0, in_place=True))
        seg_coords.append(coord.translate(0.0, -2 * amplitude, in_place=True))
        seg_coords.append(coord.translate(width / 2., 0.0, in_place=True))
        seg_coords.append(coord.translate(0.0, amplitude, in_place=True))

        for coordinate in seg_coords:
            coordinate.rotate(-rotation, in_place=True)
            coordinate.translate(offset[0], offset[1], in_place=True)
            self.current_coord = coordinate.copy()
            coordinate.translate(0.0, 0.5, in_place=True)  # move 0.5 up since we start at Y = 0.0
            # Scale all coordinates to fit the canvas and move to its middle
            coordinate.scale(self.canvas_width / 2., self.canvas_height / 2., in_place=True)
            if self.coord_mode == "cartesian":
                coordinate.translate(self.canvas_middle[0], self.canvas_middle[1], in_place=True)
            else:
                coordinate.to_polar(in_place=True)
                self.prev = coordinate.copy()
            self.check_oob(coordinate)

        return seg_coords

    def generate_segment_coordinates_spiral(self,
                                            offset: Coordinate,
                                            theta: float,
                                            amplitude: float,
                                            b: float = 0.5):
        """
        Generate a set of coordinates that represent one segment of an archimedean spiral. The segment is a line
        multiplied by a rectangular wave.

        Args:
            offset (Coordinate): The offset to translate the segment to
            theta (float): The rotation in radians of the segment
            amplitude (float): The amplitude of the overlaid rectangular wave
            b (float): The b in the equation r = b * theta, determines the distance between the spiral lines
        """
        seg_coords = []

        start = offset
        stop = Coordinate(b * theta * np.cos(theta), b * theta * np.sin(theta))
        diff = stop - start
        mid = start + 0.5 * diff

        if amplitude >= 0.05:
            # find an orthogonal vector (vec_x * vec_y = 0)
            orth = Coordinate(-diff[1], diff[0])
            orth_length = orth.length()  # length of orthogonal vector stays the same
            if orth_length >= 0.05:
                orth.scale(1. / orth_length, 1. / orth_length, in_place=True)  # normalize orthogonal vector
                orth.scale(amplitude, amplitude, in_place=True)  # scale orthogonal vector according to amplitude
                orth.translate(start[0], start[1], in_place=True)  # add orthogonal vector to start coordinate
                seg_coords.append(orth)

                orth = Coordinate(-diff[1], diff[0])
                orth.scale(1. / orth_length, 1. / orth_length, in_place=True)
                orth.scale(amplitude, amplitude, in_place=True)
                orth.translate(mid[0], mid[1], in_place=True)
                seg_coords.append(orth)

                orth = Coordinate(diff[1], -diff[0])
                orth.scale(1. / orth_length, 1. / orth_length, in_place=True)
                orth.scale(amplitude, amplitude, in_place=True)
                orth.translate(mid[0], mid[1], in_place=True)
                seg_coords.append(orth)

                orth = Coordinate(diff[1], -diff[0])
                orth.scale(1. / orth_length, 1. / orth_length, in_place=True)
                orth.scale(amplitude, amplitude, in_place=True)
                orth.translate(stop[0], stop[1], in_place=True)
                seg_coords.append(orth)

        seg_coords.append(stop)
        self.current_coord = stop.copy()
        for coordinate in seg_coords:
            # Scale all coordinates to fit the canvas and move to its middle
            coordinate.scale(1. / self.max_size, 1. / self.max_size, in_place=True)
            coordinate.scale(self.canvas_width, self.canvas_height, in_place=True)
            if self.coord_mode == "cartesian":
                coordinate.translate(self.canvas_middle[0], self.canvas_middle[1], in_place=True)
            else:
                coordinate.to_polar()
        return seg_coords

    def generate_segment_coordinates_heart(self, t1: float, t2: float, amplitude: float):
        x, y = get_heart_coordinates_cartesian(t1, self.amp_factor)
        start = Coordinate(x, y)

        x, y = get_heart_coordinates_cartesian(t2, self.amp_factor)
        stop = Coordinate(x, y)

        coordinates = multiply_line_with_rect_wave(start, stop, amplitude)

        self.current_coord = stop.copy()

        for coordinate in coordinates:
            coordinate.scale(self.canvas_width, self.canvas_height, in_place=True)
            if self.coord_mode == "cartesian":
                coordinate.translate(self.canvas_middle[0], self.canvas_middle[1], in_place=True)
            else:
                coordinate.to_polar(in_place=True)
            self.check_oob(coordinate)

        return coordinates

    def generate_segment_coordinates(self, amplitude: float = 0.1) -> list[Coordinate]:
        """
        Generates segment coordinates according to current mode (rect_circle, rect_spiral or rect_line)

        Args:
            amplitude (float): The amplitude of the overlaid rectangular wave
        """
        offset = self.current_coord
        amplitude = 0.1
        if self.mode == "rect_circle":
            r = self.current_segment * (self.rotations * 360. / self.num_segments)
            return self.generate_segment_coordinates_circle(self.current_width * 2, offset, r, amplitude)
        elif self.mode == "rect_line":
            return self.generate_segment_coordinates_line(self.current_width * 2, offset, amplitude)
        elif self.mode == "rect_spiral":
            # x = b * theta * cos(theta)
            # y = b * theta * sin(theta)
            theta = np.deg2rad((self.current_segment / self.num_segments) * (self.rotations * 360))
            return self.generate_segment_coordinates_spiral(offset, theta, amplitude, self.spiral_b)
        elif self.mode == "rect_heart":
            t1 = (self.current_segment / self.num_segments) * self.max_t
            t2 = ((self.current_segment + 1) / self.num_segments) * self.max_t
            return self.generate_segment_coordinates_heart(t1, t2, amplitude)

    def coordinates_to_commands(self, coordinates: list[Coordinate]) -> list[bytes]:
        """
        Converts a list of Coordinates to a list of GCode commands

        Args:
            coordinates (list[Coordinate]): List of Coordinates to convert

        Returns:
            list[str]: The list of Coordinates converted to GCode commands
        """
        cmds = []
        for coordinate in coordinates:
            cmds.append(self.create_line_command(coordinate))
        return cmds

    def get_segment_commands(self, buffer, val_min, val_max):
        """
        Determines amplitude for a segment based on an input buffer and min a max values and returns a list of GCode
        commands for the next segment according to these values.

        Args:
            buffer: A buffer containing values according to a single bandpower over time (usually Alpha)
            val_min: The tracked minimal bandpower value
            val_max: The tracked maximal bandpower value

        Returns: The list of GCode commands for the next segment
        """
        if self.current_segment >= self.num_segments:
            return []

        if abs(val_max - val_min) < 0.0001:
            amp = 0.0
        else:
            amp = (np.mean(buffer) - val_min) / (val_max - val_min)
            amp *= self.amp_factor

        ret = self.generate_segment_coordinates(amplitude=amp)
        if self.canvas:
            self.canvas.update_vbo(ret)
        self.current_segment += 1
        return self.coordinates_to_commands(ret)

    def get_commands(self, buffer, val_min, val_max):
        """
        Get commands based on where we are in processing and an input bandpower buffer, maximum and minimum values for
        the buffer

        Args:
            buffer: A buffer containing values according to a single bandpower over time (usually Alpha)
            val_min: The tracked minimal bandpower value
            val_max: The tracked maximal bandpower value

        Returns: The list of GCode commands for the next segment
        """
        #if self.current_segment == -1:
        #    self.current_segment = 0
        #    return self.create_calibration_commands()
        #else:
        return self.get_segment_commands(buffer, val_min, val_max)


import threading


class CommunicationInterface:
    """Class that handles communicating with the device, the bandpower calculator and the command generator"""
    def __init__(self, device: explorepy.Explore, port: serial.Serial, sr: int = 250, channel_num=8,
                 drawing_mode="rect_circle", file_path=None, canvas_size=[300., 350.], n_segments=250, max_amp=1.0,
                 spiral_b=0.5, rotations=5, coordinate_system="cartesian", canvas=None):
        self.explore_device = device
        self.serial_port = port
        self.calibration_thread = threading.Thread(target=self.write_calibration_commands)
        self.calibration_thread_started = False
        self.sr = sr
        self.file = None
        if file_path:
            p, _ = os.path.splitext(file_path)
            p = f"{p}.gcode"
            if os.path.isfile(p):
                print(f"Given file path {file_path} exists already!")
            else:
                print(f"Opening file {file_path}")
                self.file = open(p, "w")

        self.val_buffer_time = 8
        self.val_buffer_max_length = self.sr * 8

        self.update_rate = 30  # in Hz
        self.sleep_after_write = 0.1  # in s

        self.bp_buffer_max_length = self.update_rate

        self.val_buffers = [np.empty(self.val_buffer_max_length)] * channel_num
        self.val_current_indices = [0] * channel_num
        self.val_lengths = [0] * channel_num
        self.val_max_lengths = np.full(channel_num, self.val_buffer_max_length)

        self.bp_buffer = {'Delta': np.empty(self.update_rate),
                          'Theta': np.empty(self.update_rate),
                          'Alpha': np.empty(self.update_rate),
                          'Beta': np.empty(self.update_rate),
                          'Gamma': np.empty(self.update_rate)}
        self.bp_current_indices = {'Delta': 0, 'Theta': 0, 'Alpha': 0, 'Beta': 0, 'Gamma': 0}
        self.bp_lengths = {'Delta': 0, 'Theta': 0, 'Alpha': 0, 'Beta': 0, 'Gamma': 0}
        self.bp_max_lengths = {'Delta': self.update_rate,
                               'Theta': self.update_rate,
                               'Alpha': self.update_rate,
                               'Beta': self.update_rate,
                               'Gamma': self.update_rate}

        self.alpha_max = 0.0
        self.alpha_min = 1.0

        self.bp_calculator = BandpowerCalculator(fs=float(sr))
        self.command_generator = CommandGenerator(mode=drawing_mode, num_segments=n_segments, max_amp=max_amp,
                                                  spiral_b=spiral_b, rotations=rotations, width=canvas_size[0],
                                                  height=canvas_size[1], coordinate_system=coordinate_system,
                                                  canvas=canvas)

        self.explore_device.stream_processor.subscribe(callback=self.on_exg, topic=TOPICS.filtered_ExG)

        self.mode = 0  # 0 == calibrate, 1 == send
        self.start_ts = -1
        self.calibration_time = 20  # in s

    def on_exg(self, packet):
        """Get a packet and input it into the internal circular buffer"""
        p = packet.get_data()[1]
        for i in range(len(p)):
            # Small implementation of a circular buffer for the packet
            self.val_buffers[i][self.val_current_indices[i]] = p[i][0]
            self.val_current_indices[i] += 1
            self.val_lengths[i] = min(self.val_lengths[i] + 1, self.val_buffer_max_length)
            self.val_current_indices[i] %= self.val_buffer_max_length

    def wait_for_idle(self):
        print("Waiting until plotter is idle...")
        self.serial_port.write(b"?")
        ret = self.serial_port.read_until()
        while ret[1:4] == b'Run':
            time.sleep(0.05)
            self.serial_port.write(b"?")
            ret = self.serial_port.read_until()

    def write_calibration_commands(self) -> bool:
        print("Getting calibration commands...")
        cmds = self.command_generator.create_calibration_commands()
        if len(cmds) <= 0:
            raise ValueError("Got no calibration commands!")

        if self.file and not self.file.closed:
            print("Writing calibration commands to file...")
            for c in cmds:
                self.file.write(c.decode("utf-8"))

        if not self.serial_port:
            return True

        # Read from stream
        print("Reading starting information from plotter...")
        it = 0
        ret = b''
        comm = b''
        while ret != b"[MSG:'$H'|'$X' to unlock]\r\n":
            if it >= 10:
                raise ConnectionError(
                    f"Not getting expected starting phrases from plotter! Serial input read so far: {comm}, it: {it}")
            ret = self.serial_port.read_until()
            comm += ret
            it += 1
        print("Finished reading starting information from plotter!")

        # Write calibration commands and check replies
        print("Writing calibration commands to plotter...")
        for cmd in cmds:
            self.serial_port.write(cmd)
            ret = self.serial_port.read_until()
            while ret != b'ok\r\n':
                print("Waiting for ok...")
                if ret != '':
                    raise ValueError(f"Got unexpected reply from plotter: {ret}")
                ret = self.serial_port.read_until()
        print("Finished writing calibration commands to plotter!")
        return True

    def write_commands(self) -> bool:
        """Gets commands based on the bandpower buffer and write them to a file and the port (if available)"""
        cmds = self.command_generator.get_commands(self.bp_buffer['Alpha'], self.alpha_min, self.alpha_max)
        if len(cmds) <= 0:
            if self.file and not self.file.closed:
                print(f"Closing file {self.file.name}")
                self.file.close()
            return False

        if self.file and not self.file.closed:
            print("Writing commands to file:")
            print(cmds)
            for c in cmds:
                self.file.write(c.decode("utf-8"))

        if not self.serial_port:
            return True

        print("Writing commands to plotter...")
        for cmd in cmds:
            print(f"Writing: {cmd}")
            self.serial_port.write(cmd)
            ret = self.serial_port.read_until()
            while ret != b'ok\r\n':
                print("Waiting for ok...")
                if ret != '':
                    raise ValueError(f"Got unexpected reply from plotter: {ret}")
                ret = self.serial_port.read_until()
            self.wait_for_idle()
        print("Finished writing commands to plotter!")
        return True

    def run_logic(self):
        #print("Start of run logic")
        ret = True
        r = self.get_bandpowers()
        if r and self.mode == 0:
            if not self.calibration_thread.is_alive() and not self.calibration_thread_started:
                print("Starting calibration thread for pen plotter...")
                self.calibration_thread.start()
                self.calibration_thread_started = True
            if self.start_ts == -1:
                print(f"Starting bandpower calibration for {self.calibration_time}s...")
                self.start_ts = time.time()
            diff = time.time() - self.start_ts
            if diff > self.calibration_time:
                print(f"Finished calibrating, max alpha was: {self.alpha_max}, min alpha was: {self.alpha_min}")
                print("Starting stream to pen plotter now...")
                self.mode = 1
        if self.mode == 1:
            if self.calibration_thread.is_alive():
                print("Calibration thread for pen plotter is still alive, attempting to wait for it to finish...")
                self.calibration_thread.join()
                print("Calibration thread for pen plotter has finished, continuing...")
            ret = self.write_commands()
        return ret

    def run(self, event = None):
        """Main loop that's run to get the bandpowers from the internal value buffer and keep track of current mode
        (calibration vs. streaming)"""
        if _VISPY_AVAILABLE:
            if not self.run_logic():
                app.quit()
        else:
            while True:
                if not self.run_logic():
                    break
                time.sleep(1. / self.update_rate)

    def get_bandpowers(self):
        """Get the bandpowers for the current value buffers and write them into the internal bandpower circular
        buffers"""
        if np.array_equal(self.val_lengths, self.val_max_lengths):
            ret = self.bp_calculator.compute(np.array(self.val_buffers))
            for k in self.bp_buffer.keys():
                self.bp_buffer[k][self.bp_current_indices[k]] = ret[k]
                self.bp_current_indices[k] += 1
                self.bp_lengths[k] = min(self.bp_lengths[k] + 1, self.bp_buffer_max_length)
                self.bp_current_indices[k] %= self.bp_buffer_max_length
            self.alpha_max = max(np.mean(self.bp_buffer['Alpha'][:self.bp_lengths['Alpha']]), self.alpha_max)
            self.alpha_min = min(np.mean(self.bp_buffer['Alpha'][:self.bp_lengths['Alpha']]), self.alpha_min)
            return True
        return False


if _VISPY_AVAILABLE:
    v_shader = """
    #version 120

    attribute vec2 pos;

    uniform float min_x;
    uniform float min_y;
    uniform float max_y;
    uniform float max_x;

    uniform float middle_not_zero;

    uniform float aspect_ratio;

    void main() {
        float new_x = (2.0f * (pos.x + middle_not_zero * min_x) / (max_x - min_x) - middle_not_zero * 1.0f) / aspect_ratio;
        float new_y = 2.0f * (pos.y + middle_not_zero * min_y) / (max_y - min_y) - middle_not_zero * 1.0f;

        gl_Position = vec4(new_x, new_y, 0.0f, 1.0f);
    }
    """

    f_shader = """
    #version 120

    void main()
    {
        gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
    }
    """

    class PatternCanvas(app.Canvas):
        def __init__(self, mode="cartesian"):
            super().__init__()
            max_coordinates = 10000

            self.mode = mode

            self.background_color = (1., 1., 1., 1.)

            self.vbo = gloo.VertexBuffer(np.zeros(max_coordinates, dtype=[("pos", np.float32, 2)]))
            self.vbo_offset = 0
            self.indices = gloo.IndexBuffer(np.arange(1, dtype=np.uint16))

            self.pattern_program = gloo.Program(vert=v_shader, frag=f_shader)
            self.pattern_program.bind(self.vbo)
            self.pattern_program["aspect_ratio"] = self.size[0] / self.size[1]
            self.show()

        def set_mode(self, mode):
            self.mode = mode

        def set_uniforms(self, min_x, min_y, max_x, max_y):
            self.pattern_program["min_x"] = min_x
            self.pattern_program["min_y"] = min_y
            self.pattern_program["max_x"] = max_x
            self.pattern_program["max_y"] = max_y
            if self.mode == "polar":
                self.pattern_program["middle_not_zero"] = 0.0
            else:
                self.pattern_program["middle_not_zero"] = 1.0

        def update_vbo(self, coordinates: list[Coordinate]):
            n_coords = len(coordinates)
            coords = np.empty(n_coords, dtype=[("pos", np.float32, 2)])
            for i in range(n_coords):
                if self.mode == "cartesian":
                    coords["pos"][i] = coordinates[i].as_tuple()
                else:
                    coords["pos"][i] = coordinates[i].from_polar()
            self.vbo.set_subdata(coords, offset=self.vbo_offset, copy=True)
            self.vbo_offset += n_coords
            self.indices = gloo.IndexBuffer(np.arange(self.vbo_offset, dtype=np.uint16))

        def on_timer(self, event):
            self.update()

        def on_draw(self, event):
            gloo.clear(self.background_color)
            self.pattern_program.draw(mode="line_strip", indices=self.indices)

        def on_resize(self, event):
            width, height = event.physical_size
            if hasattr(self, "pattern_program"):
                self.pattern_program["aspect_ratio"] = width/height
            gloo.set_viewport(0, 0, width, height)


def main():
    draw_canvas = True
    if _VISPY_AVAILABLE:
        vispy.use("Glfw")
    pattern_canvas = PatternCanvas() if draw_canvas and _VISPY_AVAILABLE else None

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-p", "--port", nargs=1, type=str, required=True,
                            help="The port that the pen plotter is connected to")
    arg_parser.add_argument("-b", "--baud", nargs=1, type=int, default=[115200],
                            help="The baudrate for communicating with the pen plotter")
    arg_parser.add_argument("-n", "--name", nargs=1, type=str, required=True,
                            help="The name of the Explore device (i.e. \"Explore_ABCD\")")
    arg_parser.add_argument("-f", "--file", nargs=1, type=str, default=[None],
                            help="A filepath to save the generated GCode to.")
    arg_parser.add_argument("-m", "--mode", nargs=1, type=str, default=["circle"],
                            help="The pattern mode used for drawing, should be one of [line, circle, spiral, heart]",
                            choices=["line", "circle", "spiral", "heart"])
    arg_parser.add_argument("-s", "--size", nargs=2, type=float, default=[200., 200.],
                            help="The maximum canvas size of the pen plotter in mm, default is 200.0mm x 200.0mm",
                            metavar=("WIDTH", "HEIGHT"))
    # TODO add varying b, varying number of rotations, num segments
    # TODO test line again
    arg_parser.add_argument("--num_segments", nargs=1, type=int, default=[250],
                            help="Number of segments to draw (this determines how often the EEG data will be queried "
                                 "for one session)")
    arg_parser.add_argument("--amplitude_factor", nargs=1, type=float, default=[1.0],
                            help="The factor determine the maximal amplitude of the rectangular signal (default: 1.0)")
    arg_parser.add_argument("--spiral_b", nargs=1, type=float, default=[0.5],
                            help="The factor b used to calculate the segment positions inside the archimedean spiral "
                                 "(determines the distance between the lines, defaults to 0.5)")
    arg_parser.add_argument("--rotations", nargs=1, type=int, default=[5],
                            help="The number of rotations the archimedean spiral is supposed to cover (default: 5)")
    arg_parser.add_argument("--coordinate_system", nargs=1, type=str, default=["cartesian"],
                            help="The coordinate system used for plotting",
                            choices=["cartesian", "polar"])

    args = arg_parser.parse_args()

    print(f"Using port {args.port[0]}")

    p = None if args.port[0] == "debug" else args.port[0]
    baud = args.baud[0]
    device_name = args.name[0]
    size = args.size
    serial_port = serial.Serial(port=p, baudrate=baud) if p else None  # needs to match plotter
    explore_device = explorepy.Explore()
    explore_device.connect(device_name)

    gen = CommunicationInterface(explore_device,
                                 serial_port if p else p,
                                 drawing_mode=f"rect_{args.mode[0]}",
                                 file_path=args.file[0],
                                 canvas_size=size,
                                 n_segments=args.num_segments[0],
                                 max_amp=args.amplitude_factor[0],
                                 spiral_b=args.spiral_b[0],
                                 rotations=args.rotations[0],
                                 coordinate_system=args.coordinate_system[0],
                                 canvas=pattern_canvas)
    if _VISPY_AVAILABLE and pattern_canvas:
        run_update_rate = 1./10.
        t = app.Timer(interval=run_update_rate, connect=gen.run, start=True)
        t2 = app.Timer(interval=run_update_rate, connect=pattern_canvas.on_timer, start=True)
        app.run()
    else:
        gen.run()


if __name__ == '__main__':
    main()
