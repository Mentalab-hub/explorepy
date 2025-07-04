# TODO implement impedance check for the start

import os.path
import random
import time

import numpy as np
import serial
import argparse

from mne.time_frequency import psd_array_multitaper
from numpy._typing import NDArray
from scipy.integrate import simpson
from scipy.signal import periodogram, welch

import explorepy
from explorepy.stream_processor import TOPICS

class Coordinate:
    def __init__(self, x: float, y: float):
        self._coord = np.array([x, y, 1.], dtype=np.float64)

    def as_tuple(self) -> tuple[float, float]:
        return float(self._coord[0]), float(self._coord[1])

    def translate(self, x: float, y: float, in_place: bool = False):
        t_mat = np.array([[1., 0., x],
                          [0., 1., y],
                          [0., 0., 1.]])
        ret = np.matmul(t_mat, self._coord)
        if in_place: self._coord = ret
        return Coordinate(ret[0], ret[1])


    def rotate(self, angle: float, in_place: bool = False):
        angle = np.deg2rad(angle)
        r_mat = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                          [np.sin(angle), np.cos(angle), 0.0],
                          [0.0, 0.0, 1.0]])
        ret = np.matmul(r_mat, self._coord)
        if in_place: self._coord = ret
        return Coordinate(ret[0], ret[1])

    def scale(self, x: float, y: float, in_place: bool = False):
        s_mat = np.array([[x, 0.0, 0.0],
                          [0.0, y, 0.0],
                          [0.0, 0.0, 1.0]])
        ret = np.matmul(s_mat, self._coord)
        if in_place: self._coord = ret
        return Coordinate(ret[0], ret[1])

    def length(self):
        return np.sqrt(self._coord[0]**2 + self._coord[1]**2)  # drop z == 1 since we're in the xy-plane

    def __getitem__(self, item):
        if item < 0 or item > 2:
            raise ValueError(f"Can't access index {item}, valid indices for Coordinates are 0 (x) and 1 (y)")
        return float(self._coord[item])

    def __add__(self, other):
        if type(other) != Coordinate:
            raise ValueError(f"Addition to Coordinate not defined for type {type(other)}")
        x = self.as_tuple()
        y = other.as_tuple()
        return Coordinate(x[0] + y[0], x[1] + y[1])

    def __sub__(self, other):
        if type(other) != Coordinate:
            raise ValueError(f"Subtraction from Coordinate not defined for type {type(other)}")
        x = self.as_tuple()
        y = other.as_tuple()
        return Coordinate(x[0] - y[0], x[1] - y[1])

    def __mul__(self, other):
        if type(other) not in [int, float]:
            raise ValueError(f"Coordinate multiplication is currently only implemented for uniform scaling with int or float")
        x = self.as_tuple()
        return Coordinate(x[0] * other, x[1] * other)

    def __rmul__(self, other):
        return self * other

    def copy(self):
        return Coordinate(self._coord[0], self._coord[1])

    def __str__(self):
        return f"({self._coord[0]}, {self._coord[1]})"

class BandpowerCalculator:
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
        bandpowers = {}
        for name, band in self.bands.items():
            bp = self._bandpower(data, self.fs, method, band, relative)  # array of channels
            bandpowers[name] = np.mean(bp)
        return bandpowers

    def _bandpower(self, data, fs, method, band, relative):
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


class CommandGenerator:
    _valid_modes = ["rect_line", "rect_circle", "rect_spiral"]
    def __init__(self, mode: str="rect_line", num_segments: int=250, rotations=5, width: float=300., height: float=350.):
        self.mode = mode if mode in self._valid_modes else "rect_line"
        self.rotations = 0
        if self.mode == "rect_circle":
            self.rotations = 1
        elif self.mode == "rect_spiral":
            self.rotations = rotations

        self.canvas_width: float = width
        self.canvas_height: float = height
        self.canvas_middle: Coordinate = Coordinate(np.round(self.canvas_width/2., 1),
                                                    np.round(self.canvas_height/2., 1))

        # Assume GL coordinate system, i.e. x, y in [-1.0; 1.0], P = (-1.0, -1.0) being bottom left
        # -> center of rotation is (0.0, 0.0)
        self.start_coord: Coordinate = Coordinate(0.0, 0.0)
        self.current_coord: Coordinate = self.start_coord
        self.current_angle: float = 0.0

        self.num_segments: int = num_segments
        self.current_segment: int = -1
        self.current_width = np.pi * 0.5 / self.num_segments

        # Note: max amp should be half of pattern size to make sure we're inside the canvas boundaries
        self.amp_factor = 0.5 if self.mode == "rect_circle" else 1.0
        self.spiral_b = 0.5

    def get_calibration_commands(self) -> list[str]:
        cmds = ["G21\n", # programming in mm
                "G90\n", # programming in absolute positioning
                "F800\n", # set speed/feedrate
                self.create_line_command(self.canvas_middle), # move to middle
                ]
        return cmds

    def create_raise_pen_command(self):
        raise NotImplementedError

    def create_lower_pen_command(self):
        raise NotImplementedError

    def create_line_command(self, stop: Coordinate) -> str:
        stop_tuple = stop.as_tuple()
        return f"G1 X{np.round(stop_tuple[0], 1)} Y{np.round(stop_tuple[1], 1)}\n"

    def generate_segment_coordinates_circle(self,
                                     width: float,
                                     offset: Coordinate,
                                     rotation: float,
                                     scale: tuple[float, float] = (1.0, 1.0),
                                     amplitude: float=0.1) -> list[Coordinate]:
        seg_coords = []

        coord = Coordinate(0.0, 0.0)

        seg_coords.append(coord.translate(0.0, amplitude, in_place=True))
        seg_coords.append(coord.translate(width/2., 0.0, in_place=True))
        seg_coords.append(coord.translate(0.0, -2 * amplitude, in_place=True))
        seg_coords.append(coord.translate(width/2., 0.0, in_place=True))
        seg_coords.append(coord.translate(0.0, amplitude, in_place=True))

        for coordinate in seg_coords:
            coordinate.scale(scale[0], scale[1], in_place=True)
            coordinate.rotate(-rotation, in_place=True)
            coordinate.translate(offset[0], offset[1], in_place=True)
            self.current_coord = coordinate.copy()
            coordinate.translate(0.0, 0.5, in_place=True)  # move 0.5 up since we start at Y = 0.0
            coordinate.scale(self.canvas_width/2., self.canvas_height/2., in_place=True)
            coordinate.translate(self.canvas_middle[0], self.canvas_middle[1], in_place=True)

        return seg_coords

    def generate_segment_coordinates(self,
                                     amplitude: float=0.1,
                                     scale: tuple[float, float] = (1.0, 1.0)) -> list[Coordinate]:
        offset = self.current_coord
        if self.mode == "rect_circle":
            r = self.current_segment * (self.rotations * 360. / self.num_segments)
            return self.generate_segment_coordinates_circle(self.current_width*2, offset, r, scale, amplitude)
        elif self.mode == "rect_spiral":
            # x = b * theta * cos(theta)
            # y = b * theta * sin(theta)
            theta = np.deg2rad((self.current_segment / self.num_segments) * (self.rotations * 360))
            return self.generate_segment_coordinates_spiral(offset, theta, amplitude, self.spiral_b)

    def create_line_commands(self):
        """Test method to create a whole list of commands to draw a line with random amplitude segments"""
        rng = random.Random()
        rng.seed(time.time())
        w = 1./self.num_segments
        coordinates = []
        for i in range(self.num_segments):
            amp = rng.randint(0, 100) / 500
            ret = self.generate_segment_coordinates(width=w, offset=self.current_coord.as_tuple(), rotation=0.0,
                                                    amplitude=amp)
            self.current_coord = ret[-1]
            self.current_segment += 1
            coordinates.extend(ret)
        cmds = []
        cmds.extend(self.get_calibration_commands())
        cmds.extend(self.coordinates_to_commands(coordinates))
        return cmds

    def create_circle_commands(self):
        """Test method to create a whole list of commands to draw a circle with random amplitude segments"""
        rng = random.Random()
        rng.seed(time.time())
        w = 1./self.num_segments
        r = 360. / self.num_segments
        coordinates = []
        while self.current_segment < self.num_segments:
            amp = rng.randint(0, 100) / 1000
            ret = self.generate_segment_coordinates(width=w, offset=self.current_coord.as_tuple(),
                                                    rotation=self.current_segment*r, amplitude=amp)
            self.current_coord = ret[-1]
            self.current_segment += 1
            coordinates.extend(ret)
        cmds = []
        cmds.extend(self.get_calibration_commands())
        cmds.extend(self.coordinates_to_commands(coordinates))
        return cmds

    def create_spiral_commands(self, rotations=2):
        """Test method to create a whole list of commands to draw a spiral with random amplitude segments"""
        rng = random.Random()
        rng.seed(time.time())
        w = 1./self.num_segments
        r = rotations * 360. / self.num_segments
        coordinates = []
        while self.current_segment < self.num_segments:
            amp = rng.randint(0, 100) / 500
            ret = self.generate_segment_coordinates(width=w, offset=self.current_coord.as_tuple(),
                                                    rotation=self.current_segment*r, amplitude=amp)
            w += (rotations*0.1/self.num_segments)
            self.current_coord = ret[-1]
            self.current_segment += 1
            coordinates.extend(ret)
        cmds = []
        cmds.extend(self.get_calibration_commands())
        cmds.extend(self.coordinates_to_commands(coordinates))
        return cmds

    def coordinates_to_commands(self, coordinates: list[Coordinate]) -> list[str]:
        cmds = []
        for coordinate in coordinates:
            cmds.append(self.create_line_command(coordinate))
        return cmds

    def create_pattern_commands(self):
        if self.mode == "rect_line":
            return self.create_line_commands()
        elif self.mode == "rect_circle":
            return self.create_circle_commands()
        elif self.mode == "rect_spiral":
            return self.create_spiral_commands(rotations=4)

    def get_segment_commands(self, buffer, val_min, val_max):
        if self.current_segment >= self.num_segments: return []

        if abs(val_max - val_min) < 0.0001 or True:
            amp = 0.0
        else:
            amp = (np.mean(buffer) - val_min) / (val_max - val_min)
            amp *= self.amp_factor

        amp = (self.current_segment % 2) * self.amp_factor  # debug

        ret = self.generate_segment_coordinates(amplitude=amp)
        self.current_segment += 1
        return self.coordinates_to_commands(ret)

    def generate_segment_coordinates_spiral(self, offset: Coordinate, theta: float, amplitude: float, b: float = 0.5):
        seg_coords = []

        start = offset
        stop = Coordinate(b * theta * np.cos(theta), b * theta * np.sin(theta))
        diff = stop - start
        mid = start + 0.5 * diff

        # find an orthogonal vector (vec_x * vec_y = 0) and scale it according to amplitude
        if amplitude >= 0.05:
            orth = Coordinate(-diff[1], diff[0])
            orth_length = orth.length()  # length of orthogonal vector stays the same
            if orth_length >= 0.05:
                orth.scale(1. / orth_length, 1. / orth_length, in_place=True)
                orth.scale(amplitude, amplitude, in_place=True)
                orth.translate(start[0], start[1], in_place=True)
                seg_coords.append(orth)  # seg_coords.append(coord.translate(0.0, amplitude, in_place=True))

                orth = Coordinate(-diff[1], diff[0])
                orth.scale(1. / orth_length, 1. / orth_length, in_place=True)
                orth.scale(amplitude, amplitude, in_place=True)
                orth.translate(mid[0], mid[1], in_place=True)
                seg_coords.append(orth)  # seg_coords.append(coord.translate(width/2., 0.0, in_place=True))

                orth = Coordinate(diff[1], -diff[0])
                orth.scale(1. / orth_length, 1. / orth_length, in_place=True)
                orth.scale(amplitude, amplitude, in_place=True)
                orth.translate(mid[0], mid[1], in_place=True)
                seg_coords.append(orth)  # seg_coords.append(coord.translate(0.0, -2 * amplitude, in_place=True))

                orth = Coordinate(diff[1], -diff[0])
                orth.scale(1. / orth_length, 1. / orth_length, in_place=True)
                orth.scale(amplitude, amplitude, in_place=True)
                orth.translate(stop[0], stop[1], in_place=True)
                seg_coords.append(orth)  # seg_coords.append(coord.translate(width/2., 0.0, in_place=True))

        seg_coords.append(stop)
        self.current_coord = stop.copy()
        for c in seg_coords:
            c.scale(1.0, 1.0, in_place=True)
            c.translate(0.0, 0.0, in_place=True)
        return seg_coords

    def get_commands(self, buffer, val_min, val_max):
        if self.current_segment == -1:
            self.current_segment = 0
            return self.get_calibration_commands()
        else:
            return self.get_segment_commands(buffer, val_min, val_max)


class CommunicationInterface:
    def __init__(self, device: explorepy.Explore, port: serial.Serial, sr: int = 250, channel_num=8,
                 drawing_mode="rect_circle", file_path=None, canvas_size=[300., 350.]):
        self.explore_device = device
        self.serial_port = port
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
        self.command_generator = CommandGenerator(mode=drawing_mode, width=canvas_size[0], height=canvas_size[1])

        self.explore_device.stream_processor.subscribe(callback=self.on_exg, topic=TOPICS.filtered_ExG)

        self.mode = 0  # 0 == calibrate, 1 == send
        self.start_ts = -1
        #self.calibration_time = 20  # in s
        self.calibration_time = 20

    def on_exg(self, packet):
        p = packet.get_data()[1]
        # TODO: Explain that this is a very simplified way of implementing a circular buffer
        for i in range(len(p)):
            self.val_buffers[i][self.val_current_indices[i]] = p[i][0]
            self.val_current_indices[i] += 1
            self.val_lengths[i] = min(self.val_lengths[i] + 1, self.val_buffer_max_length)
            self.val_current_indices[i] %= self.val_buffer_max_length

    def write_commands(self) -> bool:
        cmds = self.command_generator.get_commands(self.bp_buffer['Alpha'], self.alpha_min, self.alpha_max)
        if len(cmds) <= 0:
            if self.file and not self.file.closed:
                print(f"Closing file {self.file.name}")
                self.file.close()
            return False

        if self.file and not self.file.closed:
            self.file.writelines(cmds)

        if not self.serial_port: return True
        for cmd in cmds:
            self.serial_port.write(cmd)
            time.sleep(0.1)
        return True

    def run(self):
        while True:
            r = self.get_bandpowers()
            if r and self.mode == 0:
                if self.start_ts == -1:
                    print(f"Starting calibration for {self.calibration_time}s...")
                    self.start_ts = time.time()
                diff = time.time() - self.start_ts
                if diff > self.calibration_time:
                    print(f"Finished calibrating, max alpha was: {self.alpha_max}, min alpha was: {self.alpha_min}")
                    print("Starting stream to pen plotter now...")
                    self.mode = 1
            if self.mode == 1:
                ret = self.write_commands()
                if not ret:
                    break
            time.sleep(1./self.update_rate)

    def get_bandpowers(self):
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


def main():
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
                            help="The pattern mode used for drawing, should be one of [line, circle, spiral]",
                            choices=["line", "circle", "spiral"])
    arg_parser.add_argument("-s", "--size", nargs=2, type=float, default=[200., 200.],
                            help="The maximum canvas size of the pen plotter in mm, default is 200.0mm x 200.0mm",
                            metavar=("WIDTH", "HEIGHT"))

    args = arg_parser.parse_args()

    p = None if args.port[0] == "debug" else args.port[0]
    baud = args.baud[0]
    device_name = args.name[0]
    size = args.size if args.size[0] >= 50. and args.size[1] >= 50. else [50., 50.]
    serial_port = serial.Serial(port=p, baudrate=baud) if p else None # needs to match plotter
    explore_device = explorepy.Explore()
    explore_device.connect(device_name)

    gen = CommunicationInterface(explore_device, serial_port if p else p, drawing_mode=f"rect_{args.mode[0]}",
                                 file_path=args.file[0], canvas_size=size)
    gen.run()


if __name__ == '__main__':
    main()
