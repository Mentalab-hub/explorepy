import asyncio
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
    def __init__(self, mode: str="rect_line", num_segments: int=250, rotations=4):
        self.mode = mode if mode in self._valid_modes else "rect_line"
        self.rotations = rotations

        self.canvas_width: float = 10.0
        self.canvas_height: float = 10.0

        # Assume GL coordinate sytem, i.e. x, y in [-1.0; 1.0], P = (-1.0, -1.0) being bottom left
        # -> center of rotation is (0.0, 0.0)
        self.start_coord: Coordinate = Coordinate(-1.0, 0.0)
        self.end_coord: Coordinate = Coordinate(1.0, 0.0)
        self.current_coord: Coordinate = self.start_coord
        self.current_angle: float = 0.0

        self.num_segments: int = num_segments
        self.current_segment: int = 0
        self.current_width = 1. / self.num_segments

    def create_calib_command(self) -> list[str]:
        cmds = []
        cmds.append("G21\n")  # programming in mm
        cmds.append("G90\n")  # programming in absolute positioning
        cmds.append("G1 F8000\n")  # set speed/feedrate
        # cmds.append("M03 S150")  # spindle on with 150RPM
        return cmds

    def create_raise_pen_command(self):
        raise NotImplementedError

    def create_lower_pen_command(self):
        raise NotImplementedError

    def create_line_command(self, stop: Coordinate) -> str:
        stop_tuple = stop.as_tuple()
        return f"G1 X{np.round(stop_tuple[0], 3)} Y{np.round(stop_tuple[1], 3)}\n"

    def generate_segment_coordinates(self, width: float, offset: tuple[float, float], rotation: float, scale: tuple[float, float] = (1.0, 1.0), amplitude: float=0.1) -> list[Coordinate]:
        seg_coords = []

        coord = Coordinate(0.0, 0.0)

        seg_coords.append(coord.translate(0.0, amplitude, in_place=True))
        seg_coords.append(coord.translate(width/2., 0.0, in_place=True))
        seg_coords.append(coord.translate(0.0, -2 * amplitude, in_place=True))
        seg_coords.append(coord.translate(width/2., 0.0, in_place=True))
        seg_coords.append(coord.translate(0.0, amplitude, in_place=True))

        for coordinate in seg_coords:
            coordinate.scale(scale[0], scale[1], in_place=True)
            coordinate.scale(self.canvas_width, self.canvas_height, in_place=True)
            coordinate.rotate(rotation, in_place=True)
            coordinate.translate(offset[0], offset[1], in_place=True)

        return seg_coords

    def create_line_commands(self):
        """Test method to create a whole list of commands to draw a line with random amplitude segments"""
        rng = random.Random()
        rng.seed(time.time())
        w = 1./self.num_segments
        coordinates = []
        for i in range(self.num_segments):
            amp = rng.randint(0, 100) / 500
            ret = self.generate_segment_coordinates(width=w, offset=self.current_coord.as_tuple(), rotation=0.0, amplitude=amp)
            self.current_coord = ret[-1]
            self.current_segment += 1
            coordinates.extend(ret)
        cmds = []
        cmds.extend(self.create_calib_command())
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
            amp = rng.randint(0, 100) / 500
            ret = self.generate_segment_coordinates(width=w, offset=self.current_coord.as_tuple(), rotation=self.current_segment*r, amplitude=amp)
            self.current_coord = ret[-1]
            self.current_segment += 1
            coordinates.extend(ret)
        cmds = []
        cmds.extend(self.create_calib_command())
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
            ret = self.generate_segment_coordinates(width=w, offset=self.current_coord.as_tuple(), rotation=self.current_segment*r, amplitude=amp)
            w += (rotations*0.1/self.num_segments)
            self.current_coord = ret[-1]
            self.current_segment += 1
            coordinates.extend(ret)
        cmds = []
        cmds.extend(self.create_calib_command())
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

    def get_segment_commands(self, buffer):
        rng = random.Random()
        rng.seed(time.time())
        r = self.rotations * 360. / self.num_segments

        if self.current_segment >= self.num_segments: return []

        amp = rng.randint(0, 100) / 500
        ret = self.generate_segment_coordinates(width=self.current_width, offset=self.current_coord.as_tuple(), rotation=self.current_segment*r, amplitude=amp)
        self.current_width += (self.rotations*0.1/self.num_segments)
        self.current_coord = ret[-1]
        self.current_segment += 1
        return self.coordinates_to_commands(ret)


class CommunicationInterface:
    def __init__(self, device: explorepy.Explore, port: serial.Serial, sr: int = 250, channel_num=32):
        self.explore_device = device
        self.serial_port = port
        self.sr = sr

        self.val_buffer_time = 8
        self.val_buffer_max_length = self.sr * 8

        self.update_rate = 30  # in Hz

        self.bp_buffer_max_length = self.update_rate

        self.val_buffers = [np.empty(self.val_buffer_max_length)] * channel_num
        self.val_current_indices = [0] * channel_num
        self.val_lengths = [0] * channel_num
        self.val_max_lengths = np.full(channel_num, self.val_buffer_max_length)

        self.bp_buffer = {'Delta': np.empty(self.update_rate), 'Theta': np.empty(self.update_rate),
                          'Beta': np.empty(self.update_rate), 'Gamma': np.empty(self.update_rate)}
        self.bp_current_indices = {'Delta': 0, 'Theta': 0, 'Beta': 0, 'Gamma': 0}
        self.bp_lengths = {'Delta': 0, 'Theta': 0, 'Beta': 0, 'Gamma': 0}
        self.bp_max_lengths = {'Delta': self.update_rate, 'Theta': self.update_rate, 'Beta': self.update_rate,
                               'Gamma': self.update_rate}

        self.bp_calculator = BandpowerCalculator(fs=float(sr))
        self.command_generator = CommandGenerator()

        self.explore_device.stream_processor.subscribe(callback=self.on_exg, topic=TOPICS.filtered_ExG)

    def on_exg(self, packet):
        p = packet.get_data()[1]
        # TODO: Explain that this is a very simplified way of implementing a circular buffer
        for i in range(len(p)):
            self.val_buffers[i][self.val_current_indices[i]] = p[i][0]
            self.val_current_indices[i] += 1
            self.val_lengths[i] = min(self.val_lengths[i] + 1, self.val_buffer_max_length)
            self.val_current_indices[i] %= self.val_buffer_max_length

    def write_commands(self) -> None:
        cmds = self.command_generator.get_segment_commands(self.bp_buffer)
        print(cmds)
        if not self.serial_port:
            return
        for cmd in cmds:
            self.serial_port.write(cmd)

    def run(self):
        while True:
            if np.array_equal(self.val_lengths, self.val_max_lengths):
                ret = self.bp_calculator.compute(np.array(self.val_buffers))
                for k in self.bp_buffer.keys():
                    self.bp_buffer[k][self.bp_current_indices[k]] = ret[k]
                    self.bp_current_indices[k] += 1
                    self.bp_lengths[k] = min(self.bp_lengths[k] + 1, self.bp_buffer_max_length)
                    self.bp_current_indices[k] %= self.bp_buffer_max_length
                self.write_commands()
            time.sleep(1./self.update_rate)

def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-p", "--port", nargs=1, type=str, required=True,
                            help="The port that the pen plotter is gonnected to")
    arg_parser.add_argument("-b", "--baud", nargs=1, type=int, default=[115200],
                            help="The baudrate for communicating with the pen plotter")
    arg_parser.add_argument("-n", "--name", nargs=1, type=str, required=True,
                            help="The name of the Explore device (i.e. \"Explore_ABCD\")")

    args = arg_parser.parse_args()

    p = None if args.port[0] == "debug" else args.port[0]
    baud = args.baud[0]
    device_name = args.name[0]
    serial_port = serial.Serial(port=p, baudrate=baud) if p else None # needs to match plotter
    explore_device = explorepy.Explore()
    explore_device.connect(device_name)

    gen = CommunicationInterface(explore_device, serial_port if p else p)
    gen.run()


def main_debug():
    """Creates gcode with """
    command_generator = CommandGenerator(mode="rect_spiral")
    gcode = command_generator.create_pattern_commands()

    with open(file="test_code.gcode", mode="w") as f:
        f.writelines(gcode)


if __name__ == '__main__':
    main()
    #main_debug()
