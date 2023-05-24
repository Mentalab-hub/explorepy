import struct
import time


class Debug:
    settings = {
        "BPS": True,
        "BIN": True,
        "RSSI": True,
        "DROPPED_COUNTER": True,
        "DROPPED_TS": True
    }

    def __init__(self, bps_buffer_size=100, settings=None, print_to_console=True):
        """
        Initializes a debug class to monitor incoming raw packets and calculate metrics on them.

        Args:
            settings (dict): A dictionary of the format {"SETTING_NAME": Boolean, ...} that declares which metrics to
            monitor. Valid setting names are BPS, BIN, DROPPED_COUNTER, DROPPED_TS
            bps_buffer_size (int): Size of the circular buffer (number of packets) to calculate the bps on
        """
        print("Debug mode active")
        self.bps = []
        self.bps_avg = -1
        self.bps_max = -1
        self.bps_min = -1
        self.bps_pos = 0
        self.bps_buffer_size = bps_buffer_size

        self.rssi = None

        self.last_packet = None

        self.dropped_counter = {}
        self.dropped_ts = {}

        if settings:
            self.settings = settings

        self.print_to_console = print_to_console

    def update_bps(self, packet):
        """
        Updates the internal average, max and min bytes/s with information from an incoming packet.

        The min and max values are not reset when the sampling rate changes.
        """
        num_bytes = len(packet.bin_data)
        if len(self.bps) < self.bps_buffer_size:
            self.bps.append((time.time(), num_bytes))
            t_diff = self.bps[len(self.bps) - 1][0] - self.bps[0][0]
        else:
            self.bps[self.bps_pos] = (time.time(), num_bytes)
            self.bps_pos = (self.bps_pos + 1) % self.bps_buffer_size
            t_diff = self.bps[self.bps_pos - 1][0] - self.bps[self.bps_pos][0]
        if abs(t_diff) < 0.0001:
            t_diff = -1
        res = 0
        for _, e in self.bps:
            res += e
        res /= t_diff
        self.bps_avg = res
        if len(self.bps) == self.bps_buffer_size:
            if (self.bps_min == -1 or self.bps_min > res):
                self.bps_min = res
            if self.bps_max < res:
                self.bps_max = res

    def update_dropped_counter(self, packet):
        # The counter resets when commands are sent and it seems to be shared across all packet types
        counter = packet.bin_data[1]
        if self.dropped_counter:
            distance = counter - self.dropped_counter
            if not (distance == 1 or distance == -255):
                print(f"Packet counter interrupted or reset, received counter: {counter},"
                      f"previous counter: {self.dropped_counter}, counter distance: {distance}")
        self.dropped_counter = counter

    def update_dropped_ts(self, packet):
        pid = packet.bin_data[0]
        ts = struct.unpack('<I', packet.bin_data[4:8])[0]
        if pid in self.dropped_ts:
            if self.dropped_ts[pid] > ts:
                print(f"Dropped packet with ID {pid} (Order of timestamps wrong)")
        self.dropped_ts[pid] = ts

    def __str__(self):
        msg = ""
        msg += "[Bytes per second]\tavg: {:.2f}\tmin: {:.2f}\tmax: {:.2f}\t\t"\
            .format(self.bps_avg, self.bps_min, self.bps_max) if self.settings["BPS"] else ""
        msg += f"[RSSI]:\t{self.rssi}\t" if self.settings["RSSI"] else ""
        msg += f"\n[Last packet]\t\t{self.last_packet}" if self.settings["BIN"] else ""
        return msg

    def process_bin(self, packet):
        if self.settings["BPS"]:
            self.update_bps(packet)
        if self.settings["BIN"]:
            self.last_packet = packet
        if self.settings["DROPPED_COUNTER"]:
            self.update_dropped_counter(packet)
        if self.settings["DROPPED_TS"]:
            self.update_dropped_ts(packet)

        if self.print_to_console:
            print(self)

