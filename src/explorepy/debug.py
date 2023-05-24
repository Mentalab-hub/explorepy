import struct
import time


class Debug:
    settings = {
        "BPS": True,
        "BIN": True,
        "RSSI": False,
        "DROPPED_COUNTER": False,
        "DROPPED_TS": True
    }

    def __init__(self, bps_max=100, settings=None):
        """
        Initializes a debug class to monitor incoming raw packets and calculate metrics on them.

        Args:
            settings (dict): A dictionary of the format {"SETTING_NAME": Boolean, ...} that declares which metrics to
            monitor. Valid setting names are BPS, BIN, DROPPED_COUNTER, DROPPED_TS
            bps_max (int): Size of the circular buffer (number of packets) to calculate the bps on
        """
        print("Debug mode active")
        self.bps = []
        self.bps_avg = 0
        self.bps_pos = 0
        self.bps_max = bps_max

        self.rssi = None

        self.last_packet = None

        self.dropped_counter = {}
        self.dropped_ts = {}
        if settings:
            self.settings = settings

    def update_bps(self, packet):
        """
        Updates the internal average bits/s with information from an incoming packet.
        """
        num_bytes = len(packet.bin_data)
        if len(self.bps) < self.bps_max:
            self.bps.append((time.time(), num_bytes))
            t_diff = self.bps[len(self.bps) - 1][0] - self.bps[0][0] if len(self.bps) > 1 else -1
        else:
            self.bps[self.bps_pos] = (time.time(), num_bytes)
            self.bps_pos = (self.bps_pos + 1) % self.bps_max
            t_diff = self.bps[self.bps_pos - 1][0] - self.bps[self.bps_pos][0]
        res = 0
        for _, e in self.bps:
            res += e
        res /= t_diff
        self.bps_avg = res * 8

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

    def write_report(self):
        msg = ""
        msg += f"Bits per second: {self.bps_avg}\t" if self.settings["BPS"] else ""
        msg += f"RSSI: {self.rssi}\t" if self.settings["RSSI"] else ""
        msg += "\n"
        msg += f"Last packet as bin:\n{self.last_packet}" if self.settings["BIN"] else ""
        print(msg)

    def process_bin(self, packet):
        if self.settings["BPS"]:
            self.update_bps(packet)
        if self.settings["BIN"]:
            self.last_packet = packet
        if self.settings["DROPPED_COUNTER"]:
            self.update_dropped_counter(packet)
        if self.settings["DROPPED_TS"]:
            self.update_dropped_ts(packet)

        self.write_report()

