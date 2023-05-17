import time


class Debug:
    settings = {
        "BPS_EXG": True,
        "BIN": True
    }

    def __init__(self):
        print("Debug mode active")
        self.bps_exg = 0
        self.bps = 0
        self.rssi = None
        self.last_timestamp = 0

    def update_bps(self, packet):
        num_bits = len(packet.bin_data)
        current_time = time.time()
        if self.last_timestamp > 0:
            diff_t = current_time - self.last_timestamp
            if diff_t > 0:
                self.bps = num_bits / diff_t
        self.last_timestamp = current_time
        print(f"Overall bps: {self.bps} bps, length: {num_bits} bits")

    def process_bin(self, packet):
        #print(f"Received packet with topic BIN, type: {type(packet.bin_data)}, length: {len(packet.bin_data)}")
        if self.settings["BPS_EXG"]:
            self.update_bps(packet)
        if self.settings["BIN"]:
            print(f"Packet as hex: [{packet}]")
        #raise NotImplementedError
