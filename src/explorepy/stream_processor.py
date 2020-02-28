# -*- coding: utf-8 -*-
import numpy as np

from explorepy.parser import Parser


class StreamProcessor:
    def __init__(self):
        self.parser = None
        self.filters = []
        self.orn_calibrator = None

    def start(self, device_name=None, mac_address=None):
        self.parser = Parser(callback=self.process, mode='device')
        self.parser.start_stream(device_name, mac_address)

    def process(self, packet):
        print(packet)

    def apply_filters(self):
        pass
