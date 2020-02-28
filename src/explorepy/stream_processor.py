# -*- coding: utf-8 -*-
from enum import Enum

from explorepy.parser import Parser
from explorepy.packet import DeviceInfo, CommandRCV, CommandStatus, EEG, Orientation, Environment, MarkerEvent

TOPICS = Enum('Topics', 'raw_ExG filtered_ExG device_info marker raw_orn mapped_orn cmd_ack env')


class StreamProcessor:
    def __init__(self):
        self.parser = None
        self.filters = []
        self.orn_calibrator = None
        self.device_info = {}
        self.subscribers = {key: set() for key in TOPICS}  # keys are topics and values are sets of callbacks

    def subscribe(self, callback, topic):
        self.subscribers.setdefault(topic, set()).add(callback)

    def unsubscribe(self, callback, topic):
        self.subscribers.setdefault(topic, set()).discard(callback)

    def start(self, device_name=None, mac_address=None):
        self.parser = Parser(callback=self.process, mode='device')
        self.parser.start_stream(device_name, mac_address)

    def process(self, packet):
        if self.subscribers:
            if isinstance(packet, DeviceInfo):
                self.device_info = packet.get_info()
                self.dispatch(topic=TOPICS.device_info, packet=packet)
            elif isinstance(packet, CommandRCV) or isinstance(packet, CommandStatus):
                self.dispatch(topic=TOPICS.cmd_ack, packet=packet)
            elif isinstance(packet, EEG):
                self.dispatch(topic=TOPICS.raw_ExG, packet=packet)
                self.apply_filters(packet=packet)
                self.dispatch(topic=TOPICS.filtered_ExG, packet=packet)
            elif isinstance(packet, Orientation):
                self.dispatch(topic=TOPICS.raw_orn, packet=packet)
                self.calculate_phys_orn(packet=packet)
            elif isinstance(packet, Environment):
                self.dispatch(topic=TOPICS.env, packet=packet)
            elif isinstance(packet, MarkerEvent):
                self.dispatch(topic=TOPICS.marker, packet=packet)

        # print(packet)

    def dispatch(self, topic, packet):
        for callback in self.subscribers[topic]:
            callback(packet)

    def apply_filters(self, packet):
        pass

    def calculate_phys_orn(self, packet):
        pass
