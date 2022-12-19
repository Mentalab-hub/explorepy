import os

import yaml
from appdirs import user_config_dir


class SettingsManager:
    def __init__(self, name):
        self.settings_dict = None
        self.log_path = user_config_dir(appname="Mentalab", appauthor="explorepy")
        self.file_name = name + ".yaml"
        self.full_file_path = os.path.join(self.log_path, self.file_name)
        os.makedirs(self.log_path, exist_ok=True)

        if not os.path.exists(self.full_file_path):
            with open(self.full_file_path, 'w'):
                pass
        self.hardware_channel_mask_key = "hardware_mask"
        self.software_channel_mask_key = "software_mask"
        self.adc_mask_key = "adc_mask"
        self.channel_name_key = "channel_name"
        self.channel_count_key = "channel_count"
        self.mac_address_key = "mac_address"
        self.board_id_key = "board_id"
        self.sr_key = "sampling_rate"

    def load_current_settings(self):
        self.settings_dict = {}
        stream = open(self.full_file_path, 'r')
        self.settings_dict = yaml.load(stream, Loader=yaml.SafeLoader)
        if self.settings_dict is None:
            self.settings_dict = {}

    def get_file_path(self):
        return self.log_path + self.file_name

    def write_settings(self):
        with open(self.full_file_path, 'w+') as fp:
            yaml.safe_dump(self.settings_dict, fp, default_flow_style=False)
            fp.close()

    def set_hardware_channel_mask(self, value):
        """ Setter method for hardware channel mask for Explore Desktop"""
        self.load_current_settings()
        self.settings_dict[self.hardware_channel_mask_key] = value
        self.write_settings()

    def set_software_channel_mask(self, value):
        """ Setter method for software channel mask for Explore Desktop"""
        self.load_current_settings()
        self.settings_dict[self.software_channel_mask_key] = value
        self.write_settings()

    def set_adc_mask(self, value):
        """ method to save virtual adc mask for ONLY 32 channel board """
        self.load_current_settings()
        value_list = [int(ch) for ch in [*value]]
        self.settings_dict[self.software_channel_mask_key] = value_list
        self.settings_dict[self.adc_mask_key] = value_list
        self.write_settings()

    def get_adc_mask(self):
        self.load_current_settings()
        return self.settings_dict.get(self.adc_mask_key)

    def set_channel_count(self, channel_number):
        """ Setter method to set channel count for Explore Desktop"""
        self.load_current_settings()
        if self.channel_count_key not in self.settings_dict:
            self.settings_dict[self.channel_count_key] = channel_number
        self.write_settings()

    def get_mac_address(self):
        """Returns string representation of device mac address"""
        self.load_current_settings()
        return self.settings_dict.get(self.mac_address_key)

    def get_channel_count(self):
        '''Returns string representation of device mac address'''
        self.load_current_settings()
        return self.settings_dict.get(self.channel_count_key)

    def set_mac_address(self, mac_address):
        self.load_current_settings()
        self.settings_dict[self.mac_address_key] = mac_address
        self.write_settings()

    def update_device_settings(self, device_info_dict_update):
        self.load_current_settings()
        for key, value in device_info_dict_update.items():
            self.settings_dict[key] = value
        if "board_id" in device_info_dict_update:
            if self.settings_dict["board_id"] == "PCB_304_801_XXX":
                self.settings_dict[self.channel_count_key] = 32
                self.settings_dict[self.hardware_channel_mask_key] = [1 for _ in range(32)]
                if self.software_channel_mask_key not in self.settings_dict:
                    hardware_adc = self.settings_dict.get(self.hardware_channel_mask_key)
                    self.settings_dict[self.software_channel_mask_key] = hardware_adc
                self.settings_dict[self.adc_mask_key] = self.settings_dict.get(self.software_channel_mask_key)

        if self.channel_count_key not in self.settings_dict:
            self.settings_dict[self.channel_count_key] = 8 if sum(self.settings_dict["adc_mask"]) > 4 else 4
        self.write_settings()

    def set_sampling_rate(self, value):
        """Setter method for sampling rate value"""
        self.load_current_settings()
        self.settings_dict[self.sr_key] = value
        self.write_settings()

    def set_chan_names(self, value):
        """Setter method for channel names for Explore Desktop"""
        self.load_current_settings()
        self.settings_dict[self.channel_name_key] = value
        self.write_settings()

    def __str__(self):
        self.load_current_settings()
        return self.settings_dict
