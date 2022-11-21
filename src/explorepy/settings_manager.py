from appdirs import(
    user_config_dir
) 
import yaml

class SettingsManager:
    def __init__(self):
        self.log_path = user_config_dir(appname="Mentalab", appauthor="explorepy")
        self.file_name = "channels.yaml"
        self.channel_mask_key = "channel_mask"
        self.settings_dict = {}

    def get_settings_dictionary(self):
        stream = open(self.log_path + "/" + self.file_name, 'a+')
        self.settings_dict = yaml.load(stream, Loader=yaml.SafeLoader)
        return self.settings_dict

    def get_file_path(self):
        return (self.log_path + self.file_name)
    
    def write_settings(self):
        with open(self.log_path + "/" + self.file_name, 'w') as fp:
            yaml.dump(self.get_settings_dictionary, fp)
    
    def set_channel_mask(self, value):
        self.get_settings_dictionary()[self.channel_mask_key] = value
    
    def update_device_settings(self, device_info_dict_update):
        settings_dict = self.get_settings_dictionary()
        for key, value in enumerate(device_info_dict_update.items()):
            self.get_settings_dictionary[key] = value

        self.write_settings()



        

    
    

