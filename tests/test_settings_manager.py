from unittest.mock import (
    mock_open,
    patch
)

import pytest
import yaml

from explorepy.settings_manager import SettingsManager

channel_names = ["ch1", "ch2", "ch3", "ch4", "ch5", "ch6", "ch7", "ch8"]


@pytest.fixture
def settings_manager():
    with patch("builtins.open", new_callable=mock_open, read_data="{}"), \
         patch("os.makedirs"), \
         patch("os.path.exists", return_value=True):
        return SettingsManager("test_config")


def test_load_current_settings(settings_manager):
    with patch("builtins.open", new_callable=mock_open, read_data="{}"):
        settings_manager.load_current_settings()
        assert settings_manager.settings_dict == {}


def test_write_settings(settings_manager):
    with patch("builtins.open", new_callable=mock_open) as mock_open_file:
        settings_manager.settings_dict = {"test_key": "test_value"}
        settings_manager.write_settings()
        mock_open_file().write.assert_called()


def test_set_hardware_channel_mask(settings_manager):
    with patch("builtins.open", new_callable=mock_open, read_data=yaml.dump({"hardware_mask": [1, 0, 1]})):
        settings_manager.set_hardware_channel_mask([1, 1, 1])
        assert settings_manager.settings_dict["hardware_mask"] == [1, 1, 1]


def test_set_software_channel_mask(settings_manager):
    with patch("builtins.open", new_callable=mock_open, read_data=yaml.dump({"software_mask": [0, 0, 1]})):
        settings_manager.set_software_channel_mask([1, 1, 0])
        assert settings_manager.settings_dict["software_mask"] == [1, 1, 0]


def test_set_adc_mask(settings_manager):
    with patch("builtins.open", new_callable=mock_open):
        settings_manager.set_adc_mask("101")
        assert settings_manager.settings_dict["adc_mask"] == [1, 0, 1]


def test_get_adc_mask(settings_manager):
    with patch("builtins.open", new_callable=mock_open, read_data=yaml.dump({"adc_mask": [1, 0, 1]})):
        settings_manager.load_current_settings()
        assert settings_manager.get_adc_mask() == [1, 0, 1]


def test_set_mac_address(settings_manager):
    with patch("builtins.open", new_callable=mock_open):
        settings_manager.set_mac_address("00:1B:44:11:3A:B7")
        assert settings_manager.settings_dict["mac_address"] == "00:1B:44:11:3A:B7"


def test_get_mac_address(settings_manager):
    with patch("builtins.open", new_callable=mock_open, read_data=yaml.dump({"mac_address": "00:1B:44:11:3A:B7"})):
        settings_manager.load_current_settings()
        assert settings_manager.get_mac_address() == "00:1B:44:11:3A:B7"


def test_set_channel_count(settings_manager):
    with patch("builtins.open", new_callable=mock_open):
        settings_manager.set_channel_count(16)
        assert settings_manager.settings_dict["channel_count"] == 16


def test_get_channel_count(settings_manager):
    with patch("builtins.open", new_callable=mock_open, read_data=yaml.dump({"channel_count": 16})):
        settings_manager.load_current_settings()
        assert settings_manager.get_channel_count() == 16


def test_set_sampling_rate(settings_manager):
    with patch("builtins.open", new_callable=mock_open):
        settings_manager.set_sampling_rate(250)
        assert settings_manager.settings_dict["sampling_rate"] == 250


def test_get_sampling_rate(settings_manager):
    with patch("builtins.open", new_callable=mock_open, read_data=yaml.dump({"sampling_rate": 250})):
        settings_manager.load_current_settings()
        assert settings_manager.get_sampling_rate() == 250


def test_set_chan_names(settings_manager):
    with patch("builtins.open", new_callable=mock_open):
        settings_manager.set_chan_names([channel_names])
        assert settings_manager.settings_dict["channel_name"] == [channel_names]


def test_get_channel_names(settings_manager):
    with patch("builtins.open", new_callable=mock_open, read_data=yaml.dump({"channel_name": [channel_names]})):
        settings_manager.load_current_settings()
        assert settings_manager.get_channel_names() == [channel_names]


def test_update_device_settings(settings_manager):
    with patch("builtins.open", new_callable=mock_open):
        settings_manager.update_device_settings({"board_id": "PCB_304_801_XXX"})
        assert settings_manager.settings_dict["board_id"] == "PCB_304_801_XXX"


def test_update_device_settings_invalid_board_id(settings_manager):
    with patch("builtins.open", new_callable=mock_open):
        try:
            settings_manager.update_device_settings({"board_id": None})
            assert "board_id" in settings_manager.settings_dict
            assert settings_manager.settings_dict["board_id"] is None
        except KeyError:
            assert True


def test_save_current_session(settings_manager):
    with patch("shutil.copyfile") as mock_copyfile:
        settings_manager.save_current_session()
        mock_copyfile.assert_called()
