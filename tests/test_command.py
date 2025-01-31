import pytest

from explorepy.command import DeviceConfiguration
from explorepy.bt_mock_client import MockBtClient

def test_device_configuration():
    dev_name="Explore_DAAP"
    DeviceConfiguration(MockBtClient(dev_name))

def test_device_configuration_get_device_info():
    dev_name="Explore_DAAP"
    dev_config = DeviceConfiguration(MockBtClient(dev_name))
    with pytest.raises(NotImplementedError):
        dev_config.get_device_info()
