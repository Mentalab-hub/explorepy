from unittest import TestCase

import numpy as np

from explorepy.packet import (
    EEG94,
    EEG98
)
from explorepy.tools import ImpedanceMeasurement


class TestImpedanceMeasurement(TestCase):

    def test_calc_imp(self):
        d = res = bytes.fromhex('ff00062720003ce6ff47eeffd3f4ff5ce9ff67030064e7fff40d00ff0006b21a00d5dbffede4ffe9edfffce8ff4d010072ddffc00a00ff0006da1a003cdcff3ce5ff2eeeffcfe8ff340100afddff710a00ff00069f1f007de5ff8eedff5ff4ff09e9fff6020077e6fff40c00ff0006242300e0ebff49f3ff86f8ff5fe9ff610400b1ecff280f00ff00063620003ee6ff25eeffd2f4ff55e9ff6c03004de7ffeb0d00ff0006c01a00ebdbffdbe4fff8edfffae8ff38010071ddffb20a00ff0006d21a0022dcff19e5ff2eeeffd2e8ff3b0100abddff6f0a00ff00069b1f0069e5ff71edff52f4fffae8ff06030073e6fff70c00ff0006152300dcebff48f3ff96f8ff66e9ff4d0400acecff3f0f00ff000618200033e6ff40eeffdbf4ff46e9ff6003005be7fffa0d00ff0006a41a00dbdbfff6e4fff0edffd4e8ff4d010076ddffb50a00ff0006d71a0032dcff34e5ff1beeffd1e8ff3a0100b4ddff630a00ff0006a81f006fe5ff96edff4ef4ff18e9ff0c030078e6ffdd0c00ff0006042300daebff6af3ffacf8ff61e9ff7a0400b0ecff2b0f00ff000613200026e6ff4ceeffd5f4ff38e9ff74030058e7fff30d00afbeadde')
        eeg98 = EEG98(12345, res, 0)
        eeg98.data
        device_info = {'sampling_rate': 250, 'adc_mask': [1,1,1,1,1,1,1,1]}
        calib_info = {'slope': 219630, 'offset': 41.627}
        im = ImpedanceMeasurement(device_info, calib_info, 50)
        res = im.measure_imp(eeg98)
        print("Imp measurement res: ")
        print(res.imp_data)

    def test_measure_imp(self):
        d = bytes([i % 256 for i in range(495)])  # 495 bytes of dummy data, resembles 33 samples in 4+1 channels
        f = b'\xaf\xbe\xad\xde'
        eeg94 = EEG94(12345, d+f)
        eeg94.data = np.array(
            [[40, 3333, 78910, -30, 0], [20, -1000, 10, 30, 0], [10, 2345, 77016, 11, 45], [15, 1234, 70000, 2, 44]])
        calib_info = {'slope': 219630, 'offset': 41.627}
        eeg94.calculate_impedance(calib_info)
        print("Res: "+eeg94.imp_data)

