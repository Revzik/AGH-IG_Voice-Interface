import unittest
import matplotlib.pyplot as plt
import numpy as np

from src.utils.sound_loader import SoundLoader as Sl


class SoundLoaderTest(unittest.TestCase):
    #tutaj można pisać testy do metod z Mfcc (na wzór test_configurator.py)

    def test_remove_dc_offset(self):
        test_soundlist = [
            {'name': 1, 'fs': 1000, 'wav': 1 + np.sin(8*np.pi*np.linspace(0, 1, 1000, False))},
            {'name': 1, 'fs': 1000, 'wav': -0.5 + np.sin(2*np.pi*np.linspace(0, 1, 1000, False))},
            {'name': 1, 'fs': 1000, 'wav': np.arange(0, 10)},
            {'name': 1, 'fs': 1000, 'wav': np.ones(100)}
        ]
        sl_without_dc_test = Sl.remove_dc_offset(test_soundlist)
        for i in range(len(sl_without_dc_test)):
            self.assertAlmostEqual(np.mean(sl_without_dc_test[i]['wav']), 0, 15)


if __name__ == '__main__':
    unittest.main()
