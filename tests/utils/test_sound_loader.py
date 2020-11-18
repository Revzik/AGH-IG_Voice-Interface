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
        mean_test = np.mean(Sl.remove_dc_offset(self, test_soundlist)[:]['wav'])
        for i in range(len(mean_test)):
            self.assertGreater(mean_test[i], 10**-14)


if __name__ == '__main__':
    unittest.main()
