import unittest
import numpy as np

from src.classes.containers import SoundWave
from src.utils import sound_loader as sl


class SoundLoaderTest(unittest.TestCase):
    def test_remove_dc_offset(self):
        sound_list = [
            SoundWave(1 + np.sin(8*np.pi*np.linspace(0, 1, 1000, False)), 1000, "a"),
            SoundWave(-0.5 + np.sin(2*np.pi*np.linspace(0, 1, 1000, False)), 1000, "b"),
            SoundWave(np.arange(0, 10), 1000, "c"),
            SoundWave(np.ones(100), 1000, "d")
        ]

        for sound_wave in sound_list:
            sound_wave = sl.remove_dc_offset(sound_wave)

            self.assertAlmostEqual(np.mean(sound_wave.samples), 0, 15)


if __name__ == '__main__':
    unittest.main()
