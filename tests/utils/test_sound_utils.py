import unittest
import numpy as np

from src.classes.containers import SoundWave
from src.utils import sound_utils


class SoundUtilsTest(unittest.TestCase):
    def test_normalization(self):
        sound_waves = [
            SoundWave(np.array([1, 2, 3, 4, 5, 6])),
            SoundWave(np.array([-1, -2, -3, -4, -5, -6]))
        ]

        sound_utils.normalize(sound_waves)

        for sound_wave in sound_waves:
            self.assertEqual(1, np.sqrt(np.mean(np.power(sound_wave.samples, 2))))


if __name__ == '__main__':
    unittest.main()

