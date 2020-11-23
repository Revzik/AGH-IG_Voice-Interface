import unittest
from src.utils.sound_utils import SoundUtils
import numpy as np

class SoundUtilsTest(unittest.TestCase):


  def test_normalization(self):
        samples = [1,2,3,4,5,6]
        self.assertEqual(1, np.sqrt(np.mean(np.power(SoundUtils.normalize(samples), 2))))



if __name__ == '__main__':
    unittest.main()

