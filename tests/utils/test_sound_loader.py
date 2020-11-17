import unittest
import matplotlib.pyplot as plt
from src.utils.sound_loader import SoundLoader as Sl


class SoundLoaderTest(unittest.TestCase):
    # tutaj można pisać testy do metod z Mfcc (na wzór test_configurator.py)
    def test_sound_load_file(self):

        result = Sl.sound_load_file()
        print(len(result))
        print(result)
        plt.plot(result[1]['wav'])
        plt.show()
        pass

if __name__ == '__main__':
    unittest.main()
