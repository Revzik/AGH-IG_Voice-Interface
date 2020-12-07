import unittest
import matplotlib.pyplot as plt

from src.classes.containers import *
from src.conf import config
from src.analyze import window
from src.utils import sound_utils


class ParametrizerTest(unittest.TestCase):
    def test_windowing(self):
        config.analysis['window_length'] = 20
        config.analysis['window_overlap'] = 10
        fs = 8000
        time = 0.1
        wav_data = np.arange(0, time, 1 / fs)

        sound = SoundWave(wav_data, fs, "naprzod")

        frames = window.window(sound)

        self.assertTrue(all([frames[0].length() == f.length() for f in frames]))

        # For an ascending linear function every next window should have higher energy
        for i in range(len(frames) - 1):
            self.assertGreater(sound_utils.rms(frames[i + 1].samples), sound_utils.rms(frames[i].samples))

        # Plots
        # win_len = int(config.analysis['window_length'] * fs / 1000)
        # win_ovlap = int(config.analysis['window_overlap'] * fs / 1000)
        #
        # t = np.linspace(0, time, sound.length())
        # win_step = win_len - win_ovlap
        #
        # fig, axes = plt.subplots(2, 1, figsize=(8, 6))
        # axes[0].plot(t, sound.samples)
        # axes[0].grid()
        # axes[0].set_title("raw")
        #
        # offset = 0
        # for i, frame in enumerate(frames):
        #     if i == len(frames) - 1:
        #         t = np.linspace(t[offset], t[offset] + win_len / fs, win_len)
        #         axes[1].plot(t, frame.samples)
        #     else:
        #         axes[1].plot(t[offset:(offset + win_len)], frame.samples, alpha=0.5)
        #     offset += win_step
        # axes[1].grid()
        # axes[1].set_title("windowed")
        # axes[1].set_xlabel("t [s]")
        #
        # fig.show()

    def test_hann(self):
        win_len = 1000
        win_fun = window.hann(win_len)

        self.assertAlmostEqual(0, win_fun[0], 3)
        self.assertAlmostEqual(0, win_fun[win_len - 1], 3)
        self.assertAlmostEqual(1, win_fun[win_len // 2], 3)

        # Plots
        # fig, ax = plt.subplots(figsize=(8, 6))
        # ax.plot(win_fun)
        # fig.show()

    def tearDown(self):
        config.reload()


if __name__ == '__main__':
    unittest.main()
