import unittest
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import soundfile

from src.conf import config
from src.classes.containers import SoundWave
from src.utils import sound_utils


class SoundUtilsTest(unittest.TestCase):
    def setUp(self):
        self.init_e_threshold = config.analysis['energy_threshold']
        self.init_f_threshold = config.analysis['f_threshold']
        self.init_s_threshold = config.analysis['sf_threshold']

    def tearDown(self):
        config.analysis['energy_threshold'] = self.init_e_threshold
        config.analysis['f_threshold'] = self.init_f_threshold
        config.analysis['sf_threshold'] = self.init_s_threshold

    def test_normalization(self):
        sound_waves = [
            SoundWave(np.array([1, 2, 3, 4, 5, 6])),
            SoundWave(np.array([-1, -2, -3, -4, -5, -6]))
        ]

        for sound_wave in sound_waves:
            sound_wave = sound_utils.normalize(sound_wave)

            self.assertEqual(1, np.sqrt(np.mean(np.power(sound_wave.samples, 2))))

    def test_preemphasis(self):

        # fig, axs = plt.subplots(1, 3)
        sound_wave = SoundWave(sig.unit_impulse(100, 'mid'))

        # axs[0].plot(sound_wave.samples)
        # axs[0].set_title('Delta function')

        spectrum_sound_wave = np.abs(np.fft.fft(sound_wave.samples))
        spectrum_sound_wave_after_pre = np.abs(np.fft.fft(sound_utils.preemphasis(sound_wave).samples))

        # axs[1].plot(spectrum_sound_wave)
        # axs[1].set_title('Delta function spectrum')
        # axs[2].plot(spectrum_sound_wave_after_pre)
        # axs[2].set_title('Delta function spectrum after preemphasis')
        # plt.show()

        self.assertLess(spectrum_sound_wave_after_pre[0], spectrum_sound_wave[0])
        self.assertLess(spectrum_sound_wave_after_pre[0], spectrum_sound_wave_after_pre[int(sound_wave.length()/2)])

    def test_detect_speech(self):
        # Test on a loaded sample
        sound_wave, fs = soundfile.read('../../../Lab1 - slownik/6/do_przodu.wav')
        # sound_wave, fs = soundfile.read('../../tmp/VAD.wav')
        if len(sound_wave.shape) == 2:
            sound_wave = sound_wave[..., 0]
        sound_wave = SoundWave(sound_wave, fs, 'do_przodu')
        sound_wave = sound_utils.normalize(sound_wave)

        frame_len = int(0.01 * fs)
        chopped_sound_wave, flags = sound_utils.detect_speech(sound_wave)

        fig, ax = plt.subplots()
        t1 = np.linspace(0, sound_wave.length() / fs, sound_wave.length())
        ax.plot(t1, sound_wave.samples)
        t2 = np.linspace(0, sound_wave.length() / fs, len(flags))
        ax.plot(t2, np.array(list(map(int, flags))) * np.max(sound_wave.samples))
        fig.show()

        # Theoretical test on artificial speech sample with noise

if __name__ == '__main__':
    unittest.main()
