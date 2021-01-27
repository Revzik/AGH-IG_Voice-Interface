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
        self.init_vad_threshold = config.analysis['vad_threshold']

    def tearDown(self):
        config.analysis['vad_threshold'] = self.init_vad_threshold

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

    def test_remove_dc_offset(self):
        sound_list = [
            SoundWave(1 + np.sin(8*np.pi*np.linspace(0, 1, 1000, False)), 1000, "a"),
            SoundWave(-0.5 + np.sin(2*np.pi*np.linspace(0, 1, 1000, False)), 1000, "b"),
            SoundWave(np.arange(0, 10), 1000, "c"),
            SoundWave(np.ones(100), 1000, "d")
        ]

        for sound_wave in sound_list:
            sound_wave = sound_utils.remove_dc_offset(sound_wave)

            self.assertAlmostEqual(np.mean(sound_wave.samples), 0, 15)


    def test_detect_speech(self):
        # Theoretical test on artificial speech sample with noise
        fs = 8000
        duration = 5
        t = np.linspace(0, duration, duration * fs)
        speech_samples = {
            'start': [10000, 25000, 36000],
            'end':   [20000, 30000, 39000]
        }

        # Generating noise
        sound_wave = np.random.randn(t.size) * 0.5

        # Generating speech-like signal
        for i in range(len(speech_samples['start'])):
            start = speech_samples['start'][i]
            end = speech_samples['end'][i]
            sound_wave[start:end] += np.cos(2 * np.pi * t[start:end] * 150) +\
                                     np.cos(2 * np.pi * t[start:end] * 450)

        if len(sound_wave.shape) == 2:
            sound_wave = sound_wave[..., 0]
        sound_wave = SoundWave(sound_wave, fs, 'test')
        sound_wave = sound_utils.normalize(sound_wave)
        chopped_sound_wave, flags = sound_utils.detect_speech(sound_wave)

        self.assertTrue(all(flags[130:240]))
        self.assertTrue(all(flags[320:370]))
        self.assertTrue(all(flags[460:480]))

        self.assertFalse(any(flags[0:110]))
        self.assertFalse(any(flags[260:300]))
        self.assertFalse(any(flags[390:440]))

        # # Test on a loaded sample
        # sound_wave, fs = soundfile.read('../../../Lab1 - slownik/6/do_przodu.wav')
        # speech_samples = {
        #     'start': [4088, 12196],
        #     'end':   [8778, 29820]
        # }

        # sound_wave, fs = soundfile.read('../../tmp/VAD.wav')
        # speech_samples = {
        #     'start': [31279, 94951,  111750, 182219, 497192, 628491],
        #     'end':   [50919, 106976, 130117, 362619, 544474, 656133]
        # }

        # # Plots
        # desired_flags = {
        #     'idx': [],
        #     'flag': []
        # }
        # desired_flags['idx'].append(0)
        # desired_flags['flag'].append(False)
        # for i in range(len(speech_samples['start'])):
        #     desired_flags['idx'].append(speech_samples['start'][i] - 1)
        #     desired_flags['flag'].append(False)
        #     desired_flags['idx'].append(speech_samples['start'][i])
        #     desired_flags['flag'].append(True)
        #     desired_flags['idx'].append(speech_samples['end'][i])
        #     desired_flags['flag'].append(True)
        #     desired_flags['idx'].append(speech_samples['end'][i] + 1)
        #     desired_flags['flag'].append(False)
        # desired_flags['idx'].append(len(speech_samples['start']) - 1)
        # desired_flags['flag'].append(False)
        #
        # fig, ax = plt.subplots()
        # t1 = np.linspace(0, sound_wave.length() / fs, sound_wave.length())
        # t2 = np.linspace(0, sound_wave.length() / fs, len(flags))
        # ax.plot(t1, sound_wave.samples, linewidth=0.5, label='sound wave')
        # ax.plot(t1[desired_flags['idx']], np.array(list(map(int, desired_flags['flag']))) * np.max(sound_wave.samples) * 1.2, label='desired VAD')
        # ax.plot(t2, np.array(list(map(int, flags))) * np.max(sound_wave.samples), label='VAD')
        # fig.legend(loc='lower left')
        # fig.show()


if __name__ == '__main__':
    unittest.main()
