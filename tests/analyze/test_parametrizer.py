import unittest
import numpy as np

from src.classes.containers import SoundWave
from src.analyze import parametrizer
from src.conf import config


class ParametrizerTest(unittest.TestCase):
    def test_parametrize(self):
        fs = 44100
        dur = 5
        t = np.linspace(0, dur, dur * fs)
        x = np.random.randn(t.size) + np.cos(2 * np.pi * t * 150)
        sound_wave = SoundWave(x, fs, 'test')

        cepstrum, phrase = parametrizer.parametrize(sound_wave)

        self.assertEqual(499, cepstrum.shape[0])
        self.assertEqual('test', phrase)
        self.assertEqual(config.analysis['filterbank_size'], cepstrum.shape[1])
