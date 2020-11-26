import numpy as np


def normalize(sound_wave):
    sound_wave.samples = sound_wave.samples / np.sqrt(np.mean(np.power(sound_wave.samples, 2)))

    return sound_wave
