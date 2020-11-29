import numpy as np
from src.conf import config


def normalize(sound_wave):

    sound_wave.samples = sound_wave.samples / np.sqrt(np.mean(np.power(sound_wave.samples, 2)))

    return sound_wave


def preemphasis(sound_wave):

    pre = config.analysis['preemphasis']
    sound_wave.samples = np.append(sound_wave.samples[0], sound_wave.samples[1:] - pre * sound_wave.samples[:-1])

    return sound_wave
