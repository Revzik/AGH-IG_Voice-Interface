import numpy as np
from src.conf import config


def normalize(sound_wave):

    sound_wave.samples = sound_wave.samples / rms(sound_wave.samples)

    return sound_wave


def rms(samples):
    return np.sqrt(np.mean(np.power(samples, 2)))


def preemphasis(sound_wave):

    pre = config.analysis['preemphasis']
    sound_wave.samples = np.append(sound_wave.samples[0], sound_wave.samples[1:] - pre * sound_wave.samples[:-1])

    return sound_wave
