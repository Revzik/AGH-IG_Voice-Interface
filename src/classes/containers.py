import numpy as np

from src.conf import config


class SoundWave:
    def __init__(self, samples, fs=44100, phrase=''):
        self.samples = samples
        self.fs = fs
        self.phrase = phrase

    def __getitem__(self, i):
        return self.samples[i]

    def __setitem__(self, i, value):
        self.samples[i] = value

    def length(self):
        return self.samples.size


class Window:
    def __init__(self, samples, fs=None):
        self.samples = samples
        if fs is None:
            self.fs = config.analysis['sampling_frequency']
        else:
            self.fs = fs

    def __getitem__(self, i):
        return self.samples[i]

    def __setitem__(self, i, value):
        self.samples[i] = value

    def length(self):
        return self.samples.size


class FFTFrame:
    def __init__(self, samples, df=1):
        self.samples = samples
        self.df = df
        self.nyquist_frequency = df * samples.size / 2

    def __getitem__(self, i):
        return self.samples[i]

    def __setitem__(self, i, value):
        self.samples[i] = value

    def length(self):
        return self.samples.size

    def spectrum(self):
        return np.abs(self.samples[0:self.length() // 2])


class MelFrame:
    def __init__(self, samples, n_filters=None):
        self.samples = samples
        if n_filters is None:
            self.n_filters = config.analysis['filterbank_size']
        else:
            self.n_filters = n_filters

    def __getitem__(self, i):
        return self.samples[i]

    def __setitem__(self, i, value):
        self.samples[i] = value


class CepstralFrame:
    def __init__(self, samples):
        self.samples = samples

    def __getitem__(self, i):
        return self.samples[i]

    def __setitem__(self, i, value):
        self.samples[i] = value

    def length(self):
        return self.samples.size
