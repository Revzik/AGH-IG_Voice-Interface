import numpy as np
import scipy.signal as sig
import math

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

    def get_length(self):
        return self.samples.size

    def downsample(self):
        sampling_frequency = config.analysis['sampling_frequency']

        # down and up factor
        last_common_multiple = (self.fs * sampling_frequency) / math.gcd(self.fs, sampling_frequency)
        upsample_factor = int(last_common_multiple // self.fs)
        downsample_factor = int(last_common_multiple // sampling_frequency)

        # upsampling
        audio_up = np.zeros(self.get_length() * upsample_factor)
        audio_up[upsample_factor // 2::upsample_factor] = self.samples

        # filtering
        alias_filter = sig.firwin(301, cutoff=sampling_frequency / 2, fs=self.fs * upsample_factor)
        audio_up = downsample_factor * sig.filtfilt(alias_filter, 1, audio_up)

        # downsampling
        audio_down = audio_up[downsample_factor // 2::downsample_factor]
        self.samples = audio_down

    def remove_dc(self):
        self.samples = self.samples - np.mean(self.samples)

    def normalize(self):
        self.samples = self.samples / np.sqrt(np.mean(np.power(self.samples, 2)))


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

    def get_length(self):
        return self.samples.size


class FFTFrame:
    def __init__(self, samples, df=1):
        self.samples = samples
        self.df = df

    def __getitem__(self, i):
        return self.samples[i]

    def __setitem__(self, i, value):
        self.samples[i] = value

    def get_length(self):
        return self.samples.size

    def spectrum(self):
        return np.abs(self.samples[0:self.get_length() // 2])


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

    def get_length(self):
        return self.samples.size


class CepstralFrame:
    def __init__(self, samples):
        self.samples = samples

    def __getitem__(self, i):
        return self.samples[i]

    def __setitem__(self, i, value):
        self.samples[i] = value

    def get_length(self):
        return self.samples.size
