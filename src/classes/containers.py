import numpy as np
import scipy.signal as sig
import math

from src.conf import config


class SoundWave:
    def __init__(self, samples=np.zeros(1), fs=44100, phrase=''):
        self.samples = samples
        self.fs = fs
        self.phrase = phrase

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