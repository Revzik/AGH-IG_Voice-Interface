from src.conf import config
import math
import numpy as np
import scipy.signal as sig

class SoundLoader:
    def __init__(self):
        # tak się dobiera do ustalonych parametrów (wszystkie są w configurator.py)
        print(config.analysis['sampling_frequency'])

    def downsample(self, audio, fs):

        sampling_frequency = config.analysis['sampling_frequency']
        nww = (fs * sampling_frequency) / math.gcd(fs, sampling_frequency)
        up = int(nww // fs)
        down = int(nww // sampling_frequency)
        nww = (fs * sampling_frequency) / math.gcd(fs, sampling_frequency)
        up = int(nww // fs)
        down = int(nww // sampling_frequency)
        audio_up = np.zeros(len(audio) * up)  # zwiększamy sygnał
        audio_up[1::up] = audio
        filtr = sig.firwin(301, cutoff = sampling_frequency / 2, fs = fs * up)
        audio_up = up * sig.filtfilt(filtr, 1, audio_up)
        audio_down = audio_up[1::down]

        return audio_down