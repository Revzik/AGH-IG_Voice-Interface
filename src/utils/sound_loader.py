import os
import glob
import soundfile as sf
import scipy.signal as sig
import math
import numpy as np

from src.classes.containers import SoundWave
from src.conf import config


def load_sound_file():
    sound_list = []
    print("Please enter the path for the folder with waves: ")
    paths = input()
    for i, filename in enumerate(glob.glob(os.path.join(paths, '*.wav'))):
        wav_data, fs = sf.read(filename)
        sound_list.append(SoundWave(wav_data, fs, os.listdir(paths)[i][:-4]))

    return sound_list


def downsample(sound_wave):
    sampling_frequency = config.analysis['sampling_frequency']

    # down and up factor
    last_common_multiple = (sound_wave.fs * sampling_frequency) / math.gcd(sound_wave.fs, sampling_frequency)
    upsample_factor = int(last_common_multiple // sound_wave.fs)
    downsample_factor = int(last_common_multiple // sampling_frequency)

    # upsampling
    audio_up = np.zeros(sound_wave.length() * upsample_factor)
    audio_up[upsample_factor // 2::upsample_factor] = sound_wave.samples

    # filtering
    alias_filter = sig.firwin(301, cutoff=sampling_frequency / 2, fs=sound_wave.fs * upsample_factor)
    audio_up = downsample_factor * sig.filtfilt(alias_filter, 1, audio_up)

    # downsampling
    audio_down = audio_up[downsample_factor // 2::downsample_factor]
    sound_wave.samples = audio_down
    sound_wave.fs = sampling_frequency

    return sound_wave


def remove_dc_offset(sound_wave):
    sound_wave.samples = sound_wave.samples - np.mean(sound_wave.samples)

    return sound_wave
