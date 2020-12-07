import numpy as np
import scipy.signal as sig
import math
import numpy as np

from src.classes.containers import SoundWave
from src.conf import config
from src.analyze import window


def normalize(sound_wave):

    sound_wave.samples = sound_wave.samples / rms(sound_wave.samples)

    return sound_wave


def rms(samples):
    return np.sqrt(np.mean(np.power(samples, 2)))


def preemphasis(sound_wave):

    pre = config.analysis['preemphasis']
    sound_wave.samples = np.append(sound_wave.samples[0], sound_wave.samples[1:] - pre * sound_wave.samples[:-1])

    return sound_wave


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



def detect_speech(sound_wave):
    """
    Cuts the noise out of the sound wave using VAD.

    :param sound_wave: (SoundWave) sound wave to be evaluated
    :return: (SoundWave) input with just speech frames, (List of boolean) VAD output
    """

    threshold = config.analysis['vad_threshold']

    frames = window.split(sound_wave, 10, 0)
    flags = [False] * len(frames)

    # Detect speech based on variance difference between noise frames and speech frames
    noise_variance = get_average_variance(frames[0:5])

    for i in range(2, len(frames) - 2):
        if get_average_variance(frames[i - 2:i + 2]) - noise_variance > threshold:
            flags[i] = True

    # remove too short noise fragments
    noise_index = 0
    for i in range(len(flags) - 1):
        if flags[i] and not flags[i + 1]:
            noise_index = i
        if not flags[i] and flags[i + 1] and i - noise_index < 10:
            for j in range(noise_index, i + 1):
                flags[j] = True

    # remove too short speech fragments
    speech_index = 0
    for i in range(len(flags) - 1):
        if not flags[i] and flags[i + 1]:
            speech_index = i
        if flags[i] and not flags[i + 1] and i - speech_index < 5:
            for j in range(speech_index, i + 1):
                flags[j] = False

    # Split the signal according to the VAD
    new_frames = []
    for i, frame in enumerate(frames):
        if flags[i]:
            new_frames.append(frame)

    return SoundWave(np.array(new_frames), sound_wave.fs, sound_wave.phrase), flags


def get_average_variance(frames):
    """
    Calculates the average variance in specified frames

    :param frames: (List of 1-D ndarrays) input frames
    :return: (Float) mean variance
    """
    variance = 0
    for frame in frames:
        variance += np.var(frame)
    return variance / len(frames)
