import numpy as np

from src.conf import config


def window(sound_wave):
    """
    Splits the signal into frames using window length and window overlap specified in config
    The applied windowing function type is Hann

    :param sound_wave: (SoundWave) sound wave to be windowed
    :return: (List of 1-D ndarrays) windowed signal with Hann windows
    """

    length = config.analysis['window_length']
    overlap = config.analysis['window_overlap']

    frames = split(sound_wave, length, overlap)

    win_len = frames[0].size
    win_fun = hann(win_len)

    windows = []
    for frame in frames:
        windows.append(frame * win_fun)

    return windows


def split(sound_wave, length, overlap):
    """
    Splits the signal into frames with rectangular window

    :param sound_wave: (SoundWave) sound wave to be windowed
    :param length: (float) length of window (ms)
    :param overlap: (float) length of window overlap (ms)
    :return: (List of 1-D ndarrays) next frames from specified sound wave
    """

    # Load parameters from config
    win_len = int(length * sound_wave.fs / 1000)
    win_ovlap = int(overlap * sound_wave.fs / 1000)
    win_step = win_len - win_ovlap

    # Prepare data for windowing
    frames = []
    offset = 0

    # Chop the signal except last frame
    while offset < sound_wave.length() - win_len:
        frame = sound_wave.samples[offset:(offset + win_len)]
        frames.append(frame)
        offset += win_step

    # Pad last frame with zeros
    raw_frame = sound_wave.samples[offset::]
    padded_frame = np.hstack((raw_frame, np.zeros(win_len - raw_frame.size)))
    frames.append(padded_frame)

    return frames


def hann(N):
    """
    Computes samples for Hann window of lenght N

    :param N: (int) length of Hann window
    :return: (1-D ndarray) Hann window samples
    """
    a0 = 0.5
    a1 = 1 - a0
    return a0 - a1 * np.cos(2 * np.pi * np.arange(N) / N)
