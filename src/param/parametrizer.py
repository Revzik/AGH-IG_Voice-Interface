import numpy as np

from src.classes.containers import Window
from src.conf import config


def window(sound_wave):
    """
    Splits the signal into frames using window length and window overlap specified in config
    The applied windowing function type is Hann

    :param sound_wave: (SoundWave) sound wave to be windowed
    :return: (List of Window) next frames from specified sound wave
    """

    # Load parameters from config
    win_len = int(config.analysis['window_length'] * sound_wave.fs / 1000)
    win_ovlap = int(config.analysis['window_overlap'] * sound_wave.fs / 1000)
    win_step = win_len - win_ovlap

    # Prepare data for windowing
    frames = []
    offset = 0
    win_fun = hann(win_len)

    # Chop the signal (and apply window) except last frame
    while offset < sound_wave.length() - win_len:
        frame = sound_wave.samples[offset:(offset + win_len)] * win_fun
        frames.append(Window(frame, sound_wave.fs))
        offset += win_step

    # Pad last frame with zeros
    raw_frame = sound_wave.samples[offset::]
    padded_frame = np.hstack((raw_frame, np.zeros(win_len - raw_frame.size))) * win_fun
    frames.append(Window(padded_frame, sound_wave.fs))

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
