import numpy as np
from src.conf import config


def windowing(sound_wave, length_t, overlap_t):
    # defining the samples and needed lengths
    sampling_frequency = config.analysis['sampling_frequency']
    length_samples = int(round(length_t * sampling_frequency))
    overlap_samples = int(round(overlap_t * sampling_frequency))
    signal_length = len(sound_wave.samples)
    window_list = []
    pad_signal = []

    # checking how many windows we need in a signal
    window_quantity = int(np.ceil(float(np.abs(signal_length - length_samples)) / overlap_samples))

    # creating pad signal to add zeros to incomplete windows
    pad_signal_length = window_quantity * overlap_samples + length_samples
    zeros = np.zeros(pad_signal_length - signal_length)
    pad_signal.append(sound_wave.samples, zeros)

    indices = np.tile(np.arrange(0, length_samples), (window_quantity, 1)) + np.tile(
        np.arrange(0, window_quantity * overlap_samples, overlap_samples), (length_samples, 1))
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    for n in range(1, len(frames)):
        frames *= 0.54 - 0.46 * np.cos((2 * np.pi * n) / (length_samples - 1))
        window_list.append(frames)

    return window_list
