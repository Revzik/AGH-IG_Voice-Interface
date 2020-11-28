import numpy as np

from src.conf import config
from src.classes.containers import FFTFrame


def fft(window):
    """
    Computes FFT using recursive Cooley-Tukey algorithm

    :param window: (Window) signal array to transform.
    If length is not a power of 2, then it gets padded with zeros.
    :return: (FFTFrame) FFT of x
    """
    x = window.samples

    if x.size == 0:
        return x
    if x.size & (x.size - 1):
        target_length = first_power_of_2(x.size)
        x = np.concatenate((x, np.zeros(target_length - x.size)))

    y = cooley_tukey(x)
    df = get_df(y.size, window.fs)

    return FFTFrame(y, df)


def cooley_tukey(x):
    """
    Computes FFT using recursive Cooley-Tukey algorithm given x length is a power of 2

    :param x: (1-D numpy array) signal array to transform
    :return: (1-D complex numpy array) FFT of x
    """
    if x.size == 1:
        return x

    y_even = cooley_tukey(x[0::2])
    y_odd = cooley_tukey(x[1::2])

    mod = -2j * np.pi * np.arange(0, x.size/2) / x.size
    y_left = y_even + y_odd * np.exp(mod)
    y_right = y_even - y_odd * np.exp(mod)

    return np.concatenate((y_left, y_right))


def first_power_of_2(x):
    """
    Finds the first power of 2 above or equal to given number

    :param x: (1-D numpy array) given number (must be at least 1)
    :return: (int) the closest power of 2 above x
    """
    if x < 1:
        return None
    power = 1
    while power < x:
        power *= 2
    return power


def get_df(frame_length, fs):
    """
    Returns the df for FFTFrame

    :param frame_length: number of samples in a frame
    :param fs: sampling frequency
    :return: df
    """
    return fs / frame_length


def apply_mel_filterbank(fft_frame):
    spectrum = fft_frame.spectrum()
    filter_frequencies = get_filter_frequnecies(fft_frame)



def get_filter_frequnecies(f_max, n_filters):
    """
    Returns filterbank based on

    :param f_max: maximum frequency for filterbank
    :param n_filters: number of filters
    :return: (1-D ndarray) filter frequencies in Hertz
    """
    mel_min = 0
    mel_max = get_mel(f_max)
    filter_frequencies = np.linspace(mel_min, mel_max, n_filters)
    filter_frequencies = np.array([get_frequency(f) for f in filter_frequencies])

    return filter_frequencies


def get_mel(frequency):
    """
    Convert frequency to corresponding mel
    """
    return 2595 * np.log10(1 + frequency / 700)


def get_frequency(mel):
    """
    Convert mel to corresponding frequency
    """
    return 700 * (np.power(10, mel / 2595) - 1)
