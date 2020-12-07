import numpy as np

from src.analyze import window
from src.classes.containers import FFTFrame, MelFrame


def mfcc(sound_wave):
    frames = window.window(sound_wave)



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

    return FFTFrame(y, window.fs)


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


def apply_mel_filterbank(fft_frame, bottom_frequency, top_frequency, n_filters):
    """
    Generates and applies mel filterbank onto a given frequency spectrum

    :param fft_frame: (FFTFrame) frame to be converted into mel coefficients
    :param bottom_frequency: (float) bottom frequency for the mel filter bank
    :param top_frequency: (float) top frequency for the mel filter bank
    :param n_filters: (int) number of mel filters
    :return: (MelFrame) mel coefficients of a given frame
    """

    spectrum = fft_frame.spectrum()
    filter_frequencies = get_filter_frequencies(bottom_frequency, top_frequency, n_filters)
    filter_bins = get_filter_bins(filter_frequencies, fft_frame.length(), fft_frame.fs)
    mel_frame = MelFrame(np.zeros(n_filters), n_filters=n_filters)

    for m in range(1, n_filters + 1):
        h = np.zeros(spectrum.size)
        for k in range(filter_bins[m - 1], filter_bins[m]):
            h[k] = (k - filter_bins[m - 1]) / (filter_bins[m] - filter_bins[m - 1])
        for k in range(filter_bins[m], filter_bins[m + 1]):
            h[k] = (filter_bins[m + 1] - k) / (filter_bins[m + 1] - filter_bins[m])

        mel_frame[m - 1] = np.sum(np.dot(spectrum, h))

    return mel_frame


def get_filter_frequencies(f_min, f_max, n_filters):
    """
    Returns filterbank base frequencies (n_filters + 2)

    :param f_min: minimum frequency for filterbank
    :param f_max: maximum frequency for filterbank
    :param n_filters: number of filters
    :param n_bins: number of spectrum bins
    :return: (1-D ndarray) filter frequencies in Hertz
    """

    mel_min = get_mel(f_min)
    mel_max = get_mel(f_max)
    filter_frequencies = np.linspace(mel_min, mel_max, n_filters + 2)
    filter_frequencies = np.array([get_frequency(f) for f in filter_frequencies])

    return filter_frequencies


def get_filter_bins(filter_freq, n_bins, fs):
    """
    Returns numbers of frequency bins for a specific fft_frame

    :param filter_freq:
    :param n_bins:
    :param fs:
    :return:
    """

    return np.array([int(np.round(f * n_bins / fs)) for f in filter_freq])


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


def logarithm(mel_filter_log):

    mel_filter_log.samples = np.log10(mel_filter_log.samples)

    return mel_filter_log