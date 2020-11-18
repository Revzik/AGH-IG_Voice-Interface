from src.conf import config
import numpy as np


class Mfcc:
    def __init__(self):
        pass

    def fft(self, x):
        """
        Computes FFT using recursive Cooley-Tukey algorithm
        :param x: (1-D numpy array) signal array to transform.
        If length is not a power of 2, then it gets padded with zeros.
        :return: (1-D complex numpy array) FFT of x
        """
        if x.size == 0:
            return x
        if x.size & (x.size - 1):
            target_length = first_power_of_2(x.size)
            x = np.concatenate((x, np.zeros(target_length - x.size)))

        return cooley_tukey(x)


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


if __name__ == '__main__':
    mfcc = Mfcc()
