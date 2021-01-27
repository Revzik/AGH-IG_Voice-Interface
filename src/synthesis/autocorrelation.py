import numpy as np
from scipy.signal import argrelextrema
from src.conf import config


def acorr(window, fs = config.analysis['sampling_frequency'], fmin = config.analysis['fmin'], fmax = config.analysis['fmax'], debug=False):

    # two the same signal (autocorrelation)
    x = window
    y = window

    # Size of the autocorrelation vector
    N = x.size + y.size - 1

    # Initialize vector
    ac = np.zeros(N)

    #add zeros
    begin = np.zeros(y.size - 1)
    x = np.append(begin, x)
    x = np.append(x, begin)

    for i in range(N):
        ac[i] += np.sum(x[i: i + y.size] * y)  # autocorrelation

    #find max in autocorrelation vectro
    # + value
    start = len(y)
    ac = ac[start:]
    # find max value
    max_ac = argrelextrema(ac, np.greater)
    max_ac_index = max_ac[0][0]
    delays = np.arange(1, ac.size) / fs
    tau = delays[max_ac_index]
    #
    if (1/tau >= fmin) and (1/tau <= fmax):
        tonality = True
    else:
        tonality = False
    if not debug:
        return tonality
    else:
        return tonality, delays, ac, max_ac_index
