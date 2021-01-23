import numpy as np
from src.conf import config
import operator
import math

def acorr(window, fs = config.analysis['sampling_frequency'], fmin = config.analysis['fmin'], fmax = config.analysis['fmax']):

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
    max_ac_index, max_ac_value = max(enumerate(ac), key = operator.itemgetter(1))
    delays = np.arange((-len(y) + 1), len(x)) / fs

    # + delays
    delays = delays[len(y):]

    tau = delays[max_ac_index] / fs
    tonality = None

    if (1/tau >= fmin) and (1/tau <= fmax):
        tonality = True
    else:
        tonality = False

    return tonality

